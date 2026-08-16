"""DSL executor (§2.5) — computes whitelist ops over ALL rows.

The LLM never writes freeform formulas; every op is validated against
`shared.dsl_validator.WHITELIST` before execution. Python aggregates on the
full dataset (sampling is for LLM/UX inspection only) and every computed
value is memorialized in the evidence registry.

Growth semantics (handoff row 1): `basis` defaults to `previous_period`;
`over_column` alone means month-basis (MoM). Value = (current - baseline) /
baseline; `as_percent: true` multiplies by 100. The KPI scalar for growth is
the latest period's growth; full series live in the trend suite (Step 1.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from analysis.evidence import EvidenceRegistry
from shared.dsl_validator import validate_operation
from shared.schemas import DslOperation, KpiCandidate, KpiResult

_AGG_FUNCTIONS = {"sum", "mean", "median", "min", "max", "std", "count",
                  "nunique"}

_PERIOD_ALIASES = {
    "month": "MoM", "monthly": "MoM", "mom": "MoM",
    "week": "WoW", "weekly": "WoW", "wow": "WoW",
}


@dataclass
class OperationResult:
    """Typed result of one DSL operation execution."""
    value: Any                                    # scalar | {group: scalar}
    aggregation: str
    comparison: Optional[str] = None
    filter_str: Optional[str] = None
    growth_series: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def apply_filter(df: pd.DataFrame, flt: Dict[str, Any] | None) -> pd.DataFrame:
    """Apply a DSL filter: {col: value} equality, {col: [v1, v2]} membership.

    Multiple keys are AND-ed. Missing columns raise (caller decides).
    """
    if not flt:
        return df
    filtered = df
    for column, value in flt.items():
        if column not in filtered.columns:
            raise ValueError(f"filter column '{column}' not in data")
        if isinstance(value, (list, tuple, set)):
            filtered = filtered[filtered[column].isin(value)]
        else:
            filtered = filtered[filtered[column] == value]
    return filtered


def filter_repr(flt: Dict[str, Any] | None) -> Optional[str]:
    """Human-readable filter string for evidence lineage."""
    if not flt:
        return None
    parts = []
    for column, value in flt.items():
        if isinstance(value, (list, tuple, set)):
            parts.append(f"{column} in [{', '.join(str(v) for v in value)}]")
        else:
            parts.append(f"{column}=={value}")
    return " & ".join(parts)


# ---------------------------------------------------------------------------
# Single-operation execution (pure)
# ---------------------------------------------------------------------------


def _as_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"column '{column}' not in data")
    return pd.to_numeric(df[column], errors="coerce")


def _aggregate_value(series: pd.Series, function: str) -> Any:
    if function == "sum":
        return float(series.sum())
    if function == "mean":
        return float(series.mean())
    if function == "median":
        return float(series.median())
    if function == "min":
        return series.min()
    if function == "max":
        return series.max()
    if function == "std":
        return float(series.std())
    if function == "count":
        return int(series.count())
    if function == "nunique":
        return int(series.nunique())
    raise ValueError(f"unknown aggregate '{function}'")


def _execute_aggregate(df: pd.DataFrame, op: DslOperation) -> OperationResult:
    group_by = list(op.group_by or [])
    flt = op.filter
    filtered = apply_filter(df, flt)
    column = op.column
    if column not in filtered.columns:
        raise ValueError(f"column '{column}' not in data")

    if op.function in {"min", "max", "count", "nunique"}:
        series = filtered[column]
    else:
        series = pd.to_numeric(filtered[column], errors="coerce")

    if not group_by:
        value = _aggregate_value(series, op.function)
        return OperationResult(
            value=_to_json_scalar(value), aggregation=op.function,
            filter_str=filter_repr(flt))

    grouped = filtered.groupby(group_by, dropna=False)[column]
    if op.function == "count":
        values = grouped.count()
    elif op.function == "nunique":
        values = grouped.nunique()
    elif op.function == "sum":
        values = grouped.sum()
    elif op.function == "mean":
        values = grouped.mean()
    elif op.function == "median":
        values = grouped.median()
    elif op.function == "std":
        values = grouped.std()
    else:  # min / max
        values = grouped.min() if op.function == "min" else grouped.max()

    result: Dict[str, Any] = {}
    for key, val in values.items():
        label = _group_label(key)
        result[label] = _to_json_scalar(val)
    return OperationResult(
        value=result, aggregation=_grouped_aggregation(group_by, op.function),
        filter_str=filter_repr(flt))


def _group_label(key: Any) -> str:
    if isinstance(key, tuple):
        return " | ".join(_to_json_scalar(k) for k in key)
    return str(_to_json_scalar(key))


def _grouped_aggregation(group_by: List[str], function: str) -> str:
    return f"{'_'.join(group_by)}_{function}"


def _to_json_scalar(value: Any) -> Any:
    import pandas as pd
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def _execute_correlation(df: pd.DataFrame, op: DslOperation) -> OperationResult:
    a = _as_numeric(df, op.column_a)
    b = _as_numeric(df, op.column_b)
    flt = op.filter
    if flt:
        mask = pd.Series(True, index=df.index)
        for column, value in flt.items():
            if column not in df.columns:
                raise ValueError(f"filter column '{column}' not in data")
            if isinstance(value, (list, tuple, set)):
                mask &= df[column].isin(value)
            else:
                mask &= df[column] == value
        a = a[mask]
        b = b[mask]
    valid = a.notna() & b.notna()
    a, b = a[valid], b[valid]
    if len(a) < 3:
        raise ValueError("correlation needs >= 3 valid pairs")
    method = op.method or "pearson"
    r = a.corr(b, method=method)
    if r is None or pd.isna(r):
        raise ValueError("correlation produced NaN")
    return OperationResult(value=float(r), aggregation="correlation",
                           filter_str=filter_repr(op.filter),
                           comparison=method)


def _execute_ratio(df: pd.DataFrame, op: DslOperation) -> OperationResult:
    if op.numerator is None or op.denominator is None:
        raise ValueError("ratio needs numerator and denominator")
    num = execute_operation(df, op.numerator)
    den = execute_operation(df, op.denominator)
    if num.value is None or den.value is None:
        raise ValueError("ratio operand failed")
    try:
        value = num.value / den.value
    except (ZeroDivisionError, TypeError):
        raise ValueError("ratio division failed")
    return OperationResult(value=float(value), aggregation="ratio",
                           filter_str=filter_repr(op.filter))


# ---------------------------------------------------------------------------
# Growth
# ---------------------------------------------------------------------------


def _period_key(series: pd.Series, period: str):
    """Bucket a datetime series; return a tuple key + human label.

    Unparseable dates (NaT) yield (None, None) and are skipped by the caller.
    """
    if period == "YoY":
        keys = [(ts.year, ts.month) if not pd.isna(ts) else None
                for ts in series]
        labels = [f"{ts.year:04d}-{ts.month:02d}" if not pd.isna(ts) else None
                  for ts in series]
        return keys, labels
    if period == "WoW":
        iso = series.dt.isocalendar()
        keys = [(row.year, row.week) if not pd.isna(series.iloc[i])
                else None for i, (_, row) in enumerate(iso.iterrows())]
        labels = [f"{row.year:04d}-W{row.week:02d}"
                  if not pd.isna(series.iloc[i]) else None
                  for i, (_, row) in enumerate(iso.iterrows())]
        return keys, labels
    # MoM (default)
    keys = [(ts.year, ts.month) if not pd.isna(ts) else None
            for ts in series]
    labels = [f"{ts.year:04d}-{ts.month:02d}" if not pd.isna(ts) else None
              for ts in series]
    return keys, labels


def _execute_growth(df: pd.DataFrame, op: DslOperation) -> OperationResult:
    column = op.column
    over = op.over_column
    if column not in df.columns:
        raise ValueError(f"column '{column}' not in data")
    if over not in df.columns:
        raise ValueError(f"over_column '{over}' not in data")

    flt = op.filter
    filtered = apply_filter(df, flt)
    over_values = pd.to_datetime(filtered[over], errors="coerce")
    period = _resolve_period(op.period)

    series_list: List[Dict[str, Any]] = []
    value: Any

    group_by = list(op.group_by or [])
    if group_by:
        out: Dict[str, Any] = {}
        for group_key, group_df in filtered.groupby(group_by, dropna=False):
            label = _group_label(group_key)
            period_value = _growth_for_group(
                group_df, column, over_values.loc[group_df.index], period, op)
            if period_value is not None:
                out[label] = period_value
            for row in _growth_series_for_group(
                    group_df, column, over_values.loc[group_df.index],
                    period, op):
                series_list.append({"group": label, **row})
        value = out
    else:
        value = _growth_for_group(filtered, column, over_values, period, op)
        series_list = _growth_series_for_group(filtered, column, over_values,
                                               period, op)

    return OperationResult(
        value=_to_json_scalar(value),
        aggregation=f"growth_{period}",
        comparison=period,
        filter_str=filter_repr(flt),
        growth_series=series_list)


def growth_series(df: pd.DataFrame, op: DslOperation) -> List[Dict[str, Any]]:
    """Public access to the deterministic growth series (used by the SVG
    chart renderer — the KPI scalar alone is not enough to draw a line)."""
    return _execute_growth(df, op).growth_series


def grouped_values(df: pd.DataFrame, op: DslOperation
                   ) -> List[Tuple[str, Any]]:
    """Deterministic (label, value) pairs for grouped aggregate operations —
    used by the SVG chart renderer (KPI results are expanded per group)."""
    result = _execute_aggregate(df, op)
    if isinstance(result.value, dict):
        return sorted((str(k), v) for k, v in result.value.items())
    return []


def grouped_growth_values(df: pd.DataFrame, op: DslOperation
                          ) -> List[Tuple[str, Any]]:
    """Deterministic {group: latest-period value} pairs for grouped growth
    operations — used by the SVG chart renderer (line/area per group)."""
    result = _execute_growth(df, op)
    if isinstance(result.value, dict):
        return sorted((str(k), v) for k, v in result.value.items())
    return []


def _resolve_period(period: Optional[str]) -> str:
    if not period:
        return "MoM"
    key = period.lower()
    if key in _PERIOD_ALIASES:
        return _PERIOD_ALIASES[key]
    if period in ("YoY", "MoM", "WoW"):
        return period
    raise ValueError(f"unknown growth period '{period}'")


def _bucket_aggregate(df: pd.DataFrame, column: str,
                      over_values: pd.Series, period: str):
    """Map period key -> sum of column within that bucket (sorted)."""
    keys, labels = _period_key(over_values, period)
    sums: Dict[Any, float] = {}
    for index, (key, label, (_, row)) in enumerate(
            zip(keys, labels, df.iterrows())):
        if key is None:
            continue
        value = pd.to_numeric(row[column], errors="coerce")
        if pd.isna(value):
            continue
        sums.setdefault(key, 0.0)
        sums[key] += float(value)
    return {key: sums[key] for key in sorted(sums)}, labels


def _growth_series_for_group(df: pd.DataFrame, column: str,
                             over_values: pd.Series, period: str,
                             op: DslOperation) -> List[Dict[str, Any]]:
    buckets, _ = _bucket_aggregate(df, column, over_values, period)
    order = sorted(buckets)
    as_percent = bool(op.as_percent)
    rows: List[Dict[str, Any]] = []
    for index, key in enumerate(order):
        current = buckets[key]
        baseline = _baseline(index, key, buckets, order, period, op.basis)
        if baseline is None or baseline == 0:
            continue
        growth = (current - baseline) / baseline
        if as_percent:
            growth *= 100
        rows.append({"period": _label_for_key(key, period),
                     "value": _to_json_scalar(growth)})
    return rows


def _growth_for_group(df: pd.DataFrame, column: str, over_values: pd.Series,
                      period: str, op: DslOperation) -> Optional[float]:
    rows = _growth_series_for_group(df, column, over_values, period, op)
    if not rows:
        return None
    as_percent = bool(op.as_percent)
    latest = rows[-1]["value"]
    return float(latest) if latest is not None else None


def _baseline(index: int, key: Any, buckets: Dict[Any, float],
              order: List[Any], period: str,
              basis: Optional[str]) -> Optional[float]:
    effective = basis or "previous_period"
    if effective == "start_of_period":
        return buckets[order[0]]
    if period == "YoY":
        year, month = key
        prev_key = (year - 1, month)
        return buckets.get(prev_key)
    if index == 0:
        return None
    return buckets[order[index - 1]]


def _label_for_key(key: Any, period: str) -> str:
    if period == "WoW":
        return f"{key[0]:04d}-W{key[1]:02d}"
    return f"{key[0]:04d}-{key[1]:02d}"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def execute_operation(df: pd.DataFrame,
                      op: DslOperation | Dict[str, Any]) -> OperationResult:
    """Validate + execute one DSL operation on the full dataset (pure)."""
    if isinstance(op, dict):
        op = DslOperation(**op)
    errors = validate_operation(op)
    if errors:
        raise ValueError(f"invalid DSL operation: {'; '.join(errors)}")

    if op.function in _AGG_FUNCTIONS:
        return _execute_aggregate(df, op)
    if op.function == "correlation":
        return _execute_correlation(df, op)
    if op.function == "growth":
        return _execute_growth(df, op)
    if op.function == "ratio":
        return _execute_ratio(df, op)
    raise ValueError(f"unhandled DSL function '{op.function}'")


def execute_kpi(df: pd.DataFrame, candidate: KpiCandidate,
                registry: EvidenceRegistry,
                kpi_id_prefix: Optional[str] = None) -> List[KpiResult]:
    """Run one plan KPI; returns one KpiResult per value.

    Aggregates/growth with `group_by` produce one row per group; every row
    carries a freshly minted evidence_id registered in the registry.
    """
    op = candidate.operation
    base_id = kpi_id_prefix or candidate.kpi_id
    try:
        result = execute_operation(df, op)
    except Exception as exc:  # noqa: BLE001 -- failed ops yield value=None
        evidence_id = registry.add_value(
            None, aggregation=str(getattr(op, "function", "?")),
            comparison=getattr(op, "period", None),
            filter_str=filter_repr(op.filter))
        return [KpiResult(kpi_id=base_id, name=candidate.name,
                          operation=op, value=None, evidence_id=evidence_id,
                          computed_by="pandas")]

    if isinstance(result.value, dict):
        rows: List[KpiResult] = []
        for index, (label, value) in enumerate(sorted(result.value.items())):
            evidence_id = registry.add_value(
                value, aggregation=result.aggregation,
                comparison=result.comparison,
                filter_str=result.filter_str)
            rows.append(KpiResult(
                kpi_id=f"{base_id}-{index + 1:03d}",
                name=f"{candidate.name} [{label}]",
                operation=op, value=value, evidence_id=evidence_id,
                computed_by="pandas"))
        if not rows:
            evidence_id = registry.add_value(
                None, aggregation=result.aggregation,
                comparison=result.comparison,
                filter_str=result.filter_str)
            rows.append(KpiResult(kpi_id=base_id, name=candidate.name,
                                  operation=op, value=None,
                                  evidence_id=evidence_id,
                                  computed_by="pandas"))
        return rows

    evidence_id = registry.add_value(
        result.value, aggregation=result.aggregation,
        comparison=result.comparison, filter_str=result.filter_str)
    return [KpiResult(kpi_id=base_id, name=candidate.name, operation=op,
                      value=result.value, evidence_id=evidence_id,
                      computed_by="pandas")]


def execute_plan(df: pd.DataFrame, plan,
                 registry: EvidenceRegistry) -> List[KpiResult]:
    """Run every candidate KPI in an AnalysisPlan (or dict with candidate_kpis)."""
    if hasattr(plan, "candidate_kpis"):
        candidates = plan.candidate_kpis
    else:
        candidates = plan["candidate_kpis"]
    results: List[KpiResult] = []
    for candidate in candidates:
        if not isinstance(candidate, KpiCandidate):
            candidate = KpiCandidate(**candidate)
        results.extend(execute_kpi(df, candidate, registry))
    return results
