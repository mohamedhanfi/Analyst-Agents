"""Data-shape chart planner (§2.5) — deterministic, no fixed menu.

Inspects each KPI candidate + the numeric/ordinal columns and picks a chart
kind from the ordered rule table. The LLM only decides *which facts deserve a
chart* and may re-rank; the shape is always Python's call. Output is
ChartMetadata entries (one per chart) — the actual drawing happens in stage 5b.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, get_args

import pandas as pd

from analysis.dsl_executor import growth_series
from analysis.evidence import EvidenceRegistry
from analysis.generic._helpers import measure_columns
from shared.schemas import (AnalysisPlan, ChartKind, ChartMetadata,
                            DatasetUnderstanding, Insight, KpiCandidate)

THIN_THRESHOLD = 10          # rows below this => downgrade + reliability low_n
RULE = {                     # rule-table reference (spec §2.5)
    "rule_1": "rule_1: single dimension, <= 2 values swap -> bar/donut",
    "rule_2": "rule_2: ordered axis (dates/months) with >= 3 points -> line",
    "rule_3": "rule_3: single dimension, 3-12 values -> vertical bar",
    "rule_4": "rule_4: single dimension, 13-50 values -> horizontal bar",
    "rule_5": "rule_5: single dimension, > 50 values -> barh top-15 + '$rest'",
    "rule_6": "rule_6: numeric distribution / skew ask -> histogram",
    "rule_7": "rule_7: 2 numeric measures, asked together -> scatter + trend",
    "rule_8": "rule_8: >= 3 numeric measures -> ranked correlation heatmap",
    "rule_9": "rule_9: share / '% of whole', parts ~ 100% -> doughnut",
    "rule_10": "rule_10: ranked contribution (sum/count over dimension) "
               "3-15 values -> pareto + cumulative %",
    "rule_11": "rule_11: growth KPI over time -> waterfall of period "
               "contributions (variance view)",
}

CHART_KINDS: Tuple[str, ...] = get_args(ChartKind)
_SHARE_RE = re.compile(r"share|%\b|\bof\s+whole|\bpart\b", re.IGNORECASE)
_AGGREGATES = {"sum", "mean", "median", "count", "nunique", "min", "max",
               "std", "ratio"}
_TOP_N_ROLLUP = 15


def _thin(df: pd.DataFrame, threshold: int) -> bool:
    return len(df) < threshold


def _register(registry: EvidenceRegistry, chart: ChartMetadata) -> None:
    registry.add(chart.evidence_id, {
        "file_hash": registry.file_hash,
        "sheet": registry.sheet,
        "transformations": registry.transformations,
        "aggregation": "chart",
        "comparison": chart.kind,
        "result": chart.title or chart.reason,
    })


def _downgrade_low_n(chart: ChartMetadata, why: str) -> ChartMetadata:
    """Downgrade a too-thin chart to a simple bar + reliability stamp."""
    chart.kind = "bar"
    chart.reliability = "low_n"
    chart.reason = f"low_n ({why}): {chart.reason}"
    return chart


def _dedupe(candidates: List[ChartMetadata]) -> List[ChartMetadata]:
    seen = set()
    unique: List[ChartMetadata] = []
    for chart in candidates:
        key = (chart.kind, tuple(sorted(chart.columns)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(chart)
    return unique


# ---------------------------------------------------------------------------
# Rule selection
# ---------------------------------------------------------------------------

def _plan_kpi(df: pd.DataFrame, kpi: KpiCandidate,
              registry: EvidenceRegistry, index: int,
              threshold: int,
              proposal: Dict[str, str] | None = None) -> List[ChartMetadata]:
    op = kpi.operation
    evidence_id = registry.mint()

    if proposal is not None:
        chart = _plan_proposed(df, kpi, proposal["kind"],
                               proposal.get("reason", ""), evidence_id, index,
                               threshold)
        if chart is not None:
            _register(registry, chart)
            return [chart]
        return []

    function = op.function

    if function == "growth":
        line = _rule_2_growth(df, kpi, evidence_id, index, threshold)
        charts = [line] if line is not None else []
        if line is not None:
            _register(registry, line)
        # Rule 11: alongside the trend line, a waterfall shows how each
        # period contributes to the running total (variance view).
        if len(charts) == 1:
            waterfall = _rule_11_waterfall(df, kpi, registry,
                                           index + len(charts), threshold)
            if waterfall is not None:
                charts.append(waterfall)
        return charts
    if function == "correlation":
        chart = _rule_7_scatter(df, kpi, op.column_a or op.column,
                                op.column_b, evidence_id, index, threshold)
    elif function in _AGGREGATES and op.group_by:
        chart = _rule_dimension(df, kpi, op.group_by[0], evidence_id, index,
                                threshold, function=function)
    else:
        return []                     # headline number, no visual shape

    if chart is not None:
        _register(registry, chart)
    return [chart] if chart is not None else []


def _plan_proposed(df: pd.DataFrame, kpi: KpiCandidate, kind: str,
                   reason: str, evidence_id: str, index: int,
                   threshold: int) -> Optional[ChartMetadata]:
    """Build a chart for an LLM-proposed kind (already validated)."""
    op = kpi.operation
    columns: List[str] = []
    for name in (op.column_a, op.column_b, op.column, op.over_column,
                 *(op.group_by or [])):
        if name and name not in columns and name in df.columns:
            columns.append(name)
    chart = ChartMetadata(
        chart_id=f"CH-{index:03d}", kind=kind,
        reason=f"llm_proposed_{kind}: {reason}".strip(),
        columns=columns,
        title=kpi.name or " | ".join(columns),
        kpi_id=kpi.kpi_id,
        evidence_id=evidence_id,
    )
    if _thin(df, threshold):
        return _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
    return chart


def _rule_2_growth(df: pd.DataFrame, kpi: KpiCandidate, evidence_id: str,
                   index: int, threshold: int) -> Optional[ChartMetadata]:
    op = kpi.operation
    over = op.over_column
    if not over or over not in df.columns:
        return None
    try:
        series = growth_series(df, op)
    except (KeyError, ValueError, TypeError):
        return None
    if not series:
        return None  # nothing drawable (e.g. YoY on a 10-day sample)
    if len(series) < 3:
        return _downgrade_low_n(
            ChartMetadata(
                chart_id=f"CH-{index:03d}", kind="line",
                reason=RULE["rule_2"],
                columns=[over, op.column or ""],
                title=f"{kpi.name} ({op.period or 'MoM'} growth)",
                kpi_id=kpi.kpi_id,
                evidence_id=evidence_id,
            ), f"{len(series)} time points < 3")
    chart = ChartMetadata(
        chart_id=f"CH-{index:03d}", kind="line",
        reason=RULE["rule_2"],
        columns=[over, op.column or ""],
        title=f"{kpi.name} ({op.period or 'MoM'} growth)",
        kpi_id=kpi.kpi_id,
        evidence_id=evidence_id,
    )
    if _thin(df, threshold):
        return _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
    return chart


def _rule_7_scatter(df: pd.DataFrame, kpi: KpiCandidate, col_a: str,
                    col_b: str, evidence_id: str, index: int,
                    threshold: int) -> Optional[ChartMetadata]:
    if not col_a or not col_b or col_a not in df.columns \
            or col_b not in df.columns:
        return None
    chart = ChartMetadata(
        chart_id=f"CH-{index:03d}", kind="scatter",
        reason=RULE["rule_7"],
        columns=[col_a, col_b],
        title=f"{kpi.name} ({col_a} x {col_b})",
        kpi_id=kpi.kpi_id,
        evidence_id=evidence_id,
    )
    if _thin(df, threshold):
        return _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
    return chart


def _rule_dimension(df: pd.DataFrame, kpi: KpiCandidate, dimension: str,
                    evidence_id: str, index: int,
                    threshold: int,
                    function: str = "sum") -> Optional[ChartMetadata]:
    if dimension not in df.columns:
        return None
    values = df[dimension].dropna().nunique()
    is_share = bool(_SHARE_RE.search(kpi.name or ""))
    # Rule 10: ranked contribution — a sum/count over a dimension with
    # 3-15 values is a Pareto (sorted bars + cumulative % line) so the
    # "few items drive most of the total" pattern is explicit.
    if (function in ("sum", "count", "nunique") and not is_share
            and 3 <= values <= 15):
        kind, reason = "pareto", RULE["rule_10"]
    elif values <= 2 and is_share:
        kind, reason = "doughnut", RULE["rule_9"]
    elif values <= 2:
        kind, reason = "bar", RULE["rule_1"]
    elif values <= 12:
        kind, reason = "bar", RULE["rule_3"]
    elif values <= 50:
        kind, reason = "barh", RULE["rule_4"]
    else:
        kind, reason = "barh", f"{RULE['rule_5']} (top-{_TOP_N_ROLLUP} + $rest)"
    chart = ChartMetadata(
        chart_id=f"CH-{index:03d}", kind=kind, reason=reason,
        columns=[dimension, kpi.operation.column or ""],
        title=kpi.name or dimension,
        kpi_id=kpi.kpi_id,
        evidence_id=evidence_id,
    )
    if _thin(df, threshold):
        return _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
    return chart


def _rule_11_waterfall(df: pd.DataFrame, kpi: KpiCandidate,
                       registry: EvidenceRegistry, index: int,
                       threshold: int) -> Optional[ChartMetadata]:
    """Period-contribution waterfall for a growth KPI (rule 11).

    Uses the same period series as the trend line — each bar is one
    period's contribution, floating on the running total, so the final
    bar tops at the grand total (variance view)."""
    op = kpi.operation
    over = op.over_column
    if not over or over not in df.columns:
        return None
    try:
        series = growth_series(df, op)
    except (KeyError, ValueError, TypeError):
        return None
    if not series or len(series) < 3:
        return None
    chart = ChartMetadata(
        chart_id=f"CH-{index:03d}", kind="waterfall",
        reason=RULE["rule_11"],
        columns=[over, op.column or ""],
        title=f"{kpi.name} by period (contribution)",
        kpi_id=kpi.kpi_id,
        evidence_id=registry.mint(),
    )
    if _thin(df, threshold):
        chart = _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
    _register(registry, chart)
    return chart


# ---------------------------------------------------------------------------
# Shape-driven extras (numeric/ordinal columns beyond the KPI list)
# ---------------------------------------------------------------------------

def _plan_histograms(df: pd.DataFrame, registry: EvidenceRegistry, index: int,
                     measures: List[str],
                     threshold: int) -> List[ChartMetadata]:
    charts: List[ChartMetadata] = []
    for measure in measures:
        series = pd.to_numeric(df[measure], errors="coerce").dropna()
        if len(series) < 3:
            continue
        index += 1
        chart = ChartMetadata(
            chart_id=f"CH-{index:03d}", kind="histogram",
            reason=RULE["rule_6"], columns=[measure],
            title=f"{measure} distribution",
            evidence_id=registry.mint(),
        )
        if _thin(df, threshold):
            chart = _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
        _register(registry, chart)
        charts.append(chart)
    return charts


def _plan_measure_relation(df: pd.DataFrame, registry: EvidenceRegistry,
                           index: int, measures: List[str],
                           existing: List[ChartMetadata],
                           threshold: int) -> List[ChartMetadata]:
    """Rule 7 (scatter) for exactly 2 measures, rule 8 (heatmap) for >= 3."""
    if len(measures) < 2:
        return []
    if len(measures) == 2:
        if any(c.kind == "scatter" and set(c.columns) == set(measures)
               for c in existing):
            return []
        index += 1
        chart = ChartMetadata(
            chart_id=f"CH-{index:03d}", kind="scatter",
            reason=RULE["rule_7"], columns=list(measures),
            title=f"{measures[0]} x {measures[1]}",
            evidence_id=registry.mint(),
        )
        if _thin(df, threshold):
            chart = _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
        _register(registry, chart)
        return [chart]
    index += 1
    chart = ChartMetadata(
        chart_id=f"CH-{index:03d}", kind="heatmap",
        reason=RULE["rule_8"], columns=measures,
        title="Ranked correlation heatmap",
        evidence_id=registry.mint(),
    )
    if _thin(df, threshold):
        chart = _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
    _register(registry, chart)
    return [chart]


# ---------------------------------------------------------------------------
# Hybrid proposals — LLM suggests kinds, Python validates (§2.5)
# ---------------------------------------------------------------------------


def _role_columns(df: pd.DataFrame, understanding: DatasetUnderstanding,
                  roles: Tuple[str, ...]) -> List[str]:
    return [c.name for c in understanding.columns
            if c.role in roles and c.name in df.columns]


def _kind_fits(df: pd.DataFrame, understanding: DatasetUnderstanding,
               kpi: KpiCandidate, kind: str) -> bool:
    """Data-shape feasibility per kind — a proposal must be drawable."""
    numeric = _role_columns(df, understanding, ("measure",))
    temporal = _role_columns(df, understanding, ("temporal",))
    dimensions = _role_columns(df, understanding,
                               ("dimension", "categorical"))
    op = kpi.operation
    over = op.over_column if op.over_column in temporal else (
        temporal[0] if temporal else None)
    group_dim = (op.group_by or [None])[0]
    dimension = group_dim if group_dim in dimensions else (
        dimensions[0] if dimensions else None)

    if kind in ("line", "area"):
        return over is not None and df[over].nunique() >= 3
    if kind == "scatter":
        return len(numeric) >= 2
    if kind == "heatmap":
        return len(numeric) >= 3
    if kind in ("histogram", "boxplot"):
        target = op.column if op.column in numeric else (
            numeric[0] if numeric else None)
        return target is not None
    if kind in ("doughnut", "pie"):
        if _SHARE_RE.search(kpi.name or ""):
            return dimension is not None or len(numeric) >= 1
        return dimension is not None and df[dimension].dropna().nunique() <= 2
    if kind == "stacked_bar":
        return dimension is not None and len(numeric) >= 2
    if kind == "pareto":
        return dimension is not None and 3 <= df[dimension].dropna().nunique() <= 15
    if kind == "waterfall":
        return over is not None and df[over].nunique() >= 3
    if kind in ("bar", "barh", "lollipop"):
        return dimension is not None
    return False


def validate_proposed_kinds(df: pd.DataFrame, plan: AnalysisPlan,
                            understanding: DatasetUnderstanding,
                            proposals: List[Dict[str, Any]] | None,
                            ) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Whitelist + feasibility check for LLM-proposed chart kinds.

    Returns (accepted {kpi_id: {kind, reason}}, errors). Rejected proposals
    never reach the planner — they fall back to the rule table.
    """
    errors: List[str] = []
    accepted: Dict[str, Dict[str, str]] = {}
    known = {k.kpi_id: k for k in plan.candidate_kpis}
    for index, entry in enumerate(proposals or []):
        if not isinstance(entry, dict):
            errors.append(f"proposal #{index}: must be an object")
            continue
        kpi_id = str(entry.get("kpi_id") or "")
        kind = str(entry.get("kind") or "")
        if kpi_id not in known:
            errors.append(f"proposal #{index}: unknown kpi_id '{kpi_id}'")
            continue
        if kind not in CHART_KINDS:
            errors.append(f"{kpi_id}: unknown chart kind '{kind}'")
            continue
        if not _kind_fits(df, understanding, known[kpi_id], kind):
            errors.append(f"{kpi_id}: kind '{kind}' does not fit the data "
                          f"(see data shape rules)")
            continue
        accepted[kpi_id] = {"kind": kind,
                            "reason": str(entry.get("reason") or "")}
    return accepted, errors


# ---------------------------------------------------------------------------
# Ranking + truncation (spec §1: max_chart_count, drop lowest-ranked)
# ---------------------------------------------------------------------------

_STRENGTH = {
    "line": 100, "pareto": 95, "waterfall": 90, "scatter": 90,
    "heatmap": 85, "doughnut": 85,
    "bar": 80, "barh": 80, "histogram": 60,
}


def rank_candidates(candidates: List[ChartMetadata],
                    novelty_penalty: float = 0.0
                    ) -> List[ChartMetadata]:
    """Rank chart candidates by evidence strength (stable on chart_id).

    Task B.2: when ``novelty_penalty`` > 0, each repeat of a kind that was
    already selected earlier in the ranking loses a fraction of its base
    score (final = base * (1 - penalty)) — a bias toward variety, not a
    hard cap. Ranking order stays fully deterministic.
    """
    base = sorted(candidates, key=lambda c: (-_STRENGTH.get(c.kind, 0),
                                             c.chart_id))
    if novelty_penalty <= 0:
        return base
    seen: Dict[str, int] = {}
    scored: List[Tuple[float, ChartMetadata]] = []
    for chart in base:
        repeats = seen.get(chart.kind, 0)
        seen[chart.kind] = repeats + 1
        effective = _STRENGTH.get(chart.kind, 0) * (1 - novelty_penalty
                                                    if repeats else 1.0)
        scored.append((effective, chart))
    return [chart for _, chart in sorted(
        scored, key=lambda t: (-t[0], t[1].chart_id))]


def _claim_kind_fits(df: pd.DataFrame, understanding: DatasetUnderstanding,
                     chart: ChartMetadata, kind: str) -> bool:
    """Light feasibility check for the insight-linked override (B.3) —
    equivalent to the shape rules without needing the KPI candidate."""
    if kind in ("scatter", "heatmap"):
        numerics = [c for c in understanding.measures if c in df.columns]
        return len(numerics) >= 2
    if kind == "line":
        return bool(understanding.has_temporal_data
                    or understanding.temporal_columns)
    if kind in ("bar", "barh", "lollipop"):
        return any(c in (understanding.dimensions or []) for c in
                   (chart.columns or []))
    return True


def apply_insight_kind_overrides(charts: List[ChartMetadata],
                                 insights: List["Insight"],
                                 df: pd.DataFrame,
                                 understanding: DatasetUnderstanding,
                                 ) -> Tuple[List[ChartMetadata],
                                            List[Dict[str, str]]]:
    """Task B.3: priority override — chart kind follows the insight claim
    type, not just the dtype shape table (which stays the fallback).

    Mapping (over the existing deterministic table):
      CORRELATIONAL claim  -> scatter (or equivalent)
      trend claim (time)   -> line   (DESCRIPTIVE with growth_rate evidence)
      COMPARATIVE claim    -> bar / lollipop
      DESCRIPTIVE          -> unchanged (dtype table already produces
                              histograms for distributions)

    Returns (updated charts, applied [{chart_id, kind, reason}]) — charts
    are deep-copied; the caller re-renders the affected SVGs.
    """
    by_kpi: Dict[str, List["Insight"]] = {}
    for insight in insights:
        for kpi_id in (insight.related_kpis or []):
            by_kpi.setdefault(kpi_id, []).append(insight)

    def _is_trend_claim(insight) -> bool:
        return (insight.claim_type == "DESCRIPTIVE"
                and "growth_rate" in (insight.required_evidence or []))

    applied: List[Dict[str, str]] = []
    out: List[ChartMetadata] = []
    for chart in charts:
        chart = chart.model_copy(deep=True)
        claims = by_kpi.get(chart.kpi_id or "", [])
        if claims:
            chosen: Optional[str] = None
            chosen_insight = None
            for insight in claims:
                if insight.claim_type == "CORRELATIONAL":
                    chosen, chosen_insight = "scatter", insight
                    break
            if chosen is None:
                for insight in claims:
                    if _is_trend_claim(insight):
                        chosen, chosen_insight = "line", insight
                        break
            if chosen is None:
                for insight in claims:
                    if insight.claim_type == "COMPARATIVE":
                        chosen, chosen_insight = "bar", insight
                        break
            if (chosen is not None and chosen != chart.kind
                    and chosen in CHART_KINDS
                    and _claim_kind_fits(df, understanding, chart, chosen)):
                old_kind = chart.kind
                chart.kind = chosen
                chart.reason = (f"{chart.reason}; insight_linked: "
                                f"{chosen_insight.claim_type} -> {chosen} "
                                f"(insight {chosen_insight.insight_id})")
                applied.append({
                    "chart_id": chart.chart_id,
                    "from": old_kind, "to": chosen,
                    "reason": f"insight {chosen_insight.insight_id} "
                              f"({chosen_insight.claim_type})",
                })
        out.append(chart)
    return out, applied


def truncate(candidates: List[ChartMetadata],
             max_chart_count: int) -> Tuple[List[ChartMetadata], bool]:
    """Drop the lowest-ranked charts beyond max_chart_count."""
    if max_chart_count <= 0:
        return [], True
    kept = candidates[:max_chart_count]
    truncated = len(candidates) > max_chart_count
    return kept, truncated


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def plan_charts(df: pd.DataFrame, plan: AnalysisPlan,
                understanding: DatasetUnderstanding,
                registry: EvidenceRegistry,
                max_chart_count: int = 20,
                thin_threshold: int = THIN_THRESHOLD,
                proposals: List[Dict[str, Any]] | None = None,
                accepted_kinds: Dict[str, str] | None = None,
                novelty_penalty: float = 0.0,
                ) -> Tuple[List[ChartMetadata], bool]:
    """Plan all candidate charts from the KPI list + numeric/ordinal columns.

    ``proposals`` (optional) are LLM-suggested kinds
    ``[{kpi_id, kind, reason}]`` — validated internally (whitelist +
    data-shape feasibility); rejected entries fall back to the rule table.
    ``accepted_kinds`` (optional) is a pre-computed accepted mapping from
    ``validate_proposed_kinds`` — when provided, the internal validation is
    skipped (avoids double computation).
    Returns (chart_metadata list, charts_truncated). Order after ranking is the
    final draw order; excess candidates are dropped lowest-ranked first.
    """
    if accepted_kinds is None:
        accepted, _ = validate_proposed_kinds(df, plan, understanding,
                                              proposals)
    else:
        accepted = accepted_kinds
    measures = measure_columns(understanding, df)
    candidates: List[ChartMetadata] = []
    index = 0

    for kpi in plan.candidate_kpis:
        proposal = accepted.get(kpi.kpi_id)
        charts = _plan_kpi(df, kpi, registry, index + 1, thin_threshold,
                           proposal=proposal)
        candidates.extend(charts)
        index += len(charts)

    charts = _plan_histograms(df, registry, index, measures, thin_threshold)
    candidates.extend(charts)
    index += len(charts)

    candidates.extend(_plan_measure_relation(
        df, registry, index, measures, candidates, thin_threshold))
    candidates = _dedupe(candidates)
    ranked = rank_candidates(candidates, novelty_penalty=novelty_penalty)
    kept, truncated = truncate(ranked, max_chart_count)
    return kept, truncated
