"""Stage 3 — Data Quality: deterministic pre-cleaning gate (A2.3).

Python only, no LLM, no CrewAI: objective checks that catch what the
cleaning strategy cannot see, plus a deterministic repair that NEVER
invents data. Repair only casts types by role, drops exact duplicates and
impossible rows, and preserves/records everything else for Cleaning to
decide (§2.3 repair-scope table).

Deterministic rules encoded here:
- negative values in a measure  -> flagged "negative" (high), never fixed
- object-typed measure          -> cast to numeric (repair)
- object-typed temporal         -> cast to datetime (repair)
- string identifier             -> kept as str (no cast)
- exact duplicates              -> drop_duplicates (repair)
- unparseable temporal value    -> row dropped + logged (repair)
- year >= IMPOSSIBLE_YEAR (2100)-> row dropped + logged (repair)
- date beyond today+1y          -> flagged "future_dates" (medium, kept)
- percentage-like measure > 100 -> flagged (high, kept)
- age-like column out of range  -> flagged (high, kept)
- MAR/MNAR missingness          -> preserved + flagged (no imputation)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from shared.core.semantic_guards import NEGATIVE_ALLOWED_RE, is_mixed_unit
from shared.schemas import (
    BusinessContext,
    DataProfile,
    DataQualityReport,
    DatasetUnderstanding,
)

PERCENT_MAX = 100.0
AGE_MAX = 150.0
IMPOSSIBLE_YEAR = 2100
FUTURE_HORIZON_DAYS = 365
MISSING_GAP_THRESHOLD = 0.2   # MAR: per-group missing-rate gap
MNAR_GAP_THRESHOLD = 0.3      # MNAR: missing-rate gap across index thirds
ORPHAN_OVERLAP_MIN = 0.5      # referential: min overlap to call a ref column

SEV_HIGH = "high"
SEV_MEDIUM = "medium"

CAT_SCHEMA = "schema"
CAT_TYPE_MISMATCH = "type_mismatch"
CAT_INVALID = "invalid_value"
CAT_BUSINESS = "business_rule"
CAT_MISSINGNESS = "missingness"
CAT_DUPLICATES = "duplicates"
CAT_REFERENTIAL = "referential_integrity"
CAT_ENCODING = "encoding"

LogTool = Optional[Callable[[str, float, str], None]]


@dataclass
class DqIssue:
    """One finding; severity drives the gate (high -> needs_repair)."""
    severity: str
    category: str
    column: str
    detail: str

    def to_dict(self) -> Dict[str, str]:
        return {"severity": self.severity, "category": self.category,
                "column": self.column, "detail": self.detail}


# ---------------------------------------------------------------------------
# Part 1 — schema, invalid values, business rules
# ---------------------------------------------------------------------------


def check_schema(understanding: DatasetUnderstanding,
                 profile: DataProfile) -> List[DqIssue]:
    """Column presence + role/type consistency against the understanding."""
    issues: List[DqIssue] = []
    known = {c.name for c in understanding.columns}
    for col in profile.columns:
        if col not in known:
            issues.append(DqIssue(SEV_HIGH, CAT_SCHEMA, col,
                                  "missing_column"))
    for col in understanding.columns:
        if col.name not in profile.columns:
            issues.append(DqIssue(SEV_HIGH, CAT_SCHEMA, col.name,
                                  "unknown_column"))
    for col in understanding.columns:
        dtype = str(col.dtype)
        if col.role in ("measure", "temporal") and not dtype.startswith(
                ("int", "float", "datetime", "date", "bool")):
            issues.append(DqIssue(SEV_MEDIUM, CAT_TYPE_MISMATCH, col.name,
                                  f"type_mismatch_{dtype}"))
    return issues


def _to_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def check_invalid_values(understanding: DatasetUnderstanding, df: pd.DataFrame,
                         *,
                         percent_max: float = PERCENT_MAX,
                         age_max: float = AGE_MAX,
                         future_horizon_days: int = FUTURE_HORIZON_DAYS,
                         ) -> List[DqIssue]:
    """Impossible/implausible values, by column role (§2.3 checks)."""
    issues: List[DqIssue] = []
    today = datetime.now().date()
    horizon = timedelta(days=future_horizon_days)
    for col in understanding.columns:
        name = col.name
        if name not in df.columns:
            continue
        series = df[name]
        if col.role == "measure":
            numeric = _to_numeric(series)
            lowered = name.lower().replace(" ", "")
            # 3.2: negative values are only invalid when the measure
            # semantics forbid them — temperature_celsius, balance, growth
            # etc. may legitimately be negative.
            if (numeric < 0).any() \
                    and not NEGATIVE_ALLOWED_RE.search(lowered):
                issues.append(DqIssue(SEV_HIGH, CAT_INVALID, name,
                                      "negative"))
            if "percent" in lowered or "pct" in lowered or "%" in name:
                if (numeric > percent_max).any():
                    issues.append(DqIssue(SEV_HIGH, CAT_INVALID, name,
                                          "over_100_percent"))
            if "age" in lowered:
                if (numeric < 0).any() or (numeric > age_max).any():
                    issues.append(DqIssue(SEV_HIGH, CAT_INVALID, name,
                                          "out_of_range"))
        elif col.role == "temporal":
            parsed = pd.to_datetime(series, errors="coerce")
            non_null = series.notna()
            if (non_null & parsed.isna()).any():
                issues.append(DqIssue(SEV_HIGH, CAT_INVALID, name,
                                      "impossible"))
            elif (non_null & (parsed.dt.year >= IMPOSSIBLE_YEAR)).any():
                issues.append(DqIssue(SEV_HIGH, CAT_INVALID, name,
                                      "impossible"))
            elif (non_null
                  & (parsed.dt.date > today + horizon)).any():
                issues.append(DqIssue(SEV_MEDIUM, CAT_INVALID, name,
                                      "future_dates"))
        elif col.role not in ("measure", "temporal"):
            # 2.4: currency/unit strings ("$100", "EGP 500", "100 kg") in a
            # non-measure column — a DQ flag so numeric coercion never
            # silently strips units downstream.
            if is_mixed_unit(series):
                issues.append(DqIssue(SEV_MEDIUM, CAT_INVALID, name,
                                      "mixed_units"))
    return issues


def _parse_range_constraints(text: str) -> Dict[str, Tuple[float, float]]:
    """Parse declared range rules from free-text answers, e.g.
    'quantity between 1 and 50' / 'revenue >= 0' / 'age from 18 to 65'."""
    constraints: Dict[str, Tuple[float, float]] = {}
    lower = text.lower()
    for match in re.finditer(
            r"([a-z_][a-z0-9_]*)\s+(?:from|between)\s+(-?[0-9.]+)"
            r"\s+(?:to|and)\s+(-?[0-9.]+)", lower):
        column, lo, hi = match.group(1), float(match.group(2)), \
            float(match.group(3))
        constraints[column] = (lo, hi)
    for match in re.finditer(
            r"([a-z_][a-z0-9_]*)\s*>=\s*(-?[0-9.]+)", lower):
        column, value = match.group(1), float(match.group(2))
        lo, hi = constraints.get(column, (None, None))
        constraints[column] = (lo if lo is not None else value, hi)
    return {c: (lo, hi) for c, (lo, hi) in constraints.items()
            if lo is not None or hi is not None}


def check_business_rules(context: BusinessContext, df: pd.DataFrame
                         ) -> List[DqIssue]:
    """Declared business rules from the context; Generic Mode -> no checks.

    Only deterministic, explicitly declared constraints are enforced (range
    patterns in goal_summary/answers). Encoding check: duplicate column
    names differing only by case.
    """
    issues: List[DqIssue] = []
    seen_lower: Dict[str, str] = {}
    for col in df.columns:
        key = str(col).lower()
        if key in seen_lower:
            issues.append(DqIssue(SEV_MEDIUM, CAT_ENCODING, str(col),
                                  f"case_duplicate_of_{seen_lower[key]}"))
        seen_lower[key] = str(col)

    if context.generic_mode:
        return issues
    text = " ".join([context.goal_summary, *context.answers.values(),
                     *context.business_questions])
    constraints = _parse_range_constraints(text)
    for column, (lo, hi) in constraints.items():
        if column not in df.columns:
            continue
        numeric = _to_numeric(df[column])
        violations = ((numeric < lo) | (numeric > hi)).sum()
        if violations:
            issues.append(DqIssue(
                SEV_HIGH, CAT_BUSINESS, column,
                f"out_of_declared_range_{lo}_{hi}_x{violations}"))
    return issues


# ---------------------------------------------------------------------------
# Part 2 — missingness, duplicates, referential integrity
# ---------------------------------------------------------------------------


def _missing_assessment(df: pd.DataFrame, column: str) -> str:
    """MAR test (missing correlates with another column's groups), then an
    MNAR proxy (missing clusters across row order), else MCAR."""
    series = df[column]
    for other in df.columns:
        if other == column or df[other].nunique() > 20:
            continue
        grouped = series.isna().groupby(df[other], dropna=False).mean()
        if len(grouped) < 2:
            continue
        sizes = series.isna().groupby(df[other], dropna=False).size()
        if (sizes < 3).any():
            continue
        gap = grouped.max() - grouped.min()
        if gap > MISSING_GAP_THRESHOLD:
            return "MAR_suspected"
    if pd.api.types.is_numeric_dtype(series):
        n = len(series)
        if n >= 9:
            thirds = [series.iloc[i * n // 3:(i + 1) * n // 3]
                      for i in range(3)]
            rates = [t.isna().mean() for t in thirds]
            if max(rates) - min(rates) > MNAR_GAP_THRESHOLD:
                return "MNAR_suspected"
    return "MCAR"


def analyze_missingness(understanding: DatasetUnderstanding,
                        df: pd.DataFrame) -> Dict[str, Any]:
    """Overall + per-column missing rates with MCAR/MAR/MNAR assessment."""
    per_column: Dict[str, Any] = {}
    assessments: List[str] = []
    for col in understanding.columns:
        if col.name not in df.columns:
            continue
        series = df[col.name]
        missing = int(series.isna().sum())
        rate = missing / len(series) if len(series) else 0.0
        assessment = ("none" if missing == 0
                      else _missing_assessment(df, col.name))
        if assessment not in ("none", "MCAR"):
            assessments.append(assessment)
        per_column[col.name] = {
            "missing": missing,
            "rate": round(rate, 6),
            "assessment": assessment,
        }
    if "MAR_suspected" in assessments:
        overall = "MAR_suspected"
    elif "MNAR_suspected" in assessments:
        overall = "MNAR_suspected"
    elif any(p["missing"] for p in per_column.values()):
        overall = "MCAR"
    else:
        overall = "none"
    total_cells = len(df) * len(df.columns)
    overall_rate = (sum(p["missing"] for p in per_column.values())
                    / total_cells) if total_cells else 0.0
    return {
        "rate": round(overall_rate, 6),
        "pattern": "non_random" if overall not in ("MCAR", "none")
        else "random",
        "assessment": overall,
        "by_column": per_column,
    }


def _jsonable(value: Any) -> Any:
    """numpy scalars -> python scalars so json.dumps never chokes."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def detect_duplicates(df: pd.DataFrame
                      ) -> Tuple[int, List[Dict[str, Any]]]:
    """Exact duplicate rows: count + first 5 examples (JSON-safe dicts)."""
    mask = df.duplicated()
    count = int(mask.sum())
    examples = [_jsonable(row.to_dict())
                for _, row in df[mask].head(5).iterrows()]
    return count, examples


def check_referential_integrity(understanding: DatasetUnderstanding,
                                df: pd.DataFrame) -> List[DqIssue]:
    """Identifier nulls + orphaned references (name-based candidates only,
    so plain value overlap between unrelated columns never fires)."""
    issues: List[DqIssue] = []
    identifiers = [c for c in understanding.columns
                   if c.role == "identifier" and c.name in df.columns]
    for col in identifiers:
        missing = int(df[col.name].isna().sum())
        if missing:
            issues.append(DqIssue(SEV_MEDIUM, CAT_REFERENTIAL, col.name,
                                  f"identifier_nulls_x{missing}"))
    for ref in identifiers:
        ref_values = set(df[ref.name].dropna().unique())
        if not ref_values:
            continue
        stem = ref.name[:-3] if ref.name.endswith("_id") else ref.name
        candidates = [c for c in understanding.columns
                      if c.name in df.columns
                      and c.name != ref.name
                      and c.role in ("dimension", "free_text", "identifier")
                      and (c.name.endswith("_id") or stem in c.name)]
        for candidate in candidates:
            values = set(df[candidate.name].dropna().unique())
            if not values:
                continue
            overlap = len(values & ref_values) / len(values)
            orphans = sorted(values - ref_values)
            if overlap >= ORPHAN_OVERLAP_MIN and orphans:
                issues.append(DqIssue(
                    SEV_MEDIUM, CAT_REFERENTIAL, candidate.name,
                    f"orphaned_references_x{len(orphans)}"))
    return issues


# ---------------------------------------------------------------------------
# Part 3 — deterministic repair + report assembly
# ---------------------------------------------------------------------------


def deterministic_repair(understanding: DatasetUnderstanding,
                         df: pd.DataFrame
                         ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply the §2.3 repair table. NEVER invents data: no sign flips, no
    imputation. Returns (repaired_df, repair_log)."""
    work = df.copy()
    type_casts: Dict[str, str] = {}
    coerced_to_null: Dict[str, int] = {}
    impossible_rows: Dict[str, List[int]] = {}

    for col in understanding.columns:
        name = col.name
        if name not in work.columns:
            continue
        series = work[name]
        dtype = str(series.dtype)
        if col.role == "measure" and not dtype.startswith(
                ("int", "float", "bool")):
            original_non_null = series.notna()
            numeric = pd.to_numeric(series, errors="coerce")
            coerced = int((original_non_null & numeric.isna()).sum())
            if coerced:
                coerced_to_null[name] = coerced
            work[name] = numeric
            type_casts[name] = f"object->{numeric.dtype}"
        elif col.role == "temporal" and not dtype.startswith(
                ("datetime", "date")):
            original_non_null = series.notna()
            parsed = pd.to_datetime(series, errors="coerce")
            bad = original_non_null & parsed.isna()
            if bad.any():
                impossible_rows[name] = [
                    int(i) for i in work.index[bad]]
            work[name] = parsed
            type_casts[name] = "object->datetime64"

    for col in understanding.columns:
        name = col.name
        if col.role != "temporal" or name not in work.columns:
            continue
        parsed = pd.to_datetime(work[name], errors="coerce")
        original_non_null = df[name].notna()
        too_far = original_non_null & parsed.notna() & (
            parsed.dt.year >= IMPOSSIBLE_YEAR)
        if too_far.any():
            impossible_rows.setdefault(name, []).extend(
                int(i) for i in work.index[too_far])

    if impossible_rows:
        for name in impossible_rows:
            impossible_rows[name] = sorted(set(impossible_rows[name]))
        drop_indices = sorted({i for rows in impossible_rows.values()
                               for i in rows})
        work = work.drop(index=drop_indices)

    duplicates_removed = int(work.duplicated().sum())
    work = work.drop_duplicates()

    repair_applied = bool(type_casts or duplicates_removed or impossible_rows)
    return work, {
        "repair_applied": repair_applied,
        "duplicates_removed": duplicates_removed,
        "impossible_rows_dropped": impossible_rows,
        "type_casts": type_casts,
        "coerced_to_null": coerced_to_null,
    }


def _default_limits() -> Dict[str, float]:
    return {
        "dq_percent_max": PERCENT_MAX,
        "dq_age_max": AGE_MAX,
        "dq_future_horizon_days": FUTURE_HORIZON_DAYS,
    }


def assemble_report(understanding: DatasetUnderstanding,
                    profile: DataProfile,
                    df: pd.DataFrame,
                    context: BusinessContext,
                    *,
                     limits: Optional[Dict[str, Any]] = None,
                     log_tool: LogTool = None,
                     skip_repair: bool = False,
                      ) -> Tuple[DataQualityReport, Dict[str, Any]]:
    """Run every check + the deterministic repair; build the report.

    Gate: status is 'needs_repair' when any high-severity finding exists or
    the repair changed data; otherwise 'passed' (Cleaning may proceed).

    Parameters
    ----------
    skip_repair : bool
        If True (used for post-cleaning rechecks), skip the
        deterministic_repair step — the data has already been cleaned
        and CSV round-trips may re-introduce benign type mismatches.
    """
    cfg_limits = dict(_default_limits())
    if limits:
        cfg_limits.update({k: v for k, v in limits.items()
                           if k in cfg_limits})
    percent_max = float(cfg_limits["dq_percent_max"])
    age_max = float(cfg_limits["dq_age_max"])
    horizon = int(cfg_limits["dq_future_horizon_days"])

    def _timed(name: str, call: Callable[[], Any]) -> Any:
        start = datetime.now()
        try:
            result = call()
            if log_tool:
                log_tool(name, (datetime.now() - start).total_seconds(),
                         "passed")
            return result
        except Exception as exc:  # noqa: BLE001 -- check failure recorded
            if log_tool:
                log_tool(name, (datetime.now() - start).total_seconds(),
                         "failed")
            raise

    issues: List[DqIssue] = []
    issues += _timed("schema_checker_tool",
                     lambda: check_schema(understanding, profile))
    issues += _timed("invalid_value_checker_tool", lambda: check_invalid_values(
        understanding, df, percent_max=percent_max, age_max=age_max,
        future_horizon_days=horizon))
    issues += _timed("business_rules_checker_tool",
                     lambda: check_business_rules(context, df))
    missingness = _timed("missingness_analyzer_tool",
                         lambda: analyze_missingness(understanding, df))
    duplicates, examples = _timed(
        "duplicate_detector_tool", lambda: detect_duplicates(df))
    issues += _timed("referential_integrity_tool",
                     lambda: check_referential_integrity(understanding, df))
    if skip_repair:
        repair_log = {"repair_applied": False, "actions": []}
    else:
        _, repair_log = _timed(
            "deterministic_repair_tool",
            lambda: deterministic_repair(understanding, df))

    if missingness["assessment"] in ("MAR_suspected", "MNAR_suspected"):
        issues.append(DqIssue(SEV_MEDIUM, CAT_MISSINGNESS, "",
                              missingness["assessment"]))
    if duplicates:
        issues.append(DqIssue(SEV_MEDIUM, CAT_DUPLICATES, "",
                              f"exact_duplicates_x{duplicates}"))

    invalid: Dict[str, List[str]] = {}
    for issue in issues:
        if issue.category == CAT_INVALID:
            invalid.setdefault(issue.column, []).append(issue.detail)

    has_high = any(issue.severity == SEV_HIGH for issue in issues)
    status = "needs_repair" if (has_high or repair_log["repair_applied"]) \
        else "passed"
    report = DataQualityReport(
        status=status,
        invalid=invalid,
        missingness=missingness,
        duplicates=duplicates,
        issues=[issue.to_dict() for issue in issues],
    )
    return report, repair_log
