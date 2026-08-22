"""Stage 4 — Cleaning: §2.4 role × missingness strategy + deterministic executor.

Pure + deterministic. The LLM may propose the strategy (cleaning_strategy_tool),
but Python normalizes it (columns must exist, actions come from a fixed set) and
executes every step, logging each one. Never invents data: fills only with
median/mode/"Unknown", preserves non-random missingness as `*_missing_flag`
features (flag_and_preserve), and never silently drops what it can flag.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from shared.core.semantic_guards import IDENTIFIER_NAME_RE, NEGATIVE_ALLOWED_RE
from shared.schemas import (
    CleaningResult,
    DatasetUnderstanding,
    DataQualityReport,
)

# ---------------------------------------------------------------------------
# Strategy vocabulary
# ---------------------------------------------------------------------------

CLEANING_ACTIONS = frozenset({
    "keep",              # nothing to do (no missingness)
    "median_fill",       # MCAR < 5% measure
    "median_fill_flag",  # MCAR 5-30% measure — fill + boolean flag
    "mode_fill",         # MCAR < 5% dimension
    "unknown_fill",      # MCAR 5-30% dimension — "Unknown"
    "flag_and_preserve",  # MAR/MNAR signal — flag, never impute
    "keep_flag",         # >70% dimension — keep sparse + flag
    "drop_row",          # temporal/identifier missing — drop affected rows
    "drop_column",       # >70% — drop the column entirely
    "drop_negative",     # flagged negative measure (§2.3: cleaning decides)
    "sanitize_text",     # SQL injection suspected — strip suspicious chars
})

MISSING_MAR_RATE = 0.70
MISSING_LOW_RATE = 0.05
MISSING_MID_RATE = 0.30

_SIGNAL_ASSESSMENTS = ("MAR_suspected", "MNAR_suspected")


# ---------------------------------------------------------------------------
# Part 1 — strategy
# ---------------------------------------------------------------------------


def _column_strategy(role: str, rate: float,
                     assessment: str) -> Dict[str, Any]:
    """One §2.4 row: role × (MCAR<5% | MCAR 5-30% | MAR/MNAR | >70%)."""
    if rate <= 0.0:
        return {"action": "keep", "detail": "no_missingness"}
    if rate > MISSING_MAR_RATE:
        table = {
            "measure": ("drop_column", "more_than_70pct_missing"),
            "dimension": ("keep_flag", "more_than_70pct_missing_keep_flag"),
            "temporal": ("drop_column", "more_than_70pct_missing"),
            "identifier": ("drop_column", "more_than_70pct_missing"),
            "categorical": ("keep_flag", "more_than_70pct_missing_keep_flag"),
        }
        action, detail = table.get(role, ("keep_flag", "more_than_70pct"))
    elif assessment in _SIGNAL_ASSESSMENTS:
        table = {
            "measure": ("flag_and_preserve", "mar_mnar_signal_exclude"),
            "dimension": ("flag_and_preserve", "mar_mnar_signal"),
            "temporal": ("drop_row", "mar_mnar_signal"),
            "identifier": ("drop_row", "mar_mnar_signal"),
            "categorical": ("flag_and_preserve", "mar_mnar_signal"),
        }
        action, detail = table.get(role, ("flag_and_preserve",
                                          "mar_mnar_signal"))
    elif role == "measure":
        if rate < MISSING_LOW_RATE:
            action, detail = "median_fill", "mcar_below_5pct"
        else:
            action, detail = "median_fill_flag", "mcar_5_30pct"
    elif role in ("dimension", "categorical"):
        if rate < MISSING_LOW_RATE:
            action, detail = "mode_fill", "mcar_below_5pct"
        else:
            action, detail = "unknown_fill", "mcar_5_30pct"
    else:  # temporal / identifier / free_text
        action, detail = "drop_row", "missing_present"
    return {"action": action, "detail": detail}


def build_strategy(understanding: DatasetUnderstanding,
                   dq_report: DataQualityReport) -> Dict[str, Any]:
    """Deterministic default strategy from the §2.4 table + DQ report."""
    by_column = (dq_report.missingness or {}).get("by_column", {})
    invalid = dq_report.invalid or {}
    columns: List[Dict[str, Any]] = []
    for col in understanding.columns:
        meta = by_column.get(col.name, {"missing": 0, "rate": 0.0,
                                        "assessment": "none"})
        rate = float(meta.get("rate", 0.0))
        assessment = str(meta.get("assessment", "none"))
        flags = invalid.get(col.name, [])
        # §2.4 missingness decision (role × rate × assessment) is always
        # computed: a column may need BOTH drop_negative (4.2) AND a
        # missingness action (median_fill for measure MCAR<5% etc.) — the
        # executor applies every entry in order, so one column can carry
        # multiple actions.
        missing = _column_strategy(col.role, rate, assessment)
        actions: List[Dict[str, Any]] = []
        if missing["action"] == "drop_column":
            # >70% missing: the column is gone — nothing else to do for it.
            actions.append(missing)
        else:
            # 4.2/3.2: drop_negative is deterministic auto-apply for flagged
            # negative measures — but never for measures whose semantics allow
            # negatives (temperature, balance, growth, ...). Runs first so
            # downstream fills compute their statistic on non-negative values.
            if (col.role == "measure" and any(
                    "negative" in str(item) for item in flags)
                    and not NEGATIVE_ALLOWED_RE.search(col.name)):
                actions.append({"action": "drop_negative",
                                "detail": "negative_measure_flagged"})
            # 4.4: sql_injection_suspected — sanitize by stripping suspicious
            # SQL keywords and special characters from dimension columns.
            if any("sql_injection" in str(item) for item in flags):
                actions.append({"action": "sanitize_text",
                                "detail": "sql_injection_suspected"})
            if missing["action"] != "keep":
                actions.append(missing)
        if not actions:
            actions.append({"action": "keep", "detail": "no_missingness"})
        for decision in actions:
            columns.append({
                "column": col.name,
                "role": col.role,
                "action": decision["action"],
                "detail": decision["detail"],
            })
    # Auto-populate outlier handling from DQ issues so cleaning stages
    # handle them and the recheck doesn't re-flag them.
    outlier_columns: Dict[str, str] = {}
    for issue in (dq_report.issues or []):
        if (issue.get("category") == "invalid_value"
                and str(issue.get("detail", "")).startswith("outliers_iqr")):
            col = issue.get("column", "")
            if col:
                outlier_columns[col] = "flag"
    return {
        "columns": columns,
        "deduplicate": int(getattr(dq_report, "duplicates", 0) or 0) > 0,
        "outliers": outlier_columns,
    }


def normalize_strategy(raw: Any,
                       understanding: DatasetUnderstanding,
                       dq_report: DataQualityReport | None = None
                       ) -> Tuple[Dict[str, Any], List[str]]:
    """Validate + normalize a proposed strategy (never raises).

    Every referenced column must exist and every action must be from
    CLEANING_ACTIONS. Invalid entries are dropped with their reason in
    `errors`; unknown roles keep their deterministic default.
    """
    errors: List[str] = []
    report = dq_report or _empty_report()
    if isinstance(raw, str):
        try:
            import json
            raw = json.loads(raw)
        except Exception:
            return build_strategy(understanding, report), \
                ["strategy must be valid JSON"]
    if not isinstance(raw, dict):
        return build_strategy(understanding, report), \
            ["strategy must be an object with a 'columns' list"]

    default = build_strategy(understanding, report)
    known_columns = {c.name: c for c in understanding.columns}
    columns: List[Dict[str, Any]] = []
    for index, entry in enumerate(raw.get("columns") or []):
        if not isinstance(entry, dict):
            errors.append(f"column #{index}: must be an object")
            continue
        name = entry.get("column")
        if not name or name not in known_columns:
            errors.append(f"column #{index}: unknown column '{name}'")
            continue
        action = entry.get("action")
        if action not in CLEANING_ACTIONS:
            errors.append(f"{name}: unknown action '{action}'")
            continue
        role = known_columns[name].role
        columns.append({
            "column": name,
            "role": role,
            "action": action,
            "detail": str(entry.get("detail") or ""),
        })

    outliers = raw.get("outliers")
    if not isinstance(outliers, dict):
        outliers = {}
    cleaned_outliers: Dict[str, str] = {}
    for name, mode in outliers.items():
        if name not in known_columns:
            errors.append(f"outlier column '{name}' unknown")
            continue
        if mode not in ("flag", "drop"):
            errors.append(f"{name}: outlier mode must be 'flag' or 'drop'")
            continue
        # 3.1: outliers are only meaningful for measures — never flag/drop
        # identifier or categorical columns as "outliers".
        role = known_columns[name].role
        if role in ("identifier", "categorical") \
                or IDENTIFIER_NAME_RE.search(name):
            errors.append(f"outlier column '{name}' is not a measure")
            continue
        cleaned_outliers[name] = mode

    # Merge deterministic overrides for high-severity DQ issues that the
    # LLM may not know about (e.g. sanitize_text for SQL injection). The
    # overrides are ADDED as additional entries (never replacing the LLM's
    # chosen action): a column flagged negative still keeps its missingness
    # handling (median_fill etc.), and drop_negative is placed before the
    # column's other actions so fills run on non-negative values.
    if dq_report is not None:
        invalid = dq_report.invalid or {}
        for col_name, flags in invalid.items():
            meta = known_columns.get(col_name)
            if any("sql_injection" in str(f) for f in flags) \
                    and not _entry_has_action(columns, col_name,
                                              "sanitize_text"):
                _insert_action(columns, col_name,
                               meta.role if meta else "dimension",
                               "sanitize_text", "sql_injection_suspected")
            if any("negative" in str(f) for f in flags) and meta \
                    and meta.role == "measure" \
                    and not NEGATIVE_ALLOWED_RE.search(col_name) \
                    and not _entry_has_action(columns, col_name,
                                              "drop_negative"):
                _insert_action(columns, col_name, "measure",
                               "drop_negative", "negative_measure_flagged")

    return {
        "columns": columns,
        "deduplicate": bool(raw.get("deduplicate", default["deduplicate"])),
        "outliers": cleaned_outliers,
    }, errors


def _entry_has_action(columns: List[Dict[str, Any]], column: str,
                      action: str) -> bool:
    return any(c["column"] == column and c["action"] == action
               for c in columns)


def _insert_action(columns: List[Dict[str, Any]], column: str, role: str,
                   action: str, detail: str) -> None:
    """Insert an action entry before the column's existing entries so it
    executes first (drop_negative before fills/flags)."""
    entry = {"column": column, "role": role, "action": action,
             "detail": detail}
    index = next((i for i, c in enumerate(columns)
                  if c["column"] == column), len(columns))
    columns.insert(index, entry)


def _empty_report() -> DataQualityReport:
    return DataQualityReport(status="passed", missingness={})


# ---------------------------------------------------------------------------
# Part 2 — execution
# ---------------------------------------------------------------------------


def _cast_types(df: pd.DataFrame, understanding: DatasetUnderstanding,
                log: List[Dict[str, Any]]) -> Dict[str, str]:
    """Role-based casts: measure -> numeric, temporal -> datetime."""
    casts: Dict[str, str] = {}
    for col in understanding.columns:
        if col.name not in df.columns:
            continue
        series = df[col.name]
        if col.role == "measure" and not pd.api.types.is_numeric_dtype(
                series):
            converted = pd.to_numeric(series, errors="coerce")
            casts[col.name] = f"{series.dtype}->float64"
            df[col.name] = converted
            log.append({"op": "type_cast", "column": col.name,
                        "detail": casts[col.name]})
        elif col.role == "temporal" and not (
                pd.api.types.is_datetime64_any_dtype(series)):
            converted = pd.to_datetime(series, errors="coerce")
            casts[col.name] = f"{series.dtype}->datetime64"
            df[col.name] = converted
            log.append({"op": "type_cast", "column": col.name,
                        "detail": casts[col.name]})
    return casts


def _flag_column(df: pd.DataFrame, column: str,
                 log: List[Dict[str, Any]]) -> str:
    flag = f"{column}_missing_flag"
    df[flag] = df[column].isna()
    log.append({"op": "flag_column", "column": column, "flag": flag,
                "rows_affected": int(df[flag].sum())})
    return flag


def _iqr_bounds(series: pd.Series) -> Tuple[float, float] | None:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    if pd.isna(q1) or pd.isna(q3):
        return None
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def apply_iqr_outliers(df: pd.DataFrame, outliers: Dict[str, str],
                       log: List[Dict[str, Any]]) -> Dict[str, int]:
    """IQR outlier handling per strategy: 'flag' creates a `*_outlier_flag`,
    'drop' removes the rows. Only numeric columns are touched."""
    result: Dict[str, int] = {}
    for column, mode in outliers.items():
        if column not in df.columns:
            continue
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            log.append({"op": "iqr_outlier", "column": column,
                        "detail": "skipped_non_numeric"})
            continue
        bounds = _iqr_bounds(series)
        if bounds is None:
            log.append({"op": "iqr_outlier", "column": column,
                        "detail": "skipped_no_variance"})
            continue
        lo, hi = bounds
        mask = (series < lo) | (series > hi)
        count = int(mask.sum())
        result[column] = count
        if mode == "flag":
            df[f"{column}_outlier_flag"] = mask
        elif mode == "drop":
            df.drop(df.index[mask], inplace=True)
        log.append({"op": f"iqr_outlier_{mode}", "column": column,
                    "rows_affected": count})
    return result


def execute_strategy(df: pd.DataFrame, strategy: Dict[str, Any],
                     understanding: DatasetUnderstanding
                     ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Apply a normalized strategy. Returns (df, op_log). Never invents data."""
    log: List[Dict[str, Any]] = []
    df = df.copy()

    type_casts = _cast_types(df, understanding, log)

    if strategy.get("deduplicate"):
        before = len(df)
        df = df.drop_duplicates()
        log.append({"op": "dedup", "rows_affected": before - len(df)})

    flags_created: List[str] = []
    for entry in strategy.get("columns") or []:
        name = entry["column"]
        if name not in df.columns:
            continue
        action = entry["action"]
        series = df[name]
        if action == "keep":
            continue
        elif action == "median_fill":
            median = pd.to_numeric(series, errors="coerce").median()
            if not pd.isna(median):
                df[name] = pd.to_numeric(series, errors="coerce") \
                    .fillna(median)
                log.append({"op": "fillna_median", "column": name})
        elif action == "median_fill_flag":
            median = pd.to_numeric(series, errors="coerce").median()
            flags_created.append(_flag_column(df, name, log))
            if not pd.isna(median):
                df[name] = pd.to_numeric(series, errors="coerce") \
                    .fillna(median)
                log.append({"op": "fillna_median", "column": name})
        elif action == "mode_fill":
            mode = series.mode(dropna=True)
            if len(mode):
                df[name] = series.fillna(mode.iloc[0])
                log.append({"op": "fillna_mode", "column": name})
        elif action == "unknown_fill":
            df[name] = series.fillna("Unknown")
            log.append({"op": "fillna_unknown", "column": name,
                        "rows_affected": int(series.isna().sum())})
        elif action == "flag_and_preserve":
            flags_created.append(_flag_column(df, name, log))
        elif action == "keep_flag":
            flags_created.append(_flag_column(df, name, log))
        elif action == "drop_row":
            rows = int(series.isna().sum())
            df = df[df[name].notna()]
            log.append({"op": "drop_row", "column": name,
                        "rows_affected": rows})
        elif action == "drop_negative":
            numeric = pd.to_numeric(series, errors="coerce")
            mask = numeric < 0
            rows = int(mask.sum())
            df = df[~mask]
            log.append({"op": "drop_negative", "column": name,
                        "rows_affected": rows})
        elif action == "sanitize_text":
            import re as _re
            sql_pattern = _re.compile(
                r"(?:;|'|\b(?:DROP|DELETE|INSERT|UPDATE|SELECT|UNION|"
                r"ALTER|CREATE|EXEC|EXECUTE)\b)", _re.IGNORECASE)
            original = df[name].astype(str)
            sanitized = original.str.replace(sql_pattern, '', regex=True)
            n_changed = int((original != sanitized).sum())
            df[name] = sanitized
            log.append({"op": "sanitize_text", "column": name,
                        "rows_affected": n_changed})
        elif action == "drop_column":
            df = df.drop(columns=[name])
            log.append({"op": "drop_column", "column": name})

    # 3.1: the executor is authoritative — even a strategy that bypassed
    # normalize_strategy never flags/drops outliers on non-measure columns.
    guarded_outliers: Dict[str, str] = {}
    for column, mode in (strategy.get("outliers") or {}).items():
        meta = next((c for c in understanding.columns
                     if c.name == column), None)
        if meta is not None and (meta.role in ("identifier", "categorical")
                                 or IDENTIFIER_NAME_RE.search(column)):
            log.append({"op": "iqr_outlier", "column": column,
                        "detail": "skipped_not_measure"})
            continue
        guarded_outliers[column] = mode
    outliers = apply_iqr_outliers(df, guarded_outliers, log)

    return df, log


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def persist_attempt(run_dir: Path, df: pd.DataFrame, attempt: int) -> Path:
    """Write cleaned data with the v4.3 lineage trail: `cleaned_data.csv` is
    always the latest attempt; previous attempts are preserved as
    `cleaned_data_attempt_<n>.csv` (never overwritten silently)."""
    processed = Path(run_dir) / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    latest = processed / "cleaned_data.csv"
    if attempt > 1 and latest.exists():
        kept = processed / f"cleaned_data_attempt_{attempt - 1}.csv"
        if not kept.exists():
            shutil.copyfile(latest, kept)
    df.to_csv(latest, index=False, encoding="utf-8-sig")
    return latest


def assemble_cleaning_result(attempt: int, rows_before: int,
                             rows_after: int, duplicates_removed: int,
                             type_casts: Dict[str, str],
                             flags_created: List[str],
                             outliers: Dict[str, int],
                             status: str = "passed") -> CleaningResult:
    """Build metadata/cleaning_result.json (CleaningResult)."""
    return CleaningResult(
        attempt=attempt,
        rows_before=rows_before,
        rows_after=rows_after,
        duplicates_removed=duplicates_removed,
        type_casts=type_casts,
        flags_created=flags_created,
        outliers=outliers,
        status=status,
    )
