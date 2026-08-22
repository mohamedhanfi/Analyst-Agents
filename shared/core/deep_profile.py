"""Deep profiling: sentinel-aware missingness, robust MAD outliers, impact.

Implements improvement-plan items 3-5 for the DQ/Cleaning stages:

* Deep missingness — distinguishes genuine blanks from missing-like sentinels
  ("N/A", "null", "unknown", "-", "?"), reports zero-as-missing for measures,
  breaks missingness down by dimension segment and over time, flags columns
  whose missing flags correlate with each other, and assesses imputability
  per role.
* Outlier flags — robust MAD-modified z-scores (Iglewicz & Hoaglin, threshold
  3.5) plus context-aware within-segment outliers for dimension groups.
* Impact analysis — before/after deltas (rows, sum/mean/median per measure,
  KPI totals, cardinality) so the report always shows what was removed or
  changed and by how much.

Everything is deterministic and JSON-safe (numpy scalars unwrapped).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from shared.schemas import DatasetUnderstanding

# Missing-like sentinels found in raw files. Keys are the canonical token
# stored in the report; patterns are matched case-insensitively after strip.
MISSING_SENTINELS: Dict[str, Tuple[str, ...]] = {
    "blank": ("", " ", "\u00a0"),
    "na": ("na", "n/a", "n.a", "nan", "null", "none", "nil", "missing",
           "-", "--", "?", "??", "\u2014", "null"),
    "unknown": ("unknown", "unk", "not known", "tbd", "to be determined",
                "n.d", "n/k", "not available"),
}
# Anything in na/unknown counted as a sentinel; blank is the empty string.
_SENTINEL_RE = re.compile(
    r"^\s*(" + "|".join(
        re.escape(t) for tokens in MISSING_SENTINELS.values()
        for t in tokens) + r")\s*$", re.IGNORECASE)
_SENTINEL_TOKENS = {t: name
                    for name, tokens in MISSING_SENTINELS.items()
                    for t in tokens}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sentinel_token(value: Any) -> Optional[str]:
    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return "blank"
        return _SENTINEL_TOKENS.get(token)
    return None


def _is_zero_missing_candidate(role: str, series: pd.Series) -> bool:
    """Measures where 0 is suspicious (revenue/amount/qty) vs. legitimate
    (temperature/count/growth). Returns True when the column likely treats
    zeros as missing."""
    if role != "measure" or not pd.api.types.is_numeric_dtype(series):
        return False
    lowered = series.name.lower().replace(" ", "")
    if any(kw in lowered for kw in ("temperature", "growth", "margin",
                                    "balance", "lat", "lon", "count",
                                    "qty", "quantity", "rating", "score")):
        return False
    return True


def _imputability(role: str, rate: float) -> Dict[str, Any]:
    if role == "measure":
        if rate >= 0.5:
            verdict, why = "high_missing_drop", "over half missing - drop column"
        elif rate >= 0.1:
            verdict, why = "impute_median", "median imputation is safe"
        else:
            verdict, why = "low", "negligible missingness"
    elif role in ("dimension", "categorical"):
        verdict = ("impute_unknown_category" if rate < 0.5
                   else "high_missing_drop")
        why = ("mode/unknown-category label" if rate < 0.5
               else "over half missing - drop column")
    elif role == "identifier":
        verdict, why = "cannot_impute", "identifiers must not be invented"
    else:
        verdict, why = "cannot_impute", "temporal gaps cannot be imputed"
    return {"verdict": verdict, "reason": why}


def _co_missing_pairs(df: pd.DataFrame,
                      min_overlap: int = 3,
                      threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Columns whose missing flags overlap heavily (Jaccard on missing)."""
    pairs: List[Dict[str, Any]] = []
    cols = list(df.columns)
    for i, a in enumerate(cols):
        ma = df[a].isna()
        if int(ma.sum()) < min_overlap:
            continue
        for b in cols[i + 1:]:
            mb = df[b].isna()
            both = int((ma & mb).sum())
            if both < min_overlap:
                continue
            union = int((ma | mb).sum())
            jaccard = both / union if union else 0.0
            if jaccard >= threshold:
                pairs.append({"column_a": a, "column_b": b,
                              "co_missing": both,
                              "jaccard": round(jaccard, 3)})
    pairs.sort(key=lambda p: p["jaccard"], reverse=True)
    return pairs


def categorize_missing(series: pd.Series) -> Dict[str, int]:
    """Count missing-like sentinels and zeros in a column."""
    counts: Dict[str, int] = {"nan": int(series.isna().sum()), "blank": 0,
                              "na": 0, "unknown": 0, "zero": 0}
    for value in series.dropna():
        token = _sentinel_token(value)
        if token:
            counts[token] += 1
        elif (isinstance(value, (int, float, np.integer, np.floating))
              and not isinstance(value, bool) and value == 0):
            counts["zero"] += 1
    return counts


def _by_segment_missing(df: pd.DataFrame, column: str,
                        segments: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for seg in segments:
        if seg not in df.columns or seg == column \
                or df[seg].nunique() > 20:
            continue
        grouped = df.groupby(seg, dropna=False)[column].apply(
            lambda s: round(float(s.isna().mean()), 4)).to_dict()
        out[seg] = {str(k): v for k, v in grouped.items()}
    return out


def _time_trend(df: pd.DataFrame, temporal_col: str,
                columns: List[str]) -> Dict[str, Any]:
    if temporal_col not in df.columns or df[temporal_col].isna().all():
        return {}
    parsed = pd.to_datetime(df[temporal_col], errors="coerce")
    period = parsed.dt.to_period("M")
    trend: Dict[str, Any] = {}
    for col in columns:
        if col not in df.columns or col == temporal_col:
            continue
        rates = (df[col].isna().astype(int)
                 .groupby(period, dropna=False).mean()
                 .round(4).to_dict())
        trend[col] = {str(k): float(v) for k, v in rates.items()}
    return trend


def deep_missingness_report(understanding: DatasetUnderstanding,
                            df: pd.DataFrame,
                            ) -> Dict[str, Any]:
    """Per-column missingness: sentinels, zero-as-missing, segment + time
    breakdown, co-missing pairs, and role-aware imputability."""
    segments = [c.name for c in understanding.columns
                if c.role in ("dimension", "categorical", "identifier")]
    temporal = [c.name for c in understanding.columns
                if c.role == "temporal"]
    temporal_col = temporal[0] if temporal else None

    by_column: Dict[str, Any] = {}
    for col in understanding.columns:
        name = col.name
        if name not in df.columns:
            continue
        series = df[name]
        counts = categorize_missing(series)
        missing = int(series.isna().sum())
        rate = missing / len(series) if len(series) else 0.0
        sentinel_total = counts["blank"] + counts["na"] + counts["unknown"]
        by_column[name] = {
            "missing": missing,
            "rate": round(rate, 6),
            "sentinel_counts": {k: v for k, v in counts.items()
                                if k != "nan"},
            "sentinel_total": sentinel_total,
            "effective_missing": missing + sentinel_total,
            "effective_rate": round((missing + sentinel_total) / len(series),
                                    6) if len(series) else 0.0,
            "imputability": _imputability(col.role, rate),
            "by_segment": _by_segment_missing(df, name, segments),
        }

    return {
        "assessment": _overall_assessment(by_column),
        "by_column": by_column,
        "co_missing_pairs": _co_missing_pairs(df),
        "time_trend": _time_trend(df, temporal_col,
                                  [c.name for c in understanding.columns])
        if temporal_col else {},
    }


def _overall_assessment(by_column: Dict[str, Any]) -> str:
    eff = [c["effective_rate"] for c in by_column.values()]
    if not eff:
        return "none"
    worst = max(eff)
    if worst >= 0.5:
        return "high"
    if worst >= 0.1:
        return "moderate"
    if any(c["effective_rate"] > 0 for c in by_column.values()):
        return "low"
    return "none"


# ---------------------------------------------------------------------------
# Robust outliers (MAD modified z-score) + context-aware segment outliers
# ---------------------------------------------------------------------------

MAD_THRESHOLD = 3.5


def _modified_z(valid: pd.Series) -> Tuple[pd.Series, float]:
    median = float(valid.median())
    mad = float((valid - median).abs().median()) or 1e-12
    return 0.6745 * (valid - median) / mad, mad


def _outlier_rows(numeric: pd.Series, threshold: float = MAD_THRESHOLD,
                  ) -> List[Dict[str, Any]]:
    valid_mask = numeric.notna()
    if int(valid_mask.sum()) < 10:
        return []
    z, _ = _modified_z(numeric[valid_mask])
    flagged = z.abs() > threshold
    rows = numeric.index[valid_mask][flagged]
    out = []
    for i in rows:
        out.append({"row_index": int(i), "value": _jsonable(numeric[i]),
                    "z_score": round(float(z.loc[i]), 3)})
    out.sort(key=lambda o: o["z_score"], reverse=True)
    return out


def deep_outlier_report(understanding: DatasetUnderstanding,
                        df: pd.DataFrame,
                        ) -> Dict[str, Any]:
    """MAD modified-z outlier flags per measure + within-segment outliers."""
    segments = [c.name for c in understanding.columns
                if c.role in ("dimension", "categorical", "identifier")]
    report: Dict[str, Any] = {}
    for col in understanding.columns:
        if col.role != "measure" or col.name not in df.columns:
            continue
        numeric = pd.to_numeric(df[col.name], errors="coerce")
        valid = numeric.dropna()
        if len(valid) < 10:
            report[col.name] = {"flag": "insufficient_data",
                                "n": int(len(valid))}
            continue
        rows = _outlier_rows(numeric)
        report[col.name] = {
            "flag": "clean" if not rows else f"outliers_mad_x{len(rows)}",
            "n_outliers": len(rows),
            "worst": rows[:5],
            "threshold": "mad_modified_z_3.5",
            "by_segment": _segment_outliers(df, col.name, numeric, segments),
        }
    return report


def _segment_outliers(df: pd.DataFrame, measure: str, numeric: pd.Series,
                      segments: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for seg in segments:
        if seg not in df.columns or df[seg].nunique() > 20:
            continue
        seg_out: Dict[str, Any] = {}
        for value, idx in df.groupby(seg, dropna=False).groups.items():
            values = numeric.loc[idx]
            rows = _outlier_rows(values)
            if rows:
                seg_out[str(value)] = {"n_outliers": len(rows),
                                       "worst": rows[:3]}
        if seg_out:
            out[seg] = seg_out
    return out


# ---------------------------------------------------------------------------
# Impact analysis: what changed between two frames, per measure + KPI
# ---------------------------------------------------------------------------

def impact_analysis(understanding: DatasetUnderstanding,
                    before: pd.DataFrame,
                    after: pd.DataFrame,
                    ) -> Dict[str, Any]:
    """Before/after deltas: rows, per-measure sum/mean/median, KPI total,
    dimension cardinality. Pure comparison — never mutates."""
    measures = [c.name for c in understanding.columns
                if c.role == "measure" and c.name in before.columns]
    dims = [c.name for c in understanding.columns
            if c.role in ("dimension", "categorical", "identifier")
            and c.name in before.columns]

    per_measure: Dict[str, Any] = {}
    for m in measures:
        b = pd.to_numeric(before[m], errors="coerce")
        a = pd.to_numeric(after[m], errors="coerce")
        b_sum = float(b.sum()) if b.notna().any() else 0.0
        a_sum = float(a.sum()) if a.notna().any() else 0.0
        b_mean = float(b.mean()) if b.notna().any() else None
        a_mean = float(a.mean()) if a.notna().any() else None
        b_median = float(b.median()) if b.notna().any() else None
        a_median = float(a.median()) if a.notna().any() else None
        per_measure[m] = {
            "rows_before": int(b.notna().sum()),
            "rows_after": int(a.notna().sum()),
            "sum_before": round(b_sum, 6),
            "sum_after": round(a_sum, 6),
            "sum_delta": round(a_sum - b_sum, 6),
            "sum_delta_pct": round(
                (a_sum - b_sum) / b_sum * 100, 4) if b_sum else None,
            "mean_before": round(b_mean, 6) if b_mean is not None else None,
            "mean_after": round(a_mean, 6) if a_mean is not None else None,
            "median_before": round(b_median, 6)
            if b_median is not None else None,
            "median_after": round(a_median, 6)
            if a_median is not None else None,
        }

    cardinality = {d: {"before": int(before[d].nunique()),
                       "after": int(after[d].nunique())}
                   for d in dims}

    kpi = None
    if measures:
        m = measures[0]
        b_sum = per_measure[m]["sum_before"]
        a_sum = per_measure[m]["sum_after"]
        kpi = {"column": m, "sum_before": b_sum, "sum_after": a_sum,
               "delta": round(a_sum - b_sum, 6),
               "delta_pct": round((a_sum - b_sum) / b_sum * 100, 4)
               if b_sum else None}

    return {
        "rows_before": int(len(before)),
        "rows_after": int(len(after)),
        "rows_removed": int(len(before) - len(after)),
        "kpi": kpi,
        "by_measure": per_measure,
        "dimension_cardinality": cardinality,
    }