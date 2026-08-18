"""Shared business-relevance gates (semantic sweep §0).

One canonical place for "is this numeric column actually meaningful" —
every stage (KPI engine, correlation engine, chart planner, insights, QA)
calls into this module instead of growing its own copy, so the
identifier/code/ordinal/unit fixes stay in sync pipeline-wide.

Deterministic heuristics only; no I/O; import-light (pandas is imported
locally). The ambiguous bands are left to the LLM fallbacks in the agents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

# Task A identifier patterns: value-aggregation on these is meaningless.
IDENTIFIER_NAME_RE = re.compile(
    r"phone|tel|mobile|fax|zip|postal|ssn|national_id|passport|iban|"
    r"account_no|card_number|\bid\b|_id$|^id_|code$|رقم|هاتف|جوال",
    re.IGNORECASE)

# Continuous-metric names (2.2): the code-like check must never downgrade
# these to categorical just because they are low-cardinality integers.
CONTINUOUS_METRIC_NAME_RE = re.compile(
    r"amount|price|pricing|qty|quantity|count|total|revenue|income|"
    r"cost|expense|spend|sales|profit|duration|days|hours|minutes|"
    r"age|value|volume|weight|mass|size|length|width|height|"
    r"balance|margin|rate|interest", re.IGNORECASE)

# Ordinal-scale names (2.3): low-cardinality integers that CAN be averaged
# (mean satisfaction of 3.4) but need different interpretation framing.
ORDINAL_NAME_RE = re.compile(
    r"rating|satisfaction|score|rank|level|priority|grade|stage|nps|"
    r"star|tier", re.IGNORECASE)

# Names where negative values are semantically valid (3.2): temperatures,
# geo coordinates, balances, deltas, growth/margins, returns.
NEGATIVE_ALLOWED_RE = re.compile(
    r"temp|celsius|fahrenheit|latitude|longitude|altitude|elevation|"
    r"balance|delta|difference|diff|growth|margin|return", re.IGNORECASE)

_CURRENCY = r"(?:\$|€|£|EGP|USD|EUR|SAR|AED|KWD|QAR|BHD|OMR|LE)"
_UNITS = (r"(?:kg|kgs?|g|gm|lbs?|oz|cm|mm|km|m|ml|l|mg|%|pct|inch|ft|yd|"
          r"sqm|m²)")
_UNIT_VALUE_RE = re.compile(
    rf"^\s*(?:{_CURRENCY}\s*)?[0-9][0-9.,]*\s*(?:{_UNITS}|{_CURRENCY})?\s*$",
    re.IGNORECASE)
_PLAIN_NUMBER_RE = re.compile(r"^\s*[0-9][0-9.,]*\s*$")


@dataclass
class IdentifierSignal:
    """Result of is_identifier_like: confidence 0-1 + human-readable
    signals explaining the score (audited into dataset_understanding.json)."""
    score: float
    signals: List[str]


def is_identifier_like(column_name: str,
                       series: Union["pd.Series", None] = None
                       ) -> IdentifierSignal:
    """Semantic pre-check (task A.1): is a numeric column really a measure,
    or an ID-like value that must never be aggregated?

    Runs BEFORE any numeric column defaults to measure. Combines a name
    pattern match with value-shape signals (digit-length consistency,
    integral values, unique ratio, leading zeros). Returns a confidence
    score; callers force role=identifier above the configured threshold
    and may consult the LLM in the ambiguous band (0.3–threshold).
    """
    import pandas as pd  # local: keeps this module import-light

    signals: List[str] = []
    name = (column_name or "").strip().lower()
    score = 0.0
    if name and IDENTIFIER_NAME_RE.search(name):
        signals.append("name matches identifier pattern")
        score += 0.75

    if series is not None:
        vals = pd.Series(series).dropna()
        if len(vals) == 0:
            return IdentifierSignal(min(score, 1.0), signals)
        numeric = pd.to_numeric(vals, errors="coerce").dropna()
        if len(numeric) > 0:
            if float((numeric % 1 == 0).mean()) >= 0.99:
                signals.append("all values integral (no meaningful sum)")
                score += 0.2
            try:
                lengths = numeric.abs().apply(lambda v: len(str(int(v))))
            except (ValueError, OverflowError):  # pragma: no cover
                lengths = None
            if lengths is not None and len(lengths) > 0:
                med = float(lengths.median())
                if float((abs(lengths - med) <= 1).mean()) >= 0.9:
                    signals.append("digit length nearly constant")
                    score += 0.2
            if float(numeric.nunique()) / len(numeric) > 0.9:
                signals.append("high cardinality (unique ratio > 0.9)")
                score += 0.15
        str_vals = vals.astype(str).str.strip()
        if str_vals.str.match(r"^0[0-9]+$").any():
            signals.append("leading zeros in original values")
            score += 0.4
    return IdentifierSignal(min(score, 1.0), signals)


def is_code_like(column_name: str,
                 series: Union["pd.Series", Any]) -> bool:
    """Encoded categoricals (2.2): low-cardinality all-integer columns whose
    name is neither a continuous metric nor an ordinal scale — gender 0/1,
    status 1/2/3 — are categorical, never measures.

    All-unique integers (unique ratio > 0.9) are identifier-like, not codes,
    so phone numbers never downgrade to categorical on small files.
    """
    if series is None:
        return False
    import pandas as pd

    vals = pd.Series(series).dropna()
    if len(vals) == 0:
        return False
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric.isna().any():           # any unparseable value -> not a code
        return False
    if float((numeric % 1 == 0).mean()) < 0.99:
        return False
    if int(numeric.nunique()) > 10:
        return False
    if float(numeric.nunique()) / len(numeric) > 0.9:
        return False                   # all-unique ints are IDs, not codes
    name = (column_name or "").strip().lower()
    if CONTINUOUS_METRIC_NAME_RE.search(name):
        return False
    if ORDINAL_NAME_RE.search(name):
        return False
    return True


def is_ordinal_like(column_name: str,
                    series: Union["pd.Series", Any]) -> bool:
    """Ordinal scales (2.3): 3-20 distinct all-integer levels on a
    rating/satisfaction/score/rank...-named column. Stays a measure (it can
    be averaged) but carries the ordinal flag for correct interpretation."""
    if series is None:
        return False
    import pandas as pd

    vals = pd.Series(series).dropna()
    if len(vals) == 0:
        return False
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric.isna().any():
        return False
    if float((numeric % 1 == 0).mean()) < 0.99:
        return False
    if not (3 <= int(numeric.nunique()) <= 20):
        return False
    return bool(ORDINAL_NAME_RE.search((column_name or "").strip().lower()))


def is_mixed_unit(series: Union["pd.Series", Any]) -> bool:
    """Mixed currency/unit strings (2.4): values like "$100", "EGP 500",
    "100 kg" — a signal that numeric coercion would silently strip units."""
    if series is None:
        return False
    import pandas as pd

    vals = pd.Series(series).dropna()
    if len(vals) == 0:
        return False
    str_vals = vals.astype(str).str.strip()
    if str_vals.str.match(_PLAIN_NUMBER_RE.pattern).all():
        return False
    return bool(str_vals.str.match(_UNIT_VALUE_RE.pattern).any())


_VALUE_AGGS = frozenset({"sum", "mean", "avg", "median", "std", "min",
                         "max", "correlation"})
_CODE_AGGS = frozenset({"sum", "mean", "avg"})


def aggregation_is_meaningful(column_name: str,
                              series: Union["pd.Series", Any],
                              agg_type: str) -> Tuple[bool, str]:
    """Single gate (§0): may this value-aggregation run on this column?

    Identifiers can always be counted/nuniqued but never value-aggregated;
    code-like categoricals are never summed or averaged. Returns
    (ok, reason) — callers drop the operation when not ok.
    """
    agg = (agg_type or "").lower()
    name = (column_name or "").strip()
    if agg in _VALUE_AGGS and name and IDENTIFIER_NAME_RE.search(name):
        return False, "identifier-like column"
    if agg in _CODE_AGGS and is_code_like(name, series):
        return False, "code-like categorical"
    return True, ""