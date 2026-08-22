"""Stage 3b — data contracts + normalization layer (§4.2).

A column contract is the declared (initially heuristic, later user-
confirmable) shape of a column: expected type, nullability, allowed range,
forbidden sentinel values, uniqueness, unit/currency and format hint. The
contract is what "quality" is measured against, instead of relying on the
column name or dtype alone.

The normalization layer re-expresses the same values in a canonical form —
it NEVER invents data: it parses currency/percent strings to numbers,
normalizes whitespace/case/Unicode for categories, parses dates, and
converts units only when the canonical unit is derivable from the column
name and the conversion is a pure scale factor (kg<->g, km<->m, ...). Every
change is logged so the cleaning step's lineage records it.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from shared.schemas import DatasetUnderstanding

# Values commonly used as "missing" placeholders — flagged, never silently
# dropped (Step 3 deep-missingness distinguishes blank/N-A/unknown/null).
SENTINELS = frozenset({
    "", "nan", "n/a", "na", "null", "none", "nil", "unknown", "?", "-",
    "#n/a", "n.a.", "na.", "undefined",
})

_CURRENCY_SYMBOLS = re.compile(r"[$€£]|EGP|USD|EUR|SAR|AED|KWD|QAR|BHD|OMR|LE|JPY|GBP|CNY|JOD",
                               re.IGNORECASE)
_PERCENT_HINT = re.compile(r"percent|pct|%|ratio|rate\b|margin|discount|growth",
                           re.IGNORECASE)
_WEIGHT_HINT = re.compile(r"\b(weight|mass|kg|kgs|kilogram|g\b|grams)\b",
                          re.IGNORECASE)
_DISTANCE_HINT = re.compile(r"\b(distance|length|km|kilometer|km|m\b|mm|cm)\b",
                            re.IGNORECASE)
_VOLUME_HINT = re.compile(r"\b(volume|liter|litre|l\b|ml|gal)\b", re.IGNORECASE)

# pure scale-factor conversions only (never cross-currency — needs a rate)
_SCALE_FACTORS = {
    "kg": {"g": 0.001, "mg": 1e-6, "lbs": 0.45359237, "lb": 0.45359237,
           "oz": 0.028349523},
    "g": {"kg": 1000.0, "mg": 0.001, "lbs": 453.59237, "oz": 28.349523},
    "lbs": {"kg": 2.2046226218, "g": 0.0022046226218, "oz": 16.0},
    "oz": {"kg": 35.27396195, "g": 0.03527396195, "lbs": 0.0625},
    "km": {"m": 0.001, "cm": 1e-5, "mm": 1e-6},
    "m": {"km": 1000.0, "cm": 0.01, "mm": 0.001},
    "cm": {"m": 100.0, "mm": 0.1, "km": 100000.0},
    "l": {"ml": 0.001},
    "ml": {"l": 1000.0},
}

_UNIT_NAME_TO_CANONICAL = {
    "kg": "kg", "kgs": "kg", "kilogram": "kg", "kilograms": "kg",
    "g": "g", "gm": "g", "grams": "g", "gram": "g",
    "km": "km", "kilometer": "km", "kilometers": "km",
    "m": "m", "meter": "m", "meters": "m", "metre": "m",
    "cm": "cm", "centimeter": "cm", "mm": "mm", "millimeter": "mm",
    "l": "l", "liter": "l", "litre": "l", "liters": "l",
    "ml": "ml", "milliliter": "ml", "millilitre": "ml",
    "lbs": "lbs", "lb": "lbs", "pound": "lbs", "pounds": "lbs",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
}


@dataclass
class ColumnContract:
    """Declared shape of one column (initially heuristic)."""
    column: str
    expected_type: str            # numeric | date | categorical | id | free_text
    nullable: bool = True
    allowed_min: Optional[float] = None
    allowed_max: Optional[float] = None
    forbidden_values: List[str] = field(default_factory=list)
    unique: bool = False
    unit: Optional[str] = None
    format_hint: Optional[str] = None
    source: str = "heuristic"     # heuristic | user

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column, "expected_type": self.expected_type,
            "nullable": self.nullable, "allowed_min": self.allowed_min,
            "allowed_max": self.allowed_max,
            "forbidden_values": list(self.forbidden_values),
            "unique": self.unique, "unit": self.unit,
            "format_hint": self.format_hint, "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnContract":
        return cls(
            column=str(data.get("column", "")),
            expected_type=str(data.get("expected_type", "categorical")),
            nullable=bool(data.get("nullable", True)),
            allowed_min=data.get("allowed_min"),
            allowed_max=data.get("allowed_max"),
            forbidden_values=list(data.get("forbidden_values") or []),
            unique=bool(data.get("unique", False)),
            unit=data.get("unit"),
            format_hint=data.get("format_hint"),
            source=str(data.get("source", "heuristic")),
        )


# ---------------------------------------------------------------------------
# Amount / unit parsing
# ---------------------------------------------------------------------------

_AMOUNT_RE = re.compile(
    r"^\s*(?P<cur_pre>[$\u20ac\u00a3]|EGP|USD|EUR|SAR|AED|KWD|QAR|BHD|OMR|LE|JPY|GBP|CNY|JOD)?\s*"
    r"(?P<num>[0-9][0-9., ]*)\s*(?P<unit>kg|kgs?|g|gm|lbs?|oz|cm|mm|km|m|ml|l|k|m|%|pct)?\s*"
    r"(?P<cur_post>[$\u20ac\u00a3]|EGP|USD|EUR|SAR|AED|KWD|QAR|BHD|OMR|LE|JPY|GBP|CNY|JOD)?\s*$",
    re.IGNORECASE)


def parse_amount(value: Any) -> Optional[float]:
    """Parse a human-formatted amount to a plain float.

    Handles '$1,200', 'EGP 1,200', '1.2k', '12%', '1 200', '500 g', '€9.99'.
    Percent strings stay as percent POINTS (12% -> 12.0), matching the
    over-100-percent DQ check. Returns None when the value is not an amount.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number
    text = str(value).strip()
    if not text:
        return None
    match = _AMOUNT_RE.match(text)
    if not match:
        return None
    num = match.group("num")
    suffix = (match.group("unit") or "").lower()
    multiplier = 1.0
    if suffix in ("k", "m"):
        multiplier = 1000.0 if suffix == "k" else 1000000.0
        suffix = ""
    digits = num.replace(",", "").replace(" ", "")
    if "." in digits:
        try:
            return float(digits) * multiplier
        except ValueError:
            return None
    if digits.isdigit():
        return float(digits) * multiplier
    return None


def _canonical_unit_from_name(name: str) -> Optional[str]:
    lowered = name.lower().replace("_", " ").replace("-", " ")
    if "percent" in lowered or "pct" in lowered or "%" in name:
        return "%"
    for key in ("kg", "kgs", "kilogram"):
        if key in lowered:
            return "kg"
    if re.search(r"\b(g|gm|grams?)\b", lowered):
        return "g"
    for key in ("km", "kilometer"):
        if key in lowered:
            return "km"
    if re.search(r"\b(meter|metre|m)\b", lowered):
        return "m"
    if re.search(r"\b(cm|centimeter)\b", lowered):
        return "cm"
    if re.search(r"\b(mm|millimeter)\b", lowered):
        return "mm"
    if re.search(r"\b(milliliters?|ml)\b", lowered):
        return "ml"
    if re.search(r"\b(liters?|litres?|l)\b", lowered):
        return "l"
    if re.search(r"\b(lbs?|pounds?)\b", lowered):
        return "lbs"
    if re.search(r"\b(usd|dollars?)\b", lowered):
        return "USD"
    if re.search(r"\b(egp|pounds?)\b", lowered) or name.upper() == "EGP":
        return "EGP"
    for cur in ("EUR", "SAR", "AED", "KWD", "QAR", "BHD", "OMR", "JPY", "GBP",
                "CNY", "JOD"):
        if cur in name.upper():
            return cur
    return None


def _unit_of_amount(text: str) -> Optional[str]:
    match = _AMOUNT_RE.match(text.strip())
    if not match:
        return None
    unit = match.group("unit")
    if unit and unit.lower() not in ("k", "m"):
        return _UNIT_NAME_TO_CANONICAL.get(unit.lower(), unit.lower())
    cur = match.group("cur_pre") or match.group("cur_post")
    if cur:
        mapping = {"$": "USD", "€": "EUR", "£": "GBP", "EGP": "EGP",
                   "USD": "USD", "EUR": "EUR", "SAR": "SAR", "AED": "AED",
                   "KWD": "KWD", "QAR": "QAR", "BHD": "BHD", "OMR": "OMR",
                   "LE": "EGP", "JPY": "JPY", "GBP": "GBP", "CNY": "CNY",
                   "JOD": "JOD"}
        return mapping.get(cur.upper())
    return None


# ---------------------------------------------------------------------------
# Contract generation (heuristic)
# ---------------------------------------------------------------------------


def build_contracts(understanding: DatasetUnderstanding,
                    df: pd.DataFrame) -> List[ColumnContract]:
    """Generate a heuristic contract per column from the understanding.

    The user may later confirm/override it (``source='user'``); until then
    it is advisory — the DQ checks already encode the hard rules, and the
    contract adds ranges, sentinels, uniqueness and units.
    """
    contracts: List[ColumnContract] = []
    for col in understanding.columns:
        name = col.name
        series = df[name] if name in df.columns else pd.Series(dtype=object)
        missing_rate = float(series.isna().mean()) if len(series) else 0.0
        nunique = int(series.nunique(dropna=True)) if len(series) else 0
        nullable = missing_rate > 0.0
        role = col.role

        if role == "measure":
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric) < 2:
                # currency/percent strings are not numeric yet — parse them
                numeric = pd.Series(
                    [parse_amount(v) for v in series.dropna()]).dropna()
            if len(numeric) >= 2:
                lo = float(numeric.min())
                hi = float(numeric.max())
                # widen the range slightly so legitimate extremes survive
                span = hi - lo
                lo, hi = lo - 0.05 * span, hi + 0.05 * span
            else:
                lo = hi = None
            unit = _canonical_unit_from_name(name)
            contracts.append(ColumnContract(
                column=name, expected_type="numeric", nullable=nullable,
                allowed_min=lo, allowed_max=hi, unit=unit,
                forbidden_values=[], source="heuristic"))
        elif role == "temporal":
            sample = [str(v) for v in series.dropna().head(3)]
            fmt = "%Y-%m-%d"
            if sample and all(re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", s)
                              for s in sample):
                fmt = "%m/%d/%Y"
            contracts.append(ColumnContract(
                column=name, expected_type="date", nullable=nullable,
                format_hint=fmt, source="heuristic"))
        elif role == "identifier":
            contracts.append(ColumnContract(
                column=name, expected_type="id", nullable=nullable,
                unique=nunique == len(series.dropna()),
                source="heuristic"))
        elif role in ("dimension", "categorical"):
            contracts.append(ColumnContract(
                column=name, expected_type="categorical", nullable=nullable,
                unique=False, source="heuristic"))
        else:  # free_text
            contracts.append(ColumnContract(
                column=name, expected_type="free_text", nullable=nullable,
                source="heuristic"))
    return contracts


def validate_contracts(contracts: List[ColumnContract], df: pd.DataFrame
                       ) -> List[Dict[str, Any]]:
    """Check a dataframe against its contracts. Returns violations:
    [{"column", "kind", "detail", "rows"}]. Advisory — the report surfaces
    them; cleaning decides. Never mutates the frame."""
    violations: List[Dict[str, Any]] = []
    for contract in contracts:
        name = contract.column
        if name not in df.columns:
            continue
        series = df[name]

        if contract.allowed_min is not None or contract.allowed_max is not None:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            lo = contract.allowed_min
            hi = contract.allowed_max
            if lo is not None:
                count = int((numeric < lo).sum())
                if count:
                    violations.append({"column": name,
                                       "kind": "below_contract_range",
                                       "detail": f"< {lo}", "rows": count})
            if hi is not None:
                count = int((numeric > hi).sum())
                if count:
                    violations.append({"column": name,
                                       "kind": "above_contract_range",
                                       "detail": f"> {hi}", "rows": count})

        if contract.forbidden_values:
            str_vals = series.dropna().astype(str).str.strip().str.lower()
            hits = str_vals.isin(contract.forbidden_values).sum()
            if hits:
                violations.append({"column": name, "kind": "forbidden_value",
                                   "detail": ",".join(
                                       contract.forbidden_values[:5]),
                                   "rows": int(hits)})

        if contract.unique:
            non_null = series.dropna()
            if non_null.duplicated().any():
                violations.append({"column": name, "kind": "uniqueness",
                                   "detail": "duplicated values",
                                   "rows": int(non_null.duplicated().sum())})

        if not contract.nullable:
            count = int(series.isna().sum())
            if count:
                violations.append({"column": name, "kind": "unexpected_null",
                                   "detail": "nulls in non-nullable column",
                                   "rows": count})
    return violations


# ---------------------------------------------------------------------------
# Normalization layer (deterministic, non-destructive)
# ---------------------------------------------------------------------------


def _nfc_normalize(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    return value


def _collapse_ws(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", value.strip())


def normalize_columns(df: pd.DataFrame,
                      understanding: DatasetUnderstanding,
                      contracts: Optional[List[ColumnContract]] = None
                      ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Canonicalize units/currency/percent/dates/categories in place.

    Returns (normalized_df, ops). Operations are logged per column so the
    cleaning lineage records them (deterministic, versioned, no invented
    values). Columns the understanding never sees are left untouched.
    """
    work = df.copy()
    ops: List[Dict[str, Any]] = []
    contracts = contracts or build_contracts(understanding, work)
    by_column = {c.column: c for c in contracts}

    for col in understanding.columns:
        name = col.name
        if name not in work.columns:
            continue
        series = work[name]
        contract = by_column.get(name)

        # --- measures: currency / percent / thousands / k-m suffixes -------
        if col.role == "measure" and not pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) == 0:
                continue
            parsed = non_null.map(parse_amount)
            ok = int(parsed.notna().sum())
            if ok / len(non_null) >= 0.95:
                changed = int((parsed.astype(float) !=
                               pd.to_numeric(non_null, errors="coerce")).sum())
                if changed > 0:
                    work[name] = series.map(
                        lambda v: parse_amount(v)
                        if v is not None and not isinstance(v, (int, float))
                        else v)
                    work[name] = pd.to_numeric(work[name], errors="coerce")
                    unit = _unit_of_amount(
                        str(non_null.iloc[0])) if len(non_null) else None
                    ops.append({"op": "parse_amount", "column": name,
                                "rows_affected": changed,
                                "detail": f"unit={unit or 'none'}",
                                "before": "string", "after": "float64"})
            else:
                ops.append({"op": "parse_amount", "column": name,
                            "rows_affected": 0,
                            "detail": "skipped_less_than_95pct_parseable"})

        # --- measures: pure scale-factor unit conversion (kg<->g, ...) -----
        if col.role == "measure" and pd.api.types.is_numeric_dtype(
                work[name]):
            canonical = (contract.unit if contract else None) \
                or _canonical_unit_from_name(name)
            if canonical and canonical in _SCALE_FACTORS:
                text_samples = [
                    str(v) for v in df[name].dropna().head(20)
                    if isinstance(v, str)]
                converted = 0
                new_series = work[name].astype(float)
                for raw in text_samples:
                    u = _unit_of_amount(raw)
                    factor = (_SCALE_FACTORS[canonical].get(u)
                              if u else None)
                    if factor is None or u == canonical:
                        continue
                    mask = df[name].astype(str) == raw
                    new_series.loc[mask] = parse_amount(raw) * factor
                    converted += int(mask.sum())
                if converted:
                    work[name] = new_series
                    ops.append({"op": "convert_unit", "column": name,
                                "rows_affected": converted,
                                "detail": f"to {canonical}"})

        # --- dates: parse multi-format, normalize to UTC-naive ------------
        if col.role == "temporal" and not pd.api.types.is_datetime64_any_dtype(
                work[name]):
            parsed = pd.to_datetime(work[name], errors="coerce",
                                    utc=True)
            if parsed.notna().any():
                parsed = parsed.dt.tz_localize(None) if parsed.dt.tz is not None \
                    else parsed
                changed = int(work[name].notna().sum()
                              - parsed.isna().sum())
                if changed > 0:
                    work[name] = parsed
                    ops.append({"op": "parse_datetime", "column": name,
                                "rows_affected": changed,
                                "detail": "normalized to UTC-naive"})

        # --- categories / free text: whitespace + case + Unicode ----------
        elif col.role in ("dimension", "categorical", "free_text"):
            if not (pd.api.types.is_string_dtype(work[name])
                    or pd.api.types.is_object_dtype(work[name])):
                continue
            mask = work[name].notna()
            if not mask.any():
                continue
            original = work[name].map(_nfc_normalize, na_action="ignore")
            collapsed = original.map(_collapse_ws, na_action="ignore")
            changed = int((original[mask] != collapsed[mask]).sum())
            out = work[name].copy()
            out.loc[mask] = collapsed[mask]
            work[name] = out
            if changed:
                ops.append({"op": "normalize_text", "column": name,
                            "rows_affected": changed,
                            "detail": "strip+collapse+whitespace NFC"})
            # casefold only when it genuinely merges categories
            folded = collapsed.str.casefold()
            if collapsed.nunique() > folded.nunique():
                out = work[name].copy()
                out.loc[mask] = folded[mask]
                work[name] = out
                ops.append({"op": "normalize_case", "column": name,
                            "rows_affected": int((collapsed[mask]
                                                  != folded[mask]).sum()),
                            "detail": "casefold unified categories"})
    return work, ops


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_contracts(run_dir: str | Path,
                   contracts: List[ColumnContract]) -> Path:
    run_dir = Path(run_dir)
    meta = run_dir / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    path = meta / "data_contracts.json"
    path.write_text(
        json.dumps([c.to_dict() for c in contracts], ensure_ascii=False,
                   indent=2),
        encoding="utf-8")
    return path


def load_contracts(run_dir: str | Path) -> List[ColumnContract]:
    path = Path(run_dir) / "metadata" / "data_contracts.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [ColumnContract.from_dict(d) for d in data]
    except (json.JSONDecodeError, OSError, TypeError):
        return []