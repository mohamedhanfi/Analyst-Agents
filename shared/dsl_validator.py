"""DSL whitelist + semantic validator (§2.5) — dual use.

Shared gate for BOTH consumers of the KPI DSL:
- Understanding (Day 3) builds the analysis plan -> validates the LLM's
  raw JSON before it is accepted.
- Analyst (Day 7) executes the plan -> validates each parsed op before
  the executor runs it.

What this module enforces (what Pydantic alone cannot):
- function must be in the whitelist (no freeform formulas, ever)
- every required field per function is present (e.g. ratio needs a
  denominator; growth needs over_column)
- no forbidden fields per function (e.g. a sum carrying column_a is a bug)
- enum values + types for raw dicts that never touch the Pydantic model
- ratio numerator/denominator are themselves valid nested operations

Golden rule: the LLM decides WHAT to run from this menu; Python computes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Union

from pydantic import BaseModel

from shared.schemas import AnalysisPlan, DslOperation

WHITELIST = frozenset({
    "sum", "mean", "median", "count", "nunique", "min", "max", "std",
    "growth", "correlation", "ratio",
})

# Per-function required fields (presence + non-None)
REQUIRED_FIELDS: Dict[str, set] = {
    "sum": {"column"}, "mean": {"column"}, "median": {"column"},
    "count": {"column"}, "nunique": {"column"}, "min": {"column"},
    "max": {"column"}, "std": {"column"},
    "correlation": {"column_a", "column_b"},
    "growth": {"column", "over_column"},
    "ratio": {"numerator", "denominator"},
}

# Per-function allowed fields (anything else is rejected — strict whitelist)
ALLOWED_FIELDS: Dict[str, set] = {
    "sum": {"column", "group_by", "filter"},
    "mean": {"column", "group_by", "filter"},
    "median": {"column", "group_by", "filter"},
    "count": {"column", "group_by", "filter"},
    "nunique": {"column", "group_by", "filter"},
    "min": {"column", "group_by", "filter"},
    "max": {"column", "group_by", "filter"},
    "std": {"column", "group_by", "filter"},
    "correlation": {"column_a", "column_b", "method", "filter"},
    "growth": {"column", "over_column", "period", "basis",
               "as_percent", "group_by", "filter"},
    "ratio": {"numerator", "denominator"},
}

# Fields that must hold a string column name
STRING_FIELDS = {"column", "column_a", "column_b", "over_column"}

# Enum-valued fields (raw dicts bypass Pydantic, so we enforce membership)
ENUM_FIELDS: Dict[str, set] = {
    "method": {"pearson", "spearman"},
    "period": {"YoY", "MoM", "WoW"},
    "basis": {"previous_period", "start_of_period"},
}


class DslValidationError(Exception):
    """Raised when a value is not a valid DSL operation/plan input."""


OpInput = Union[dict, DslOperation]


def validate_operation(op: OpInput) -> List[str]:
    """Validate one DSL operation; return error strings (empty = valid).

    Accepts a raw dict (LLM JSON) or a parsed ``DslOperation``. ``None``
    values are treated as absent, so a Pydantic ``model_dump()`` of a valid
    op (which serializes unused fields as null) passes cleanly.
    """
    data = _to_dict(op)
    if "function" not in data:
        return ["operation is missing 'function'"]

    function = data["function"]
    if not isinstance(function, str) or function not in WHITELIST:
        return [f"unknown function '{function}' "
                f"(whitelist: {', '.join(sorted(WHITELIST))})"]

    errors: List[str] = []

    for field in sorted(set(data) - ALLOWED_FIELDS[function] - {"function"}):
        errors.append(f"{function}: unexpected field '{field}'")

    for field in sorted(REQUIRED_FIELDS[function]):
        if field not in data:
            errors.append(f"{function}: missing required field '{field}'")

    errors.extend(_check_types(function, data))

    if function == "ratio":
        for field in ("numerator", "denominator"):
            if field in data:
                nested = data[field]
                if not isinstance(nested, (dict, BaseModel)):
                    errors.append(
                        f"{function}.{field}: must be a DSL operation object")
                    continue
                nested_errors = validate_operation(nested)
                errors.extend(f"{function}.{field}: {e}" for e in nested_errors)

    return errors


def validate_plan(plan: Union[AnalysisPlan, dict]) -> List[str]:
    """Validate a full analysis plan; return error strings (empty = valid).

    Accepts an ``AnalysisPlan`` model or a raw dict with ``candidate_kpis``.
    """
    data = _to_dict(plan)
    if not isinstance(data, dict):
        return ["plan must be an object with 'candidate_kpis'"]

    candidates = data.get("candidate_kpis")
    if candidates is None:
        return ["plan is missing 'candidate_kpis'"]
    if not isinstance(candidates, list):
        return ["'candidate_kpis' must be a list"]

    errors: List[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            errors.append(f"candidate #{index}: must be an object")
            continue
        operation = candidate.get("operation")
        if operation is None:
            errors.append(f"candidate #{index}: missing 'operation'")
            continue
        label = candidate.get("kpi_id") or f"#{index}"
        errors.extend(f"{label}: {e}"
                      for e in validate_operation(operation))
    return errors


# ------------------------------------------------------------- internals

def _to_dict(value: Any) -> Any:
    """Normalize a BaseModel/raw dict, dropping None values (None = absent)."""
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if v is not None}
    raise DslValidationError(
        f"expected a dict or DslOperation, got {type(value).__name__}")


def _check_types(function: str, data: dict) -> List[str]:
    errors: List[str] = []

    for field in STRING_FIELDS:
        if field in data and (not isinstance(data[field], str)
                              or not data[field].strip()):
            errors.append(f"{function}: '{field}' must be a non-empty string")

    for field, allowed in ENUM_FIELDS.items():
        if field in data and data[field] not in allowed:
            errors.append(
                f"{function}: '{field}' must be one of {sorted(allowed)}")

    if "as_percent" in data and not isinstance(data["as_percent"], bool):
        errors.append(f"{function}: 'as_percent' must be a boolean")

    if "group_by" in data:
        group_by = data["group_by"]
        if (not isinstance(group_by, list)
                or not all(isinstance(c, str) and c.strip()
                           for c in group_by)):
            errors.append(
                f"{function}: 'group_by' must be a list of column names")

    if "filter" in data:
        flt = data["filter"]
        if (not isinstance(flt, dict) or not flt
                or not all(isinstance(k, str) and k for k in flt)):
            errors.append(
                f"{function}: 'filter' must be a non-empty dict "
                f"with string keys")

    return errors
