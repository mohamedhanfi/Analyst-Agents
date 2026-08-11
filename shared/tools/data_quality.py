"""CrewAI @tool wrappers — stage 3 Data Quality tools.

Deterministic gate tools (§2.3): they never reach an LLM inside a crew —
the stage is engine-only. The wrappers keep the same @tool contract so
they stay usable from crew/flows.py and show up in the run.jsonl audit
trail, while all logic lives in shared/core/data_quality.py.

Each tool consumes the same metadata JSON the pipeline already produced
(DataProfile, DatasetUnderstanding, BusinessContext) + the extracted CSV
path — never raw cell content beyond what the extracted file itself is.
"""
from __future__ import annotations

import json

from crewai.tools import tool

from shared.core.data_quality import (
    analyze_missingness,
    check_business_rules,
    check_invalid_values,
    check_referential_integrity,
    check_schema,
    detect_duplicates,
    deterministic_repair,
)
from shared.schemas import (
    BusinessContext,
    DataProfile,
    DatasetUnderstanding,
)


def _load(profile_json: str, understanding_json: str,
          context_json: str) -> tuple[DataProfile, DatasetUnderstanding,
                                      BusinessContext]:
    profile = DataProfile.model_validate(json.loads(profile_json))
    understanding = DatasetUnderstanding.model_validate(
        json.loads(understanding_json))
    context = BusinessContext.model_validate(json.loads(context_json))
    return profile, understanding, context


def _load_df(extracted_csv: str):
    import pandas as pd
    return pd.read_csv(extracted_csv, encoding="utf-8-sig")


@tool("schema_checker_tool")
def schema_checker_tool(profile_json: str, understanding_json: str) -> str:
    """Schema gate: columns present/absent + role/type consistency.

    profile_json: metadata/data_profile.json content.
    understanding_json: metadata/dataset_understanding.json content.
    Returns JSON issues list (severity/category/column/detail).
    """
    profile, understanding, _ = _load(profile_json, understanding_json, "{}")
    issues = check_schema(understanding, profile)
    return json.dumps([i.to_dict() for i in issues], ensure_ascii=False,
                      indent=2)


@tool("invalid_value_checker_tool")
def invalid_value_checker_tool(understanding_json: str,
                               extracted_csv: str) -> str:
    """Invalid/implausible values by column role (§2.3).

    understanding_json: metadata/dataset_understanding.json content.
    extracted_csv: path of the extracted dataset CSV.
    Returns JSON issues (negative/over_100_percent/out_of_range/
    impossible/future_dates).
    """
    _, understanding, _ = _load("{}", understanding_json, "{}")
    issues = check_invalid_values(understanding, _load_df(extracted_csv))
    return json.dumps([i.to_dict() for i in issues], ensure_ascii=False,
                      indent=2)


@tool("business_rules_checker_tool")
def business_rules_checker_tool(context_json: str,
                                extracted_csv: str) -> str:
    """Declared business rules from the context + encoding sanity.

    context_json: knowledge/business_context.json content.
    extracted_csv: path of the extracted dataset CSV.
    Generic Mode skips rule enforcement. Returns JSON issues.
    """
    context = BusinessContext.model_validate(json.loads(context_json))
    issues = check_business_rules(context, _load_df(extracted_csv))
    return json.dumps([i.to_dict() for i in issues], ensure_ascii=False,
                      indent=2)


@tool("missingness_analyzer_tool")
def missingness_analyzer_tool(understanding_json: str,
                              extracted_csv: str) -> str:
    """Missing rates + MCAR/MAR/MNAR assessment (§2.3).

    understanding_json: metadata/dataset_understanding.json content.
    extracted_csv: path of the extracted dataset CSV.
    Returns {"rate", "pattern", "assessment", "by_column"}.
    """
    _, understanding, _ = _load("{}", understanding_json, "{}")
    result = analyze_missingness(understanding, _load_df(extracted_csv))
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool("duplicate_detector_tool")
def duplicate_detector_tool(extracted_csv: str) -> str:
    """Exact duplicate rows: count + first 5 examples.

    extracted_csv: path of the extracted dataset CSV.
    Returns {"duplicates": n, "examples": [...]}.
    """
    count, examples = detect_duplicates(_load_df(extracted_csv))
    return json.dumps({"duplicates": count, "examples": examples},
                      ensure_ascii=False, indent=2)


@tool("referential_integrity_tool")
def referential_integrity_tool(understanding_json: str,
                               extracted_csv: str) -> str:
    """Identifier nulls + orphaned references (§2.3).

    understanding_json: metadata/dataset_understanding.json content.
    extracted_csv: path of the extracted dataset CSV.
    Returns JSON issues.
    """
    _, understanding, _ = _load("{}", understanding_json, "{}")
    issues = check_referential_integrity(understanding,
                                         _load_df(extracted_csv))
    return json.dumps([i.to_dict() for i in issues], ensure_ascii=False,
                      indent=2)


@tool("deterministic_repair_tool")
def deterministic_repair_tool(understanding_json: str,
                              extracted_csv: str) -> str:
    """Apply the §2.3 repair table in memory; report what changed.

    understanding_json: metadata/dataset_understanding.json content.
    extracted_csv: path of the extracted dataset CSV.
    Never invents data (no sign flips, no imputation). Returns the
    repair_log: {"repair_applied", "duplicates_removed",
    "impossible_rows_dropped", "type_casts", "coerced_to_null"}.
    """
    _, understanding, _ = _load("{}", understanding_json, "{}")
    _, repair_log = deterministic_repair(understanding,
                                         _load_df(extracted_csv))
    return json.dumps(repair_log, ensure_ascii=False, indent=2)
