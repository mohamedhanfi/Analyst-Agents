"""CrewAI @tool wrappers — stage 4 Cleaning tools (§2.4).

The LLM only *proposes* the strategy (cleaning_strategy_tool); every
execution tool is deterministic Python that loads the CSV on disk, applies a
single operation in-memory, and returns the JSON result/log. No tool writes
files — Cleaning owns persistence (data/processed/cleaned_data.csv).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
from crewai.tools import tool

from shared.core.cleaning import (
    build_strategy,
    execute_strategy,
    normalize_strategy,
)
from shared.schemas import (
    BusinessContext,
    DataProfile,
    DataQualityReport,
    DatasetUnderstanding,
)

_FILL_METHODS = frozenset({"median_fill", "median_fill_flag", "mode_fill",
                           "unknown_fill"})
_FLAG_TYPES = {"missing": "flag_and_preserve", "preserve": "flag_and_preserve",
               "keep": "keep_flag"}


def _load_understanding(understanding_json: str) -> DatasetUnderstanding:
    return DatasetUnderstanding.model_validate(json.loads(understanding_json))


def _load_dq_report(dq_report_json: str) -> DataQualityReport:
    return DataQualityReport.model_validate(json.loads(dq_report_json))


def _load_df(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def _single_strategy(understanding: DatasetUnderstanding, column: str,
                     action: str) -> Dict[str, Any]:
    role = next((c.role for c in understanding.columns
                 if c.name == column), "dimension")
    return {"columns": [{"column": column, "role": role, "action": action,
                         "detail": "tool_call"}],
            "deduplicate": False, "outliers": {}}


@tool("cleaning_strategy_tool")
def cleaning_strategy_tool(understanding_json: str, dq_report_json: str,
                           proposed_strategy_json: str = "") -> str:
    """Build the §2.4 cleaning strategy for a run.

    understanding_json: metadata/dataset_understanding.json content.
    dq_report_json: metadata/data_quality_report.json content.
    proposed_strategy_json (optional): the LLM's raw strategy
    {"columns": [{"column", "action"}], "deduplicate": bool, "outliers":
    {column: "flag"|"drop"}} — validated/normalized (unknown columns/actions
    dropped with reasons); when omitted, the deterministic §2.4 default is
    returned.
    Returns {"strategy": {...}, "errors": [...]}.
    """
    understanding = _load_understanding(understanding_json)
    report = _load_dq_report(dq_report_json)
    if proposed_strategy_json and proposed_strategy_json.strip():
        strategy, errors = normalize_strategy(proposed_strategy_json,
                                              understanding, report)
    else:
        strategy, errors = build_strategy(understanding, report), []
    return json.dumps({"strategy": strategy, "errors": errors},
                      ensure_ascii=False, indent=2)


@tool("fillna_tool")
def fillna_tool(cleaned_csv: str, understanding_json: str, column: str,
                method: str) -> str:
    """Fill one column (median/mode/"Unknown") per the strategy.

    cleaned_csv: path of the dataset CSV. understanding_json: dataset
    understanding content. column: target. method: median_fill |
    median_fill_flag | mode_fill | unknown_fill.
    Returns JSON {"column", "method", "ops": [...]}.
    """
    if method not in _FILL_METHODS:
        return json.dumps({"column": column, "method": method,
                           "error": f"unknown fill method '{method}'"},
                          ensure_ascii=False, indent=2)
    understanding = _load_understanding(understanding_json)
    df = _load_df(cleaned_csv)
    _, log = execute_strategy(df, _single_strategy(understanding, column,
                                                   method), understanding)
    ops = [op for op in log if op.get("column") == column]
    return json.dumps({"column": column, "method": method, "ops": ops},
                      ensure_ascii=False, indent=2)


@tool("flag_column_tool")
def flag_column_tool(cleaned_csv: str, understanding_json: str,
                     column: str, flag_type: str = "missing") -> str:
    """Create the boolean `{column}_missing_flag` (flag_and_preserve).

    cleaned_csv: path of the dataset CSV. understanding_json: dataset
    understanding content. column: target. flag_type: "missing" (default) |
    "preserve" (both keep the NaNs) | "keep" (sparse >70% keep).
    Returns JSON {"flag", "rows_flagged", "ops": [...]}.
    """
    action = _FLAG_TYPES.get(flag_type)
    if action is None:
        return json.dumps({"column": column, "flag_type": flag_type,
                           "error": "flag_type must be missing|preserve|keep"},
                          ensure_ascii=False, indent=2)
    understanding = _load_understanding(understanding_json)
    df = _load_df(cleaned_csv)
    cleaned, log = execute_strategy(
        df, _single_strategy(understanding, column, action), understanding)
    flag = f"{column}_missing_flag"
    rows_flagged = int(cleaned[flag].sum()) if flag in cleaned.columns else 0
    return json.dumps({"flag": flag, "rows_flagged": rows_flagged,
                       "ops": [op for op in log
                               if op.get("column") == column]},
                      ensure_ascii=False, indent=2)


@tool("type_caster_tool")
def type_caster_tool(cleaned_csv: str, understanding_json: str,
                     column: str) -> str:
    """Cast one column by role (measure->numeric, temporal->datetime).

    cleaned_csv: path of the dataset CSV. understanding_json: dataset
    understanding content. column: target.
    Returns JSON {"column", "casts": [{"detail": "object->float64"}]}.
    """
    understanding = _load_understanding(understanding_json)
    df = _load_df(cleaned_csv)
    _, log = execute_strategy(df, _single_strategy(understanding, column,
                                                   "keep"), understanding)
    casts = [op for op in log
             if op.get("op") == "type_cast" and op.get("column") == column]
    return json.dumps({"column": column, "casts": casts},
                      ensure_ascii=False, indent=2)


@tool("dedup_tool")
def dedup_tool(cleaned_csv: str) -> str:
    """Count exact duplicate rows on the dataset CSV (read-only preview).
    Returns JSON {"rows_before", "rows_after", "duplicates_removed"}.
    """
    df = _load_df(cleaned_csv)
    before = len(df)
    after = len(df.drop_duplicates())
    return json.dumps({"rows_before": before, "rows_after": after,
                       "duplicates_removed": before - after},
                      ensure_ascii=False, indent=2)


@tool("iqr_outlier_tool")
def iqr_outlier_tool(cleaned_csv: str, understanding_json: str,
                     column: str, mode: str = "flag") -> str:
    """IQR outlier handling on one numeric column.

    cleaned_csv: path of the dataset CSV. understanding_json: dataset
    understanding content. column: target. mode: "flag" (creates
    `{column}_outlier_flag`) | "drop".
    Returns JSON {"column", "mode", "outliers": <count>, "ops": [...]}.
    """
    if mode not in ("flag", "drop"):
        return json.dumps({"column": column, "mode": mode,
                           "error": "mode must be 'flag' or 'drop'"},
                          ensure_ascii=False, indent=2)
    understanding = _load_understanding(understanding_json)
    df = _load_df(cleaned_csv)
    _, log = execute_strategy(
        df, {"columns": [], "deduplicate": False, "outliers": {column: mode}},
        understanding)
    ops = [op for op in log if op.get("column") == column
           and str(op.get("op", "")).startswith("iqr_outlier")]
    count = ops[-1].get("rows_affected", 0) if ops else 0
    return json.dumps({"column": column, "mode": mode, "outliers": count,
                       "ops": ops}, ensure_ascii=False, indent=2)


@tool("dq_recheck_tool")
def dq_recheck_tool(cleaned_csv: str, understanding_json: str,
                    profile_json: str, context_json: str = "",
                    limits_json: str = "") -> str:
    """Re-run Agent 3's checks on the cleaned output (§2.4 recheck).

    cleaned_csv: path of the cleaned dataset CSV. understanding_json:
    dataset understanding content. profile_json: data_profile.json content.
    context_json (optional): business_context.json content (Generic Mode when
    omitted). limits_json (optional): config limits subset.
    Returns {"status": "passed"|"needs_repair", "report": {...}}.
    """
    from shared.core.data_quality import assemble_report

    understanding = _load_understanding(understanding_json)
    profile = DataProfile.model_validate(json.loads(profile_json))
    if context_json and context_json.strip():
        context = BusinessContext.model_validate(json.loads(context_json))
    else:
        context = BusinessContext(file_name="", generic_mode=True)
    limits: Dict[str, Any] = {}
    if limits_json and limits_json.strip():
        try:
            limits = json.loads(limits_json)
        except json.JSONDecodeError:
            limits = {}
    df = _load_df(cleaned_csv)
    df, _ = execute_strategy(
        df, {"columns": [], "deduplicate": False, "outliers": {}},
        understanding)
    report, _ = assemble_report(understanding=understanding, profile=profile,
                                df=df, context=context, limits=limits)
    return json.dumps({"status": report.status,
                       "report": report.model_dump()},
                      ensure_ascii=False, indent=2)
