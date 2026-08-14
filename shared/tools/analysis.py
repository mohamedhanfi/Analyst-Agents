"""CrewAI @tool wrappers — stage 5 Analysis tools (§2.5).

Deterministic Python only: every tool loads the dataset CSV on disk and runs
the whitelist DSL / statistical suite / chart planner, returning JSON. No tool
writes files — the Analysis agent owns persistence (kpis.json,
statistical_results.json, chart_metadata.json, evidence_registry.json).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
from crewai.tools import tool

from analysis.chart_planner import plan_charts
from analysis.dsl_executor import execute_plan
from analysis.evidence import EvidenceRegistry
from analysis.generic import run_statistical_suite
from shared.schemas import (AnalysisPlan, DatasetUnderstanding)


def _load_understanding(understanding_json: str) -> DatasetUnderstanding:
    return DatasetUnderstanding.model_validate(json.loads(understanding_json))


def _load_plan(plan_json: str) -> AnalysisPlan:
    return AnalysisPlan.model_validate(json.loads(plan_json))


def _load_df(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2,
                      default=str)


@tool("dsl_executor_tool")
def dsl_executor_tool(csv_path: str, understanding_json: str,
                      plan_json: str) -> str:
    """Execute every whitelist KPI in the analysis plan over ALL rows.

    csv_path: path of the dataset CSV. understanding_json:
    metadata/dataset_understanding.json content. plan_json:
    metadata/analysis_plan.json content (candidate_kpis only, no freeform
    formulas). Returns {"kpis": [...]} with one entry per computed value
    (grouped ops yield one per group); every value carries a minted
    evidence_id.
    """
    df = _load_df(csv_path)
    understanding = _load_understanding(understanding_json)
    plan = _load_plan(plan_json)
    registry = EvidenceRegistry(file_hash=__file__, sheet=None,
                                transformations=["cleaned_data"])
    results = execute_plan(df, plan, registry)
    return _json({"kpis": [r.model_dump() for r in results]})


@tool("statistical_suite_tool")
def statistical_suite_tool(csv_path: str, understanding_json: str,
                           tests_json: str = "") -> str:
    """Run the §2.5 statistical suite (descriptive/correlation/trend/anova).

    csv_path: path of the dataset CSV. understanding_json: dataset
    understanding content. tests_json (optional): JSON list of test categories
    from AnalysisPlan.statistical_tests; empty => deterministic default
    (descriptive + correlation + trend). Returns {"results": [...]} with one
    StatisticalResult per test, each evidence-minted.
    """
    df = _load_df(csv_path)
    understanding = _load_understanding(understanding_json)
    registry = EvidenceRegistry(file_hash=__file__, sheet=None,
                                transformations=["cleaned_data"])
    tests: List[str] = []
    if tests_json and tests_json.strip():
        try:
            tests = json.loads(tests_json)
        except json.JSONDecodeError:
            tests = []
    results = run_statistical_suite(df, understanding, registry, tests=tests)
    return _json({"results": [r.model_dump() for r in results]})


@tool("chart_planner_tool")
def chart_planner_tool(csv_path: str, understanding_json: str,
                       plan_json: str, limits_json: str = "") -> str:
    """Pick chart kinds for candidate visuals from the §2.5 rule table.

    csv_path: path of the dataset CSV. understanding_json: dataset
    understanding content. plan_json: analysis plan content. limits_json
    (optional): config limits subset ({"max_chart_count": 20}).
    Returns {"charts": [...], "charts_truncated": bool} — the shape is
    deterministic; the LLM may later re-rank, not redraw.
    """
    df = _load_df(csv_path)
    understanding = _load_understanding(understanding_json)
    plan = _load_plan(plan_json)
    registry = EvidenceRegistry(file_hash=__file__, sheet=None,
                                transformations=["cleaned_data"])
    limits: Dict[str, Any] = {}
    if limits_json and limits_json.strip():
        try:
            limits = json.loads(limits_json)
        except json.JSONDecodeError:
            limits = {}
    max_chart_count = limits.get("max_chart_count", 20)
    max_chart_count = 20 if max_chart_count is None else int(max_chart_count)
    thin_threshold = limits.get("thin_threshold", 10)
    thin_threshold = 10 if thin_threshold is None else int(thin_threshold)
    charts, truncated = plan_charts(df, plan, understanding, registry,
                                    max_chart_count=max_chart_count,
                                    thin_threshold=thin_threshold)
    return _json({"charts": [c.model_dump() for c in charts],
                  "charts_truncated": truncated})
