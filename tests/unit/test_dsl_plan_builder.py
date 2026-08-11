"""Unit tests for build_analysis_plan + dsl_plan_builder_tool (§2.2)."""
from __future__ import annotations

import json

from shared.core.understanding import build_analysis_plan
from shared.dsl_validator import validate_plan
from shared.schemas import AnalysisPlan
from shared.tools.understanding import dsl_plan_builder_tool


VALID_KPIS = [
    {"kpi_id": "KPI-001", "name": "Total Revenue",
     "operation": {"function": "sum", "column": "revenue"}},
    {"kpi_id": "KPI-002", "name": "AOV",
     "operation": {"function": "ratio",
                   "numerator": {"function": "sum", "column": "revenue"},
                   "denominator": {"function": "count", "column": "order_id"}}},
]


def test_valid_plan_passes_clean():
    plan, errors = build_analysis_plan(
        {"candidate_kpis": VALID_KPIS, "statistical_tests": ["descriptive"]})
    assert errors == []
    assert len(plan.candidate_kpis) == 2
    assert validate_plan(plan) == []


def test_unknown_function_dropped_with_reason():
    raw = {"candidate_kpis": VALID_KPIS + [
        {"kpi_id": "KPI-003", "name": "Evil",
         "operation": {"function": "evil", "column": "x"}}]}
    plan, errors = build_analysis_plan(raw)
    assert len(plan.candidate_kpis) == 2
    assert any("KPI-003" in e and "unknown function" in e for e in errors)


def test_missing_column_dropped():
    raw = {"candidate_kpis": [{"kpi_id": "KPI-X",
                               "operation": {"function": "sum"}}]}
    plan, errors = build_analysis_plan(raw)
    assert plan.candidate_kpis == []
    assert any("KPI-X" in e and "missing required field 'column'" in e
               for e in errors)


def test_bad_nested_ratio_dropped():
    raw = {"candidate_kpis": [{"kpi_id": "KPI-R",
                               "operation": {"function": "ratio",
                                             "numerator": "sum",
                                             "denominator": "count"}}]}
    plan, errors = build_analysis_plan(raw)
    assert plan.candidate_kpis == []
    assert any("ratio.numerator" in e for e in errors)


def test_statistical_tests_whitelist():
    plan, errors = build_analysis_plan(
        {"candidate_kpis": [], "statistical_tests":
         ["descriptive", "trend", "magic"]})
    assert plan.statistical_tests == ["descriptive", "trend"]
    assert any("unknown statistical test 'magic'" in e for e in errors)


def test_missing_operation_reported():
    plan, errors = build_analysis_plan(
        {"candidate_kpis": [{"kpi_id": "KPI-9", "name": "X"}]})
    assert plan.candidate_kpis == []
    assert any("missing 'operation'" in e for e in errors)


def test_kpi_id_and_name_auto_generated():
    plan, errors = build_analysis_plan(
        {"candidate_kpis": [{"operation": {"function": "sum",
                                           "column": "revenue"}}]})
    assert len(plan.candidate_kpis) == 1
    assert plan.candidate_kpis[0].kpi_id == "KPI-001"
    assert plan.candidate_kpis[0].name == "Sum of revenue"


def test_accepts_analysisplan_model():
    plan, errors = build_analysis_plan(AnalysisPlan(
        candidate_kpis=[{"kpi_id": "K", "name": "N",
                         "operation": {"function": "count",
                                       "column": "a"}}]))
    assert errors == []
    assert plan.candidate_kpis[0].kpi_id == "K"


def test_accepts_json_string():
    plan, errors = build_analysis_plan(json.dumps(
        {"candidate_kpis": VALID_KPIS}))
    assert errors == []
    assert len(plan.candidate_kpis) == 2


def test_garbage_input_never_raises():
    plan, errors = build_analysis_plan(["not", "a", "plan"])
    assert plan.candidate_kpis == []
    assert errors
    plan2, errors2 = build_analysis_plan("{not json")
    assert errors2 == ["plan must be valid JSON"]
    plan3, errors3 = build_analysis_plan(None)
    assert plan3.candidate_kpis == []
    assert errors3


def test_plan_builder_tool_roundtrip():
    out = json.loads(dsl_plan_builder_tool.run(json.dumps(
        {"candidate_kpis": VALID_KPIS + [
            {"kpi_id": "KPI-BAD",
             "operation": {"function": "evil", "column": "x"}}]})))
    assert len(out["plan"]["candidate_kpis"]) == 2
    assert any("KPI-BAD" in e and "unknown function" in e for e in out["errors"])
    out2 = json.loads(dsl_plan_builder_tool.run(json.dumps(
        {"candidate_kpis": [{"kpi_id": "KPI-BAD",
                             "operation": {"function": "evil"}}]})))
    assert len(out2["plan"]["candidate_kpis"]) == 0
    assert any("unknown function" in e for e in out2["errors"])
