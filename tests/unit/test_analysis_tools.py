"""Unit tests for the stage-5a @tool wrappers (§2.5): DSL execution, the
statistical suite, and the chart planner — all deterministic Python."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from shared.tools.analysis import (
    chart_planner_tool,
    dsl_executor_tool,
    statistical_suite_tool,
)
from tests.unit.conftest import SALES

UNDERSTANDING = json.dumps({
    "detected_domain": "sales", "domain_confidence": 0.9,
    "entities": [], "temporal_columns": ["date"],
    "dimensions": ["product", "category"], "measures": ["revenue", "quantity"],
    "identifiers": [],
    "columns": [
        {"name": "date", "role": "temporal", "dtype": "str",
         "nunique": 6, "nullable": False},
        {"name": "product", "role": "dimension", "dtype": "str",
         "nunique": 2, "nullable": False},
        {"name": "category", "role": "dimension", "dtype": "str",
         "nunique": 2, "nullable": False},
        {"name": "revenue", "role": "measure", "dtype": "int64",
         "nunique": 6, "nullable": False},
        {"name": "quantity", "role": "measure", "dtype": "int64",
         "nunique": 6, "nullable": False},
    ],
    "has_temporal_data": True, "limitations": [],
})

PLAN = json.dumps({
    "candidate_kpis": [
        {"kpi_id": "KPI-001", "name": "total revenue",
         "operation": {"function": "sum", "column": "revenue"}},
        {"kpi_id": "KPI-002", "name": "revenue growth",
         "operation": {"function": "growth", "column": "revenue",
                       "over_column": "date", "period": "YoY"}},
        {"kpi_id": "KPI-003", "name": "revenue vs quantity",
         "operation": {"function": "correlation", "column_a": "revenue",
                       "column_b": "quantity"}},
        {"kpi_id": "KPI-004", "name": "revenue by product",
         "operation": {"function": "sum", "column": "revenue",
                       "group_by": ["product"]}},
    ],
    "statistical_tests": ["descriptive", "correlation", "trend", "anova"],
    "has_temporal_data": True, "limitations": [],
})


@pytest.fixture
def sales_csv(tmp_path):
    path = tmp_path / "sales.csv"
    SALES.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def test_dsl_executor_tool_runs_all_kpis(sales_csv):
    payload = json.loads(dsl_executor_tool.run(sales_csv, UNDERSTANDING, PLAN))
    kpis = payload["kpis"]
    assert len(kpis) == 1 + 1 + 1 + 2  # sum + growth(latest) + corr + grouped(2)
    assert all(k["computed_by"] == "pandas" for k in kpis)
    assert all(k["evidence_id"] for k in kpis)


def test_dsl_executor_tool_kpi_values(sales_csv):
    kpis = json.loads(dsl_executor_tool.run(sales_csv, UNDERSTANDING, PLAN))["kpis"]
    total = next(k for k in kpis if k["kpi_id"] == "KPI-001")
    assert total["value"] == 850.0
    grouped = {k["kpi_id"]: k["value"] for k in kpis
               if k["kpi_id"].startswith("KPI-004")}
    assert grouped == {"KPI-004-001": 480.0, "KPI-004-002": 370.0}


def test_statistical_suite_tool_defaults(sales_csv):
    payload = json.loads(statistical_suite_tool.run(sales_csv, UNDERSTANDING))
    categories = {r["category"] for r in payload["results"]}
    assert categories == {"descriptive", "correlation", "trend"}
    assert all(r["evidence_id"] for r in payload["results"])


def test_statistical_suite_tool_respects_tests(sales_csv):
    payload = json.loads(statistical_suite_tool.run(
        sales_csv, UNDERSTANDING, json.dumps(["descriptive"])))
    assert {r["category"] for r in payload["results"]} == {"descriptive"}


def test_statistical_suite_tool_anova_maps_comparison(sales_csv):
    payload = json.loads(statistical_suite_tool.run(
        sales_csv, UNDERSTANDING, json.dumps(["anova"])))
    assert {r["category"] for r in payload["results"]} == {"comparison"}
    assert "t_test" in {r["test_name"] for r in payload["results"]}


def test_statistical_suite_tool_bad_tests_json_falls_back(sales_csv):
    payload = json.loads(statistical_suite_tool.run(sales_csv, UNDERSTANDING,
                                                "not json"))
    categories = {r["category"] for r in payload["results"]}
    assert categories == {"descriptive", "correlation", "trend"}


def test_chart_planner_tool_produces_charts(sales_csv):
    limits = json.dumps({"thin_threshold": 0})  # keep shapes on 6-row fixture
    payload = json.loads(chart_planner_tool.run(sales_csv, UNDERSTANDING, PLAN,
                                                limits))
    assert payload["charts_truncated"] is False
    charts = payload["charts"]
    assert all(c["computed_by"] == "pandas" for c in charts)
    assert all(c["evidence_id"] for c in charts)
    kinds = {c["kind"] for c in charts}
    assert "line" in kinds and "scatter" in kinds  # growth + correlation


def test_chart_planner_tool_respects_max_chart_count(sales_csv):
    payload = json.loads(chart_planner_tool.run(
        sales_csv, UNDERSTANDING, PLAN,
        json.dumps({"max_chart_count": 1})))
    assert payload["charts_truncated"] is True
    assert len(payload["charts"]) == 1


def test_chart_planner_tool_bad_limits_json_uses_default(sales_csv):
    payload = json.loads(chart_planner_tool.run(sales_csv, UNDERSTANDING, PLAN,
                                            "not json"))
    assert payload["charts_truncated"] is False
