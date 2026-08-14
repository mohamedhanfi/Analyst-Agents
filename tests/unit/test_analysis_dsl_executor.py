"""Unit tests for analysis/dsl_executor.py — whitelist ops over all rows."""
from __future__ import annotations

import pandas as pd
import pytest

from analysis.dsl_executor import (
    execute_kpi,
    execute_operation,
    execute_plan,
)
from analysis.evidence import EvidenceRegistry
from shared.schemas import AnalysisPlan, DslOperation, KpiCandidate

SALES = pd.DataFrame({
    "date": ["2024-01-15", "2024-02-15", "2024-03-15", "2023-01-15",
             "2023-02-15", "2023-03-15"],
    "product": ["A", "B", "A", "B", "A", "B"],
    "category": ["X", "Y", "X", "Y", "X", "Y"],
    "revenue": [100, 200, 300, 50, 80, 120],
    "quantity": [1, 2, 3, 4, 5, 6],
})


def op(**kwargs):
    return DslOperation(**kwargs)


def run(op_obj):
    return execute_operation(SALES, op_obj).value


# ---------------------------------------------------------------------------
# Aggregates (no group_by)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function,column,expected", [
    ("sum", "revenue", 850.0),
    ("mean", "revenue", 850.0 / 6),
    ("median", "revenue", 110.0),
    ("std", "revenue", float(SALES["revenue"].std())),
    ("min", "revenue", 50.0),
    ("max", "revenue", 300.0),
    ("count", "product", 6),
    ("nunique", "product", 2),
])
def test_aggregates_without_group(function, column, expected):
    assert run(op(function=function, column=column)) == pytest.approx(expected)


def test_min_on_string_column_returns_string():
    assert run(op(function="min", column="product")) == "A"


# ---------------------------------------------------------------------------
# group_by
# ---------------------------------------------------------------------------


def test_grouped_sum():
    assert run(op(function="sum", column="revenue", group_by=["category"])) \
        == {"X": 480.0, "Y": 370.0}


def test_grouped_multi_column():
    result = run(op(function="sum", column="revenue",
                    group_by=["category", "product"]))
    assert result == {"X | A": 480.0, "Y | B": 370.0}


def test_grouped_count():
    assert run(op(function="count", column="product",
                  group_by=["category"])) == {"X": 3, "Y": 3}


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_filter_equality():
    assert run(op(function="sum", column="revenue",
                  filter={"category": "X"})) == 480.0


def test_filter_membership():
    assert run(op(function="sum", column="revenue",
                  filter={"category": ["X", "Y"]})) == 850.0


def test_filter_and_group():
    result = run(op(function="sum", column="revenue", group_by=["category"],
                    filter={"product": "A"}))
    assert result == {"X": 480.0}


def test_filter_unknown_column_raises():
    with pytest.raises(ValueError):
        execute_operation(SALES, op(function="sum", column="revenue",
                                    filter={"nope": "X"}))


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------


def test_correlation_pearson_matches_pandas():
    value = run(op(function="correlation", column_a="revenue",
                   column_b="quantity", method="pearson"))
    assert value == pytest.approx(
        SALES["revenue"].corr(SALES["quantity"], method="pearson"))


def test_correlation_spearman_matches_pandas():
    value = run(op(function="correlation", column_a="revenue",
                   column_b="quantity", method="spearman"))
    assert value == pytest.approx(
        SALES["revenue"].corr(SALES["quantity"], method="spearman"))


def test_correlation_too_few_rows_raises():
    small = SALES.head(2)
    with pytest.raises(ValueError):
        execute_operation(small, op(function="correlation",
                                    column_a="revenue",
                                    column_b="quantity"))


# ---------------------------------------------------------------------------
# ratio
# ---------------------------------------------------------------------------


def test_ratio_nested_ops():
    value = run(op(function="ratio",
                   numerator={"function": "sum", "column": "revenue"},
                   denominator={"function": "count", "column": "product"}))
    assert value == pytest.approx(850.0 / 6)


# ---------------------------------------------------------------------------
# growth
# ---------------------------------------------------------------------------


def test_growth_mom_defaults_to_previous_period():
    # over_column alone -> month basis (MoM); basis defaults previous_period
    value = run(op(function="growth", column="revenue", over_column="date"))
    latest = (300 - 200) / 200  # 2024-03 vs 2024-02
    assert value == pytest.approx(latest)


def test_growth_mom_series():
    result = execute_operation(
        SALES, op(function="growth", column="revenue", over_column="date",
                  period="MoM"))
    values = {row["period"]: row["value"] for row in result.growth_series}
    assert values["2023-02"] == pytest.approx((80 - 50) / 50)
    assert values["2023-03"] == pytest.approx((120 - 80) / 80)
    assert values["2024-01"] == pytest.approx((100 - 120) / 120)
    assert values["2024-02"] == pytest.approx((200 - 100) / 100)
    assert values["2024-03"] == pytest.approx((300 - 200) / 200)


def test_growth_yoy():
    value = run(op(function="growth", column="revenue", over_column="date",
                   period="YoY"))
    assert value == pytest.approx((300 - 120) / 120)  # 2024-03 vs 2023-03


def test_growth_as_percent():
    value = run(op(function="growth", column="revenue", over_column="date",
                   period="YoY", as_percent=True))
    assert value == pytest.approx(150.0)


def test_growth_start_of_period_basis():
    value = run(op(function="growth", column="revenue", over_column="date",
                   period="MoM", basis="start_of_period"))
    assert value == pytest.approx((300 - 50) / 50)


def test_growth_grouped():
    value = run(op(function="growth", column="revenue", over_column="date",
                   period="MoM", group_by=["category"]))
    assert value["X"] == pytest.approx((300 - 100) / 100)
    assert value["Y"] == pytest.approx((200 - 120) / 120)


def test_growth_skips_nat_dates():
    df = SALES.copy()
    df.loc[0, "date"] = "not-a-date"
    value = execute_operation(
        df, op(function="growth", column="revenue", over_column="date",
               period="MoM"))
    assert value.value == pytest.approx((300 - 200) / 200)


# ---------------------------------------------------------------------------
# validation gate
# ---------------------------------------------------------------------------


def test_unknown_function_rejected():
    with pytest.raises(ValueError):
        execute_operation(SALES, {"function": "eval", "column": "revenue"})


def test_missing_required_field_rejected():
    with pytest.raises(ValueError):
        execute_operation(SALES, {"function": "sum"})


# ---------------------------------------------------------------------------
# execute_kpi / execute_plan
# ---------------------------------------------------------------------------


def test_execute_kpi_mints_evidence():
    registry = EvidenceRegistry(file_hash="sha256:abc")
    result = execute_kpi(SALES,
                         KpiCandidate(kpi_id="KPI-001", name="Total revenue",
                                      operation=op(function="sum",
                                                   column="revenue")),
                         registry)
    assert len(result) == 1
    assert result[0].value == 850.0
    assert result[0].evidence_id == "EV-001"
    assert result[0].computed_by == "pandas"
    assert registry.get("EV-001").source.aggregation == "sum"


def test_execute_kpi_grouped_produces_per_group_rows():
    registry = EvidenceRegistry()
    rows = execute_kpi(SALES,
                       KpiCandidate(kpi_id="KPI-002", name="Revenue by cat",
                                    operation=op(function="sum",
                                                 column="revenue",
                                                 group_by=["category"])),
                       registry)
    assert len(rows) == 2
    assert [r.value for r in rows] == [480.0, 370.0]
    assert rows[0].kpi_id == "KPI-002-001"
    assert rows[1].kpi_id == "KPI-002-002"
    assert all(r.evidence_id for r in rows)


def test_execute_kpi_failed_op_yields_none_value():
    registry = EvidenceRegistry()
    rows = execute_kpi(SALES,
                       KpiCandidate(kpi_id="KPI-003", name="Bad corr",
                                    operation=op(function="correlation",
                                                 column_a="revenue",
                                                 column_b="quantity",
                                                 method="pearson")),
                       registry)
    # correlation needs >=3 valid pairs; SALES has 6, so craft a tiny df
    tiny = SALES.head(2)
    rows = execute_kpi(tiny,
                       KpiCandidate(kpi_id="KPI-003", name="Bad corr",
                                    operation=op(function="correlation",
                                                 column_a="revenue",
                                                 column_b="quantity")),
                       registry)
    assert rows[0].value is None
    assert rows[0].evidence_id is not None


def test_execute_plan_runs_all_candidates():
    plan = AnalysisPlan(candidate_kpis=[
        KpiCandidate(kpi_id="KPI-001", name="Total revenue",
                     operation=op(function="sum", column="revenue")),
        KpiCandidate(kpi_id="KPI-002", name="Revenue growth",
                     operation=op(function="growth", column="revenue",
                                  over_column="date", period="YoY")),
        KpiCandidate(kpi_id="KPI-003", name="Avg per product",
                     operation=op(function="ratio",
                                  numerator={"function": "sum",
                                             "column": "revenue"},
                                  denominator={"function": "count",
                                               "column": "product"})),
    ])
    registry = EvidenceRegistry()
    results = execute_plan(SALES, plan, registry)
    assert len(results) == 3
    assert all(r.evidence_id for r in results)
    assert len(registry) == 3
