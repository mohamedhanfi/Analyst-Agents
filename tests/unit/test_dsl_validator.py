"""Unit tests for shared/dsl_validator."""
from __future__ import annotations

import pytest

from shared.dsl_validator import (
    DslValidationError,
    WHITELIST,
    validate_operation,
    validate_plan,
)
from shared.schemas import AnalysisPlan, DslOperation, KpiCandidate

VALID_OPS = [
    {"function": "sum", "column": "revenue"},
    {"function": "mean", "column": "revenue"},
    {"function": "median", "column": "revenue"},
    {"function": "count", "column": "order_id"},
    {"function": "nunique", "column": "order_id"},
    {"function": "min", "column": "revenue"},
    {"function": "max", "column": "revenue"},
    {"function": "std", "column": "revenue"},
    {"function": "correlation", "column_a": "revenue",
     "column_b": "quantity", "method": "spearman"},
    {"function": "growth", "column": "revenue",
     "over_column": "order_date", "period": "YoY",
     "basis": "start_of_period", "as_percent": True},
    {"function": "ratio",
     "numerator": {"function": "sum", "column": "revenue"},
     "denominator": {"function": "count", "column": "order_id"}},
]


@pytest.mark.parametrize("op", VALID_OPS)
def test_valid_op_passes(op):
    assert validate_operation(op) == []


def test_optional_fields_are_valid():
    op = {"function": "sum", "column": "revenue",
          "group_by": ["category"], "filter": {"category": "Electronics"}}
    assert validate_operation(op) == []


def test_whitelist_matches_spec():
    assert WHITELIST == frozenset({
        "sum", "mean", "median", "count", "nunique", "min", "max", "std",
        "growth", "correlation", "ratio",
    })


@pytest.mark.parametrize("function", ["eval", "SUM", "sum(", "np.sum", "avg"])
def test_unknown_function_rejected(function):
    errors = validate_operation({"function": function, "column": "revenue"})
    assert errors and "unknown function" in errors[0]


def test_missing_function_rejected():
    errors = validate_operation({"column": "revenue"})
    assert errors and "missing 'function'" in errors[0]


@pytest.mark.parametrize("op,missing", [
    ({"function": "sum"}, "column"),
    ({"function": "correlation", "column_a": "revenue"}, "column_b"),
    ({"function": "growth", "column": "revenue"}, "over_column"),
    ({"function": "ratio",
      "numerator": {"function": "sum", "column": "revenue"}}, "denominator"),
])
def test_missing_required_field_rejected(op, missing):
    errors = validate_operation(op)
    assert any(missing in e for e in errors)


def test_forbidden_field_rejected():
    errors = validate_operation({"function": "sum", "column": "revenue",
                                 "column_a": "quantity"})
    assert any("unexpected field 'column_a'" in e for e in errors)


@pytest.mark.parametrize("field,value", [
    ("period", "QoQ"),
    ("period", "quarterly"),
    ("method", "kendall"),
    ("basis", "one_period"),
])
def test_bad_enum_values_rejected(field, value):
    op = {"function": "growth", "column": "revenue",
          "over_column": "order_date"}
    op[field] = value
    errors = validate_operation(op)
    assert any(f"'{field}' must be one of" in e for e in errors)


def test_as_percent_must_be_bool():
    errors = validate_operation({"function": "growth", "column": "revenue",
                                 "over_column": "order_date",
                                 "as_percent": "yes"})
    assert any("'as_percent' must be a boolean" in e for e in errors)


def test_group_by_must_be_list_of_strings():
    errors = validate_operation({"function": "sum", "column": "revenue",
                                 "group_by": "category"})
    assert any("'group_by' must be a list" in e for e in errors)


@pytest.mark.parametrize("flt", [{}, {1: "category"}, "category==Electronics"])
def test_filter_must_be_non_empty_string_keyed_dict(flt):
    errors = validate_operation({"function": "sum", "column": "revenue",
                                 "filter": flt})
    assert any("'filter' must be a non-empty dict" in e for e in errors)


def test_nested_ratio_recursion_validates_inner_ops():
    op = {"function": "ratio",
          "numerator": {"function": "sum", "column": "revenue"},
          "denominator": {"function": "mean", "column_b": "x"}}
    errors = validate_operation(op)
    assert any("ratio.denominator" in e for e in errors)
    assert any("missing required field 'column'" in e for e in errors)


def test_accepts_dsl_operation_instance():
    op = DslOperation(function="sum", column="revenue", group_by=["category"])
    assert validate_operation(op) == []


def test_dsl_operation_dump_with_nulls_passes():
    # model_dump() serializes unused fields as null — must be treated as absent
    op = DslOperation(function="sum", column="revenue")
    assert validate_operation(op.model_dump()) == []


def test_non_dict_input_raises():
    with pytest.raises(DslValidationError):
        validate_operation("sum(revenue)")  # type: ignore[arg-type]


def test_valid_plan_passes():
    plan = AnalysisPlan(candidate_kpis=[
        KpiCandidate(kpi_id="KPI-1", name="Total",
                     operation={"function": "sum", "column": "revenue"}),
        KpiCandidate(kpi_id="KPI-2", name="AOV",
                     operation={"function": "ratio",
                                "numerator": {"function": "sum",
                                              "column": "revenue"},
                                "denominator": {"function": "count",
                                                "column": "order_id"}}),
    ])
    assert validate_plan(plan) == []
    assert validate_plan(plan.model_dump()) == []


def test_bad_op_in_plan_flagged_with_kpi_id():
    plan = {"candidate_kpis": [
        {"kpi_id": "KPI-1", "name": "Bad",
         "operation": {"function": "nope", "column": "revenue"}},
    ]}
    errors = validate_plan(plan)
    assert any("KPI-1" in e for e in errors)
    assert any("unknown function 'nope'" in e for e in errors)


def test_plan_missing_candidate_kpis_rejected():
    assert validate_plan({}) != []


def test_candidate_missing_operation_rejected():
    errors = validate_plan({"candidate_kpis": [{"kpi_id": "KPI-1"}]})
    assert any("missing 'operation'" in e for e in errors)
