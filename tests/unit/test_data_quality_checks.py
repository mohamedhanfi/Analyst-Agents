"""Unit tests for stage-3 checks (A2.3, part 1): schema, invalid values,
business rules, encoding."""
from __future__ import annotations

import pandas as pd

from shared.core.data_quality import (
    check_business_rules,
    check_invalid_values,
    check_schema,
)
from shared.schemas import BusinessContext, ColumnUnderstanding, \
    DataProfile, DatasetUnderstanding


def make_understanding(columns) -> DatasetUnderstanding:
    return DatasetUnderstanding(
        detected_domain="sales", domain_confidence=0.8,
        columns=[ColumnUnderstanding(name=name, role=role, dtype=dtype,
                                     nunique=0, nullable=False)
                 for name, role, dtype in columns],
        measures=[c[0] for c in columns if c[1] == "measure"],
        temporal_columns=[c[0] for c in columns if c[1] == "temporal"],
        dimensions=[c[0] for c in columns if c[1] == "dimension"],
        identifiers=[c[0] for c in columns if c[1] == "identifier"],
    )


# ---------------------------------------------------------------------------
# check_schema
# ---------------------------------------------------------------------------


def test_schema_missing_column_flagged():
    profile = DataProfile(
        file_name="x.csv", file_hash="sha256:a", row_count=10,
        column_count=1, columns=["ghost"], column_types={"ghost": "int64"},
        missing_values={}, nunique={"ghost": 5})
    understanding = make_understanding([("ghost", "measure", "int64")])
    assert check_schema(understanding, profile) == []

    profile2 = DataProfile(
        file_name="x.csv", file_hash="sha256:a", row_count=10,
        column_count=2, columns=["ghost", "extra"],
        column_types={"ghost": "int64", "extra": "int64"},
        missing_values={}, nunique={"ghost": 5, "extra": 5})
    issues = check_schema(understanding, profile2)
    details = {i.column: i.detail for i in issues}
    assert details["extra"] == "missing_column"


def test_schema_unknown_column_flagged():
    understanding = make_understanding(
        [("known", "measure", "int64"), ("mystery", "measure", "int64")])
    profile = DataProfile(
        file_name="x.csv", file_hash="sha256:a", row_count=10,
        column_count=1, columns=["known"], column_types={"known": "int64"},
        missing_values={}, nunique={"known": 5})
    issues = check_schema(understanding, profile)
    details = {i.column: i.detail for i in issues}
    assert details["mystery"] == "unknown_column"


def test_schema_type_mismatch_measure_and_temporal():
    understanding = make_understanding(
        [("revenue", "measure", "object"),
         ("date", "temporal", "object"),
         ("city", "dimension", "object")])
    profile = DataProfile(
        file_name="x.csv", file_hash="sha256:a", row_count=10,
        column_count=3, columns=["revenue", "date", "city"],
        column_types={"revenue": "object", "date": "object",
                      "city": "object"},
        missing_values={}, nunique={})
    issues = check_schema(understanding, profile)
    assert {i.column for i in issues} == {"revenue", "date"}


def test_schema_string_identifier_not_a_mismatch():
    understanding = make_understanding(
        [("order_id", "identifier", "string")])
    profile = DataProfile(
        file_name="x.csv", file_hash="sha256:a", row_count=10,
        column_count=1, columns=["order_id"],
        column_types={"order_id": "string"}, missing_values={},
        nunique={"order_id": 10})
    assert check_schema(understanding, profile) == []


# ---------------------------------------------------------------------------
# check_invalid_values
# ---------------------------------------------------------------------------


def test_negative_measure_flagged():
    understanding = make_understanding([("revenue", "measure", "float64")])
    df = pd.DataFrame({"revenue": [100.0, -50.0, 30.0]})
    issues = check_invalid_values(understanding, df)
    assert [(i.column, i.detail) for i in issues] == [
        ("revenue", "negative")]
    assert issues[0].severity == "high"


def test_percent_over_100_flagged():
    understanding = make_understanding(
        [("profit_margin_percent", "measure", "float64")])
    df = pd.DataFrame({"profit_margin_percent": [85.0, 120.0]})
    issues = check_invalid_values(understanding, df)
    assert [(i.column, i.detail) for i in issues] == [
        ("profit_margin_percent", "over_100_percent")]


def test_age_out_of_range_flagged():
    understanding = make_understanding([("age", "measure", "int64")])
    df = pd.DataFrame({"age": [30, 350, -2]})
    issues = check_invalid_values(understanding, df)
    details = {i.column: i.detail for i in issues}
    assert details["age"] == "out_of_range"
    assert all(i.severity == "high" for i in issues)


def test_impossible_dates_flagged():
    understanding = make_understanding([("date", "temporal", "object")])
    df = pd.DataFrame({"date": ["2024-01-01", "30-Feb-2024", "2200-01-01",
                                "2028-01-01"]})
    issues = check_invalid_values(understanding, df)
    details = {i.column: i.detail for i in issues}
    assert details["date"] == "impossible"
    assert issues[0].severity == "high"


def test_future_dates_flagged_medium():
    understanding = make_understanding([("date", "temporal", "object")])
    df = pd.DataFrame({"date": ["2024-01-01", "2028-06-01"]})
    issues = check_invalid_values(understanding, df)
    assert [(i.column, i.detail, i.severity) for i in issues] == [
        ("date", "future_dates", "medium")]


def test_clean_frame_no_invalid_issues():
    understanding = make_understanding(
        [("revenue", "measure", "float64"),
         ("date", "temporal", "datetime64[ns]"),
         ("qty", "measure", "int64")])
    df = pd.DataFrame({
        "revenue": [10.0, 20.5],
        "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "qty": [1, 2]})
    assert check_invalid_values(understanding, df) == []


# ---------------------------------------------------------------------------
# check_business_rules
# ---------------------------------------------------------------------------


def test_generic_mode_skips_rules():
    context = BusinessContext(file_name="x.csv", generic_mode=True)
    df = pd.DataFrame({"quantity": [0, 100]})
    assert check_business_rules(context, df) == []


def test_declared_range_enforced():
    context = BusinessContext(
        file_name="x.csv", generic_mode=False,
        answers={"q1": "quantity between 1 and 50"})
    df = pd.DataFrame({"quantity": [5, 0, 99]})
    issues = check_business_rules(context, df)
    assert [(i.column, i.category, i.severity) for i in issues] == [
        ("quantity", "business_rule", "high")]
    assert issues[0].detail.startswith("out_of_declared_range")


def test_goal_summary_range_parsed():
    context = BusinessContext(
        file_name="x.csv", generic_mode=False,
        goal_summary="revenue >= 0 is mandatory")
    df = pd.DataFrame({"revenue": [10.0, -5.0]})
    issues = check_business_rules(context, df)
    assert issues and issues[0].column == "revenue"


def test_case_duplicate_columns_encoding_issue():
    context = BusinessContext(file_name="x.csv", generic_mode=True)
    df = pd.DataFrame({"Revenue": [1.0], "revenue": [2.0]})
    issues = check_business_rules(context, df)
    assert [(i.column, i.category) for i in issues] == [
        ("revenue", "encoding")]
