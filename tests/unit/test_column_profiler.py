"""Unit tests for shared/core/understanding + column_profiler_tool (§2.2)."""
from __future__ import annotations

import json

import pandas as pd

from shared.core.understanding import ColumnProfiler, ColumnFacts, infer_role
from shared.schemas import DataProfile
from shared.tools.understanding import column_profiler_tool


def make_profile(column_types, nunique, row_count=100,
                 missing=None) -> DataProfile:
    return DataProfile(
        file_name="sales.csv", file_hash="sha256:abc",
        row_count=row_count, column_count=len(column_types),
        columns=list(column_types),
        column_types=column_types, nunique=nunique,
        missing_values=missing or {},
        sample=[], validation_status="passed",
    )


def test_identifier_when_nunique_equals_row_count():
    p = make_profile({"order_id": "int64"}, {"order_id": 100}, row_count=100)
    assert ColumnProfiler().profile_columns(p)[0].suggested_role == "identifier"


def test_temporal_when_datetime_dtype():
    p = make_profile({"order_date": "datetime64[ns]"},
                     {"order_date": 60}, row_count=100)
    assert ColumnProfiler().profile_columns(p)[0].suggested_role == "temporal"


def test_numeric_high_cardinality_is_measure():
    p = make_profile({"revenue": "float64"}, {"revenue": 87}, row_count=100)
    assert ColumnProfiler().profile_columns(p)[0].suggested_role == "measure"


def test_numeric_low_cardinality_ambiguous():
    f = ColumnProfiler().profile_columns(
        make_profile({"rating": "float64"}, {"rating": 5}, row_count=100))[0]
    assert f.suggested_role == "measure"
    assert f.alternate_roles == ["categorical"]


def test_object_low_cardinality_is_dimension():
    p = make_profile({"city": "object"}, {"city": 12}, row_count=100)
    assert ColumnProfiler().profile_columns(p)[0].suggested_role == "dimension"


def test_object_high_cardinality_is_free_text():
    p = make_profile({"notes": "object"}, {"notes": 80}, row_count=100)
    assert ColumnProfiler().profile_columns(p)[0].suggested_role == "free_text"


def test_object_mid_cardinality_dimension_with_alternate():
    f = ColumnProfiler().profile_columns(
        make_profile({"note": "object"}, {"note": 30}, row_count=100))[0]
    assert f.suggested_role == "dimension"
    assert f.alternate_roles == ["free_text"]


def test_bool_is_dimension():
    p = make_profile({"is_active": "bool"}, {"is_active": 2}, row_count=100)
    assert ColumnProfiler().profile_columns(p)[0].suggested_role == "dimension"


def test_nullable_reflects_missing_values():
    p = make_profile({"city": "object"}, {"city": 11}, row_count=100,
                     missing={"city": 3})
    assert ColumnProfiler().profile_columns(p)[0].nullable is True
    p2 = make_profile({"city": "object"}, {"city": 11}, row_count=100)
    assert ColumnProfiler().profile_columns(p2)[0].nullable is False


def test_facts_include_dtype_and_nunique():
    f = ColumnProfiler().profile_columns(
        make_profile({"revenue": "float64"}, {"revenue": 87}, row_count=100))[0]
    assert f.dtype == "float64"
    assert f.nunique == 87
    assert f.name == "revenue"


def test_empty_dataset_does_not_crash():
    f = ColumnProfiler().profile_columns(
        make_profile({"a": "int64"}, {"a": 0}, row_count=0))[0]
    assert f.suggested_role in ("identifier", "dimension", "measure",
                                "temporal", "categorical", "free_text")


def test_column_profiler_tool_roundtrip():
    p = make_profile({"order_id": "int64", "revenue": "float64",
                      "city": "object"},
                     {"order_id": 100, "revenue": 87, "city": 12},
                     row_count=100)
    out = json.loads(column_profiler_tool.run(p.model_dump_json()))
    assert "columns" in out
    by_name = {c["name"]: c for c in out["columns"]}
    assert by_name["order_id"]["suggested_role"] == "identifier"
    assert by_name["revenue"]["suggested_role"] == "measure"
    assert by_name["city"]["suggested_role"] == "dimension"
    assert by_name["city"]["dtype"] == "object"


def test_infer_role_signature_stability():
    role, alternates = infer_role("float64", 5, 100)
    assert isinstance(role, str)
    assert isinstance(alternates, list)
    assert isinstance(ColumnFacts, type)
