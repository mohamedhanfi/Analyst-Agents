"""Unit tests for the A2.3 deterministic repair: the golden table — cast by
role, drop exact duplicates and impossible rows, and NEVER invent data
(no sign flips, no imputation, no removal of negatives)."""
from __future__ import annotations

import pandas as pd

from shared.core.data_quality import deterministic_repair
from shared.schemas import ColumnUnderstanding, DatasetUnderstanding


def understanding_for(columns) -> DatasetUnderstanding:
    return DatasetUnderstanding(
        detected_domain="sales", domain_confidence=0.8,
        columns=[ColumnUnderstanding(name=name, role=role, dtype=dtype,
                                     nunique=0, nullable=False)
                 for name, role, dtype in columns],
        measures=[c[0] for c in columns if c[1] == "measure"],
        temporal_columns=[c[0] for c in columns if c[1] == "temporal"],
        identifiers=[c[0] for c in columns if c[1] == "identifier"],
    )


def test_negative_values_never_sign_flipped():
    understanding = understanding_for(
        [("revenue", "measure", "float64")])
    df = pd.DataFrame({"revenue": [-5.0, 10.0, -0.5]})
    repaired, repair_log = deterministic_repair(understanding, df)
    assert repaired["revenue"].tolist() == [-5.0, 10.0, -0.5]
    assert not repair_log["repair_applied"]


def test_measure_object_cast_to_numeric():
    understanding = understanding_for(
        [("revenue", "measure", "object")])
    df = pd.DataFrame({"revenue": ["123.5", "-45", "abc", "200"]})
    repaired, repair_log = deterministic_repair(understanding, df)
    assert repaired["revenue"].iloc[0] == 123.5
    assert repaired["revenue"].iloc[1] == -45.0
    assert pd.isna(repaired["revenue"].iloc[2])
    assert repair_log["coerced_to_null"] == {"revenue": 1}
    assert "revenue" in repair_log["type_casts"]
    assert repair_log["repair_applied"]


def test_temporal_object_cast_to_datetime():
    understanding = understanding_for(
        [("date", "temporal", "object")])
    df = pd.DataFrame({"date": ["2024-01-01", "2024-02-01"]})
    repaired, repair_log = deterministic_repair(understanding, df)
    assert repaired["date"].dtype.kind == "M"
    assert repair_log["type_casts"]["date"].startswith("object->")


def test_string_identifier_kept_as_string():
    understanding = understanding_for(
        [("order_id", "identifier", "object")])
    df = pd.DataFrame({"order_id": ["A-1", "B-2", "A-1"]})
    repaired, repair_log = deterministic_repair(understanding, df)
    assert repaired["order_id"].tolist() == ["A-1", "B-2"]
    assert repair_log["type_casts"] == {}
    assert repair_log["duplicates_removed"] == 1


def test_exact_duplicates_dropped():
    understanding = understanding_for(
        [("a", "measure", "int64"), ("b", "dimension", "object")])
    df = pd.DataFrame({
        "a": [1, 2, 1, 3, 1],
        "b": ["x", "y", "x", "z", "x"],
    })
    repaired, repair_log = deterministic_repair(understanding, df)
    assert len(repaired) == 3
    assert repair_log["duplicates_removed"] == 2


def test_impossible_dates_dropped_and_logged():
    understanding = understanding_for(
        [("date", "temporal", "object"),
         ("v", "measure", "int64")])
    df = pd.DataFrame({
        "date": ["2024-01-01", "2200-01-01", "30-Feb-2024", "2024-02-02"],
        "v": [1, 2, 3, 4],
    })
    repaired, repair_log = deterministic_repair(understanding, df)
    assert repaired["date"].tolist() == [
        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-02")]
    assert repair_log["impossible_rows_dropped"]["date"] == [1, 2]


def test_missing_values_not_imputed():
    understanding = understanding_for(
        [("revenue", "measure", "float64"),
         ("date", "temporal", "datetime64[ns]")])
    df = pd.DataFrame({
        "revenue": [1.0, None, 3.0],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    })
    repaired, repair_log = deterministic_repair(understanding, df)
    assert repaired["revenue"].isna().sum() == 1
    assert not repair_log["repair_applied"]


def test_repair_log_structure():
    understanding = understanding_for(
        [("date", "temporal", "object"),
         ("revenue", "measure", "object")])
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01"],
        "revenue": ["1.0", "1.0"],
    })
    repaired, repair_log = deterministic_repair(understanding, df)
    assert set(repair_log) == {
        "repair_applied", "duplicates_removed",
        "impossible_rows_dropped", "type_casts", "coerced_to_null"}
    assert repair_log["repair_applied"]
    assert repair_log["duplicates_removed"] == 1
