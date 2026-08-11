"""Unit tests for stage-3 checks (A2.3, part 2): missingness MCAR/MAR/MNAR,
duplicates, referential integrity."""
from __future__ import annotations

import pandas as pd

from shared.core.data_quality import (
    analyze_missingness,
    check_referential_integrity,
    detect_duplicates,
)
from shared.schemas import ColumnUnderstanding, DatasetUnderstanding


def understanding_for(columns) -> DatasetUnderstanding:
    return DatasetUnderstanding(
        detected_domain="sales", domain_confidence=0.8,
        columns=[ColumnUnderstanding(name=name, role=role, dtype=dtype,
                                     nunique=0, nullable=False)
                 for name, role, dtype in columns],
        measures=[c[0] for c in columns if c[1] == "measure"],
        dimensions=[c[0] for c in columns if c[1] == "dimension"],
        identifiers=[c[0] for c in columns if c[1] == "identifier"],
    )


# ---------------------------------------------------------------------------
# missingness — MCAR / MAR / MNAR
# ---------------------------------------------------------------------------


def test_mcar_detected():
    understanding = understanding_for(
        [("amount", "measure", "float64"),
         ("group", "dimension", "object")])
    df = pd.DataFrame({
        "amount": [1.0, 2.0, float("nan"), 4.0, 5.0, float("nan"),
                   7.0, 8.0, float("nan"), 10.0, 11.0, float("nan")],
        "group": ["x"] * 6 + ["y"] * 6,
    })
    result = analyze_missingness(understanding, df)
    assert result["assessment"] == "MCAR"
    assert result["pattern"] == "random"
    assert result["by_column"]["amount"]["missing"] == 4
    assert result["rate"] == round(4 / (12 * 2), 6)


def test_mar_suspected_detected():
    understanding = understanding_for(
        [("amount", "measure", "float64"),
         ("group", "dimension", "object")])
    df = pd.DataFrame({
        "amount": [float("nan"), float("nan"), float("nan"),
                   4.0, 5.0, 6.0,
                   7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "group": ["x"] * 6 + ["y"] * 6,
    })
    result = analyze_missingness(understanding, df)
    assert result["assessment"] == "MAR_suspected"
    assert result["pattern"] == "non_random"


def test_mnar_suspected_detected():
    understanding = understanding_for(
        [("amount", "measure", "float64"),
         ("seq", "dimension", "int64")])
    df = pd.DataFrame({
        "amount": [float("nan"), float("nan"), float("nan"),
                   4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "seq": list(range(12)),
    })
    result = analyze_missingness(understanding, df)
    assert result["assessment"] == "MNAR_suspected"
    assert result["pattern"] == "non_random"


def test_no_missingness_assessment_none():
    understanding = understanding_for([("amount", "measure", "float64")])
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
    result = analyze_missingness(understanding, df)
    assert result["assessment"] == "none"
    assert result["rate"] == 0.0
    assert result["pattern"] == "random"


# ---------------------------------------------------------------------------
# duplicates
# ---------------------------------------------------------------------------


def test_duplicates_counted_with_examples():
    df = pd.DataFrame({
        "a": [1, 2, 1, 3, 1],
        "b": ["x", "y", "x", "z", "x"],
    })
    count, examples = detect_duplicates(df)
    assert count == 2
    assert len(examples) == 2
    assert examples[0]["a"] == 1


def test_no_duplicates():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert detect_duplicates(df) == (0, [])


# ---------------------------------------------------------------------------
# referential integrity
# ---------------------------------------------------------------------------


def test_orphaned_references_flagged():
    understanding = understanding_for(
        [("order_id", "identifier", "object"),
         ("order_id_alt", "dimension", "object")])
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "order_id_alt": [1, 2, 999, 999, 999],
    })
    issues = check_referential_integrity(understanding, df)
    assert [(i.column, i.detail, i.category) for i in issues] == [
        ("order_id_alt", "orphaned_references_x1",
         "referential_integrity")]


def test_identifier_nulls_flagged():
    understanding = understanding_for(
        [("order_id", "identifier", "object")])
    df = pd.DataFrame({"order_id": [1, 2, None, 4, 5]})
    issues = check_referential_integrity(understanding, df)
    assert [(i.column, i.detail) for i in issues] == [
        ("order_id", "identifier_nulls_x1")]


def test_unrelated_column_not_flagged():
    understanding = understanding_for(
        [("order_id", "identifier", "object"),
         ("product", "dimension", "object")])
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "product": ["A", "B", "C", "A", "B"],
    })
    assert check_referential_integrity(understanding, df) == []
