"""Unit tests for shared/core/cleaning.py (§2.4): §2.4 strategy table,
deterministic executor (fills/flags/drops/casts/dedup/IQR), strategy
normalization, attempt versioning, CleaningResult assembly."""
from __future__ import annotations

import pandas as pd

from shared.core.cleaning import (
    assemble_cleaning_result,
    build_strategy,
    execute_strategy,
    normalize_strategy,
    persist_attempt,
)
from shared.schemas import (
    CleaningResult,
    ColumnUnderstanding,
    DataQualityReport,
    DatasetUnderstanding,
)


def make_understanding(columns) -> DatasetUnderstanding:
    cols = [ColumnUnderstanding(name=n, role=r, dtype=d, nunique=10,
                                nullable=False)
            for n, r, d in columns]
    return DatasetUnderstanding(
        detected_domain="generic", domain_confidence=0.0, entities=[],
        columns=cols,
        measures=[n for n, r, _ in columns if r == "measure"],
        dimensions=[n for n, r, _ in columns if r in ("dimension",
                                                      "categorical")],
        temporal_columns=[n for n, r, _ in columns if r == "temporal"],
        identifiers=[n for n, r, _ in columns if r == "identifier"],
    )


def make_report(missing=None, duplicates: int = 0) -> DataQualityReport:
    return DataQualityReport(
        status="needs_repair",
        missingness={"rate": 0.0, "pattern": "random", "assessment": "none",
                     "by_column": missing or {}},
        duplicates=duplicates,
    )


def _action_for(role: str, rate: float, assessment: str = "MCAR") -> str:
    understanding = make_understanding([("col", role, "float64")])
    report = make_report(
        {"col": {"missing": 1, "rate": rate, "assessment": assessment}})
    return build_strategy(understanding, report)["columns"][0]["action"]


# ---------------------------------------------------------------------------
# §2.4 strategy table
# ---------------------------------------------------------------------------


def test_measure_mcar_below_5_is_median_fill():
    assert _action_for("measure", 0.02) == "median_fill"


def test_measure_mcar_5_30_is_median_fill_flag():
    assert _action_for("measure", 0.10) == "median_fill_flag"


def test_measure_mar_signal_is_flag_and_preserve():
    assert _action_for("measure", 0.10, "MAR_suspected") == \
        "flag_and_preserve"
    assert _action_for("measure", 0.10, "MNAR_suspected") == \
        "flag_and_preserve"


def test_measure_over_70_is_drop_column():
    assert _action_for("measure", 0.80) == "drop_column"


def test_dimension_mcar_below_5_is_mode_fill():
    assert _action_for("dimension", 0.02) == "mode_fill"


def test_dimension_mcar_5_30_is_unknown_fill():
    assert _action_for("dimension", 0.10) == "unknown_fill"


def test_dimension_mar_signal_is_flag_and_preserve():
    assert _action_for("dimension", 0.10, "MAR_suspected") == \
        "flag_and_preserve"


def test_dimension_over_70_is_keep_flag():
    assert _action_for("dimension", 0.80) == "keep_flag"


def test_temporal_missing_is_drop_row():
    assert _action_for("temporal", 0.10) == "drop_row"


def test_temporal_over_70_is_drop_column():
    assert _action_for("temporal", 0.80) == "drop_column"


def test_identifier_missing_is_drop_row():
    assert _action_for("identifier", 0.10) == "drop_row"


def test_identifier_over_70_is_drop_column():
    assert _action_for("identifier", 0.80) == "drop_column"


def test_no_missingness_is_keep():
    assert _action_for("measure", 0.0) == "keep"


def test_flagged_negative_measure_is_drop_negative():
    understanding = make_understanding([("revenue", "measure", "float64")])
    report = make_report(
        {"revenue": {"missing": 0, "rate": 0.0, "assessment": "none"}})
    report.invalid = {"revenue": ["negative"]}
    strategy = build_strategy(understanding, report)
    assert strategy["columns"][0]["action"] == "drop_negative"


def test_flagged_negative_measure_keeps_missingness_fill():
    """§2.4 regression: a measure flagged negative AND MCAR<5% needs BOTH
    drop_negative and median_fill — the negative flag must not suppress
    the missingness action (single-action-per-column bug)."""
    understanding = make_understanding([("revenue", "measure", "float64")])
    report = make_report(
        {"revenue": {"missing": 1, "rate": 0.01, "assessment": "MCAR"}})
    report.invalid = {"revenue": ["negative"]}
    strategy = build_strategy(understanding, report)
    actions = [c["action"] for c in strategy["columns"]
               if c["column"] == "revenue"]
    assert actions == ["drop_negative", "median_fill"]

    df = pd.DataFrame({"revenue": [-5.0, 10.0, None, 30.0]})
    cleaned, log = execute_strategy(df, strategy, understanding)
    assert cleaned["revenue"].tolist() == [10.0, 20.0, 30.0]
    assert any(op["op"] == "drop_negative" for op in log)
    assert any(op["op"] == "fillna_median" for op in log)
    assert not cleaned["revenue"].isna().any()


def test_normalize_preserves_llm_fill_with_negative_override():
    """The deterministic negative override is ADDED next to the LLM's fill
    action (never replacing it), and runs before it."""
    understanding = make_understanding([("revenue", "measure", "float64")])
    report = make_report(
        {"revenue": {"missing": 1, "rate": 0.01, "assessment": "MCAR"}})
    report.invalid = {"revenue": ["negative"]}
    strategy, errors = normalize_strategy(
        {"columns": [{"column": "revenue", "action": "median_fill"}]},
        understanding, report)
    assert errors == []
    actions = [c["action"] for c in strategy["columns"]
               if c["column"] == "revenue"]
    assert actions == ["drop_negative", "median_fill"]


def test_drop_negative_removes_flagged_rows():
    understanding = make_understanding([("revenue", "measure", "float64")])
    df = pd.DataFrame({"revenue": [-5.0, 10.0, -1.0, 30.0]})
    strategy = {"columns": [{"column": "revenue", "role": "measure",
                             "action": "drop_negative", "detail": ""}],
                "deduplicate": False, "outliers": {}}
    cleaned, log = execute_strategy(df, strategy, understanding)
    assert cleaned["revenue"].tolist() == [10.0, 30.0]
    assert any(op["op"] == "drop_negative" for op in log)


def test_deduplicate_defaults_from_report():
    understanding = make_understanding([("col", "measure", "float64")])
    dup_report = make_report({}, duplicates=4)
    strategy = build_strategy(understanding, dup_report)
    assert strategy["deduplicate"] is True
    clean_report = make_report({}, duplicates=0)
    assert build_strategy(understanding, clean_report)["deduplicate"] is False


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def test_median_fill_uses_median():
    understanding = make_understanding([("revenue", "measure", "float64")])
    report = make_report(
        {"revenue": {"missing": 1, "rate": 0.333, "assessment": "MCAR"}})
    df = pd.DataFrame({"revenue": [10.0, None, 30.0]})
    cleaned, log = execute_strategy(df, build_strategy(understanding, report),
                                    understanding)
    assert cleaned["revenue"].tolist() == [10.0, 20.0, 30.0]
    assert any(op["op"] == "fillna_median" for op in log)


def test_median_fill_flag_creates_missing_flag():
    understanding = make_understanding([("revenue", "measure", "float64")])
    report = make_report(
        {"revenue": {"missing": 1, "rate": 0.10, "assessment": "MCAR"}})
    df = pd.DataFrame({"revenue": [10.0, None, 30.0]})
    cleaned, log = execute_strategy(df, build_strategy(understanding, report),
                                    understanding)
    assert "revenue_missing_flag" in cleaned.columns
    assert cleaned["revenue_missing_flag"].tolist() == [False, True, False]
    assert not cleaned["revenue"].isna().any()


def test_unknown_fill_uses_unknown():
    understanding = make_understanding([("category", "dimension", "str")])
    report = make_report(
        {"category": {"missing": 1, "rate": 0.10, "assessment": "MCAR"}})
    df = pd.DataFrame({"category": ["A", None, "B"]})
    cleaned, _ = execute_strategy(df, build_strategy(understanding, report),
                                  understanding)
    assert cleaned["category"].tolist() == ["A", "Unknown", "B"]


def test_flag_and_preserve_keeps_missingness():
    understanding = make_understanding([("revenue", "measure", "float64")])
    report = make_report(
        {"revenue": {"missing": 1, "rate": 0.10,
                     "assessment": "MAR_suspected"}})
    df = pd.DataFrame({"revenue": [10.0, None, 30.0]})
    cleaned, log = execute_strategy(df, build_strategy(understanding, report),
                                    understanding)
    assert cleaned["revenue"].isna().sum() == 1
    assert "revenue_missing_flag" in cleaned.columns
    assert cleaned["revenue_missing_flag"].tolist() == [False, True, False]
    assert not any(op["op"] == "fillna_median" for op in log)


def test_drop_row_removes_missing_rows():
    understanding = make_understanding([("date", "temporal", "str")])
    report = make_report(
        {"date": {"missing": 1, "rate": 0.10, "assessment": "MCAR"}})
    df = pd.DataFrame({"date": ["2024-01-01", None, "2024-01-03"]})
    cleaned, log = execute_strategy(df, build_strategy(understanding, report),
                                    understanding)
    assert len(cleaned) == 2
    assert any(op["op"] == "drop_row" for op in log)


def test_drop_column_removes_column():
    understanding = make_understanding([("date", "temporal", "str")])
    report = make_report(
        {"date": {"missing": 3, "rate": 0.80, "assessment": "MCAR"}})
    df = pd.DataFrame({"date": ["2024-01-01", None, None]})
    cleaned, log = execute_strategy(df, build_strategy(understanding, report),
                                    understanding)
    assert "date" not in cleaned.columns
    assert any(op["op"] == "drop_column" for op in log)


def test_role_based_type_casts():
    understanding = make_understanding([("revenue", "measure", "float64"),
                                        ("date", "temporal", "str")])
    df = pd.DataFrame({"revenue": ["10", "20"],
                       "date": ["2024-01-01", "2024-01-02"]})
    cleaned, log = execute_strategy(df, build_strategy(understanding,
                                                       make_report()),
                                    understanding)
    assert pd.api.types.is_numeric_dtype(cleaned["revenue"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    assert any(op["op"] == "type_cast" for op in log)


def test_dedup_removes_exact_duplicates():
    understanding = make_understanding([("col", "measure", "float64")])
    df = pd.DataFrame({"col": [1, 1, 2]})
    strategy = {"columns": [], "deduplicate": True, "outliers": {}}
    cleaned, log = execute_strategy(df, strategy, understanding)
    assert len(cleaned) == 2
    assert any(op["op"] == "dedup" for op in log)


def test_iqr_outlier_flag_and_drop():
    understanding = make_understanding([("revenue", "measure", "float64")])
    df = pd.DataFrame({"revenue": [1, 2, 3, 4, 5, 100]})
    base = {"columns": [{"column": "revenue", "role": "measure",
                         "action": "keep", "detail": ""}],
            "deduplicate": False}

    flagged, log = execute_strategy(df, {**base, "outliers": {
        "revenue": "flag"}}, understanding)
    assert "revenue_outlier_flag" in flagged.columns
    assert bool(flagged["revenue_outlier_flag"].iloc[-1])

    dropped, _ = execute_strategy(df, {**base, "outliers": {
        "revenue": "drop"}}, understanding)
    assert len(dropped) == 5


# ---------------------------------------------------------------------------
# Strategy normalization (Python authoritative)
# ---------------------------------------------------------------------------


def test_normalize_accepts_valid_strategy():
    understanding = make_understanding([("revenue", "measure", "float64")])
    strategy, errors = normalize_strategy(
        {"columns": [{"column": "revenue", "action": "median_fill"}],
         "deduplicate": True, "outliers": {"revenue": "flag"}},
        understanding, make_report({}, duplicates=3))
    assert errors == []
    assert strategy["columns"][0]["role"] == "measure"
    assert strategy["deduplicate"] is True


def test_normalize_rejects_unknown_column():
    understanding = make_understanding([("revenue", "measure", "float64")])
    strategy, errors = normalize_strategy(
        {"columns": [{"column": "nope", "action": "keep"}]},
        understanding)
    assert any("unknown column" in e for e in errors)
    assert strategy["columns"] == []


def test_normalize_rejects_unknown_action():
    understanding = make_understanding([("revenue", "measure", "float64")])
    _, errors = normalize_strategy(
        {"columns": [{"column": "revenue", "action": "evil"}]},
        understanding)
    assert any("unknown action" in e for e in errors)


def test_normalize_falls_back_on_bad_json():
    understanding = make_understanding([("revenue", "measure", "float64")])
    strategy, errors = normalize_strategy("{not json", understanding)
    assert errors
    assert strategy["columns"]


def test_normalize_validates_outlier_modes():
    understanding = make_understanding([("revenue", "measure", "float64")])
    _, errors = normalize_strategy(
        {"columns": [], "outliers": {"revenue": "delete"}},
        understanding)
    assert any("flag' or 'drop" in e for e in errors)


# ---------------------------------------------------------------------------
# Attempt versioning + result
# ---------------------------------------------------------------------------


def test_persist_attempt_keeps_lineage(tmp_path):
    df1 = pd.DataFrame({"a": [1, 2]})
    persist_attempt(tmp_path, df1, 1)
    latest = tmp_path / "data" / "processed" / "cleaned_data.csv"
    assert latest.exists()
    assert not (tmp_path / "data" / "processed"
                / "cleaned_data_attempt_1.csv").exists()

    df2 = pd.DataFrame({"a": [3]})
    persist_attempt(tmp_path, df2, 2)
    assert (tmp_path / "data" / "processed"
            / "cleaned_data_attempt_1.csv").exists()
    kept = pd.read_csv(tmp_path / "data" / "processed"
                       / "cleaned_data_attempt_1.csv")
    current = pd.read_csv(latest)
    assert kept["a"].tolist() == [1, 2]
    assert current["a"].tolist() == [3]


def test_assemble_cleaning_result():
    result = assemble_cleaning_result(
        attempt=2, rows_before=100, rows_after=95, duplicates_removed=3,
        type_casts={"date": "object->datetime64"},
        flags_created=["revenue_missing_flag"],
        outliers={"revenue": 2}, status="passed")
    assert isinstance(result, CleaningResult)
    assert result.attempt == 2
    assert result.rows_after == 95
    assert result.outliers == {"revenue": 2}
