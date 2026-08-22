import json

import pandas as pd
import pytest

from shared.core.deep_profile import (
    categorize_missing,
    deep_missingness_report,
    deep_outlier_report,
    impact_analysis,
)
from shared.schemas import ColumnUnderstanding, DatasetUnderstanding


@pytest.fixture
def cfg():
    return {}


def _understanding(roles: dict, df: pd.DataFrame) -> DatasetUnderstanding:
    return DatasetUnderstanding(
        detected_domain="generic", domain_confidence=0.0, entities=[],
        columns=[ColumnUnderstanding(name=name, role=role,
                                     dtype=str(df[name].dtype),
                                     nunique=int(df[name].nunique()),
                                     nullable=bool(df[name].isna().any()))
                 for name, role in roles.items()])


def test_categorize_missing_sentinels():
    s = pd.Series(["", "N/A", "n/a", "unknown", None, "null", "-", "?",
                   "real", "  ", "x"])
    counts = categorize_missing(s)
    assert counts["nan"] == 1
    assert counts["blank"] == 2          # "" and "  "
    assert counts["na"] == 5             # N/A, n/a, null, -, ?
    assert counts["unknown"] == 1
    assert counts["zero"] == 0


def test_categorize_missing_zero_distinguished():
    s = pd.Series([0.0, 5.0, None, 0.0])
    counts = categorize_missing(s)
    assert counts["nan"] == 1
    assert counts["zero"] == 2


def test_deep_missingness_report_structure():
    df = pd.DataFrame({
        "region": ["north", "south", "north", "south", "north"],
        "revenue": [100.0, None, 200.0, None, "N/A"],
        "date": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01",
                 "2024-03-01"],
    })
    u = _understanding({"region": "dimension", "revenue": "measure",
                        "date": "temporal"}, df)
    rep = deep_missingness_report(u, df)
    rev = rep["by_column"]["revenue"]
    assert rev["missing"] == 2
    assert rev["sentinel_counts"]["na"] == 1
    assert rev["effective_missing"] == 3
    assert rev["imputability"]["verdict"] == "impute_median"
    assert "region" in rev["by_segment"]
    assert "2024-01" in rep["time_trend"]["revenue"]
    assert rep["assessment"] == "high"   # 3/5 rows effectively missing


def test_co_missing_pairs_detected():
    df = pd.DataFrame({
        "a": [None, None, 1, 2, 3, None],
        "b": [None, None, 1, 2, 3, None],
        "c": [1, 2, 3, 4, 5, 6],
    })
    u = _understanding({"a": "measure", "b": "measure", "c": "measure"}, df)
    rep = deep_missingness_report(u, df)
    pairs = rep["co_missing_pairs"]
    assert any(p["column_a"] == "a" and p["column_b"] == "b"
               for p in pairs)


def test_outlier_mad_flags():
    df = pd.DataFrame({
        "region": ["x"] * 19 + ["y"],
        "revenue": list(range(1, 20)) + [5000.0],
    })
    u = _understanding({"region": "dimension", "revenue": "measure"}, df)
    rep = deep_outlier_report(u, df)
    rev = rep["revenue"]
    assert rev["flag"].startswith("outliers_mad_")
    assert rev["worst"][0]["row_index"] == 19
    assert rev["worst"][0]["value"] == 5000.0


def test_outlier_context_by_segment():
    df = pd.DataFrame({
        "region": ["north"] * 10 + ["south"] * 10,
        "revenue": list(range(10, 20)) + list(range(1, 11)),
    })
    u = _understanding({"region": "dimension", "revenue": "measure"}, df)
    rep = deep_outlier_report(u, df)
    assert rep["revenue"]["by_segment"] == {}


def test_impact_analysis_deltas():
    before = pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0],
        "region": ["a", "b", "a", "b"],
    })
    after = before[before["revenue"] >= 200].reset_index(drop=True)
    u = _understanding({"revenue": "measure", "region": "dimension"}, before)
    imp = impact_analysis(u, before, after)
    assert imp["rows_before"] == 4
    assert imp["rows_after"] == 3
    assert imp["rows_removed"] == 1
    assert imp["kpi"]["sum_before"] == 1000.0
    assert imp["kpi"]["sum_after"] == 900.0
    assert imp["kpi"]["delta"] == -100.0
    assert imp["dimension_cardinality"]["region"]["before"] == 2


def test_pipeline_writes_deep_profile_and_impact(tmp_path, cfg):
    from agents.cleaning_agent import run_cleaning
    from agents.data_quality import run_data_quality
    from agents.ingestion_agent import run_ingestion
    from agents.understanding_agent import run_understanding

    rows = [
        {"date": "2024-01-01", "region": "north", "revenue": 100.0},
        {"date": "2024-01-02", "region": "south", "revenue": None},
        {"date": "2024-01-03", "region": "north", "revenue": 200.0},
        {"date": "2024-01-04", "region": "south", "revenue": 400.0},
        {"date": "2024-01-05", "region": "north", "revenue": "N/A"},
    ]
    csv = tmp_path / "sales.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    run_dir = tmp_path / "r1"
    provider = lambda _p: "Track revenue"
    assert run_ingestion(str(csv), run_dir=run_dir, cfg=cfg,
                         answer_provider=provider)["status"] == "passed"
    assert run_understanding(run_dir, cfg=cfg)["status"] == "passed"
    dq = run_data_quality(run_dir, cfg=cfg)
    assert dq["deep_profile_path"] is not None
    profile = json.loads(Path(dq["deep_profile_path"]).read_text(
        encoding="utf-8"))
    assert "missingness" in profile and "outliers" in profile
    assert "impact_raw_to_validated" in profile
    assert run_cleaning(run_dir, cfg=cfg)["status"] == "passed"
    impact = json.loads(run_dir.joinpath("metadata/impact_cleaning.json")
                        .read_text(encoding="utf-8"))
    assert impact["rows_before"] == 5
    assert impact["rows_after"] == impact["rows_before"] - impact[
        "rows_removed"]


from pathlib import Path  # noqa: E402