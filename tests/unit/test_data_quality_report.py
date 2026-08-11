"""Integration-style tests for the stage-3 report (A2.3): assemble_report
over the real sales_demo fixture + gate semantics on controlled frames."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shared.core.data_quality import assemble_report
from shared.schemas import (
    BusinessContext,
    ColumnUnderstanding,
    DataProfile,
    DatasetUnderstanding,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sales_demo.csv"


def build_understanding(dtypes: dict[str, str]) -> DatasetUnderstanding:
    roles = {
        "order_id": "identifier", "date": "temporal",
        "product": "dimension", "category": "dimension",
        "revenue": "measure", "quantity": "measure",
        "customer_email": "free_text",
    }
    columns = [ColumnUnderstanding(name=name, role=roles[name],
                                   dtype=dtypes.get(name, "object"),
                                   nunique=0, nullable=False)
               for name in roles]
    return DatasetUnderstanding(
        detected_domain="sales", domain_confidence=0.8, columns=columns,
        measures=["revenue", "quantity"],
        temporal_columns=["date"],
        dimensions=["product", "category"],
        identifiers=["order_id"])


def build_profile(df: pd.DataFrame, missing_revenue: int,
                  duplicate_rows: int) -> DataProfile:
    return DataProfile(
        file_name="sales_demo.csv", file_hash="sha256:fixture",
        row_count=len(df), column_count=len(df.columns),
        columns=list(df.columns),
        column_types={c: str(t) for c, t in df.dtypes.items()},
        missing_values={"revenue": missing_revenue},
        duplicate_rows=duplicate_rows,
        nunique={c: int(df[c].nunique()) for c in df.columns})


# ---------------------------------------------------------------------------
# the real fixture: sales_demo.csv has 2 refunds, 2 missing, 2 exact dups
# ---------------------------------------------------------------------------


def test_fixture_report_expected_findings():
    df = pd.read_csv(FIXTURE, encoding="utf-8-sig")
    understanding = build_understanding(
        {c: str(t) for c, t in df.dtypes.items()})
    profile = build_profile(df, missing_revenue=2, duplicate_rows=2)
    context = BusinessContext(file_name="sales_demo.csv", generic_mode=True)

    report, repair_log = assemble_report(understanding, profile, df,
                                         context)

    assert report.status == "needs_repair"
    assert report.invalid == {"revenue": ["negative"]}
    assert report.duplicates == 2

    missing = report.missingness
    assert missing["by_column"]["revenue"]["missing"] == 2
    assert missing["by_column"]["revenue"]["rate"] == pytest.approx(
        2 / 202, abs=0.001)
    assert missing["assessment"] == "MCAR"

    details = {i["column"]: i["detail"] for i in report.issues
               if i["category"] == "invalid_value"}
    assert details["revenue"] == "negative"
    assert any(i["severity"] == "high"
               and i["category"] == "invalid_value"
               for i in report.issues)

    assert repair_log["duplicates_removed"] == 2
    assert repair_log["impossible_rows_dropped"] == {}
    assert repair_log["type_casts"]["date"].startswith("object->")
    assert repair_log["repair_applied"]


def test_fixture_report_does_not_touch_negatives():
    df = pd.read_csv(FIXTURE, encoding="utf-8-sig")
    understanding = build_understanding(
        {c: str(t) for c, t in df.dtypes.items()})
    profile = build_profile(df, missing_revenue=2, duplicate_rows=2)
    context = BusinessContext(file_name="sales_demo.csv", generic_mode=True)

    _, repair_log = assemble_report(understanding, profile, df, context)
    negatives = df.loc[df["revenue"] < 0, "revenue"].tolist()
    assert negatives  # the fixture really has refunds
    assert repair_log["coerced_to_null"] == {}
    assert repair_log["type_casts"].get("revenue") is None


# ---------------------------------------------------------------------------
# gate semantics
# ---------------------------------------------------------------------------


def test_clean_frame_passes():
    df = pd.DataFrame({
        "order_id": [1, 2, 3],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "revenue": [10.0, 20.0, 30.0],
        "quantity": [1, 2, 3],
    })
    understanding = DatasetUnderstanding(
        detected_domain="sales", domain_confidence=0.8,
        columns=[
            ColumnUnderstanding(name="order_id", role="identifier",
                                dtype="int64", nunique=3, nullable=False),
            ColumnUnderstanding(name="date", role="temporal",
                                dtype="datetime64[ns]", nunique=3,
                                nullable=False),
            ColumnUnderstanding(name="revenue", role="measure",
                                dtype="float64", nunique=3, nullable=False),
            ColumnUnderstanding(name="quantity", role="measure",
                                dtype="int64", nunique=3, nullable=False),
        ],
        measures=["revenue", "quantity"], temporal_columns=["date"],
        identifiers=["order_id"])
    profile = build_profile(df, missing_revenue=0, duplicate_rows=0)
    context = BusinessContext(file_name="clean.csv", generic_mode=True)

    report, repair_log = assemble_report(understanding, profile, df,
                                         context)
    assert report.status == "passed"
    assert report.invalid == {}
    assert report.duplicates == 0
    assert report.missingness["assessment"] == "none"
    assert not repair_log["repair_applied"]


def test_high_severity_forces_needs_repair_even_without_repair():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "revenue": [-1.0],
    })
    understanding = DatasetUnderstanding(
        detected_domain="sales", domain_confidence=0.8,
        columns=[
            ColumnUnderstanding(name="date", role="temporal",
                                dtype="datetime64[ns]", nunique=1,
                                nullable=False),
            ColumnUnderstanding(name="revenue", role="measure",
                                dtype="float64", nunique=1, nullable=False),
        ],
        measures=["revenue"], temporal_columns=["date"])
    profile = build_profile(df, missing_revenue=0, duplicate_rows=0)
    context = BusinessContext(file_name="neg.csv", generic_mode=True)

    report, repair_log = assemble_report(understanding, profile, df,
                                         context)
    assert report.status == "needs_repair"
    assert report.invalid == {"revenue": ["negative"]}
    assert not repair_log["repair_applied"]
