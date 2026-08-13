"""Unit tests for the stage-4 @tool wrappers (A2.4): strategy building,
per-op execution previews, and the DQ re-check on cleaned output."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from shared.tools.cleaning import (
    cleaning_strategy_tool,
    dedup_tool,
    dq_recheck_tool,
    fillna_tool,
    flag_column_tool,
    iqr_outlier_tool,
    type_caster_tool,
)

UNDERSTANDING = json.dumps({
    "detected_domain": "generic", "domain_confidence": 0.0,
    "entities": [],
    "temporal_columns": ["date"], "dimensions": ["category"],
    "measures": ["revenue"], "identifiers": [],
    "columns": [
        {"name": "date", "role": "temporal", "dtype": "str",
         "nunique": 2, "nullable": True},
        {"name": "category", "role": "dimension", "dtype": "str",
         "nunique": 2, "nullable": False},
        {"name": "revenue", "role": "measure", "dtype": "float64",
         "nunique": 3, "nullable": True},
    ],
    "has_temporal_data": True, "limitations": [],
})

REPORT = json.dumps({
    "status": "needs_repair",
    "invalid": {},
    "missingness": {
        "rate": 0.166, "pattern": "random", "assessment": "MCAR",
        "by_column": {
            "date": {"missing": 1, "rate": 0.5, "assessment": "MCAR"},
            "category": {"missing": 0, "rate": 0.0, "assessment": "none"},
            "revenue": {"missing": 1, "rate": 0.5, "assessment": "MCAR"},
        },
    },
    "duplicates": 1,
    "issues": [],
})

PROFILE = json.dumps({
    "file_name": "sales.csv", "file_hash": "sha256:fixture",
    "row_count": 6, "column_count": 3,
    "columns": ["date", "category", "revenue"],
    "column_types": {"date": "str", "category": "str",
                     "revenue": "float64"},
    "missing_values": {"date": 1, "revenue": 1},
    "nunique": {}, "sample": [],
})

CONTEXT = json.dumps({"file_name": "sales.csv", "generic_mode": True})


def _write_csv(tmp_path, rows) -> str:
    df = pd.DataFrame(rows)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _parse(call) -> object:
    return json.loads(call)


def test_strategy_tool_returns_deterministic_default():
    out = _parse(cleaning_strategy_tool.run(UNDERSTANDING, REPORT))
    assert out["errors"] == []
    by_name = {c["column"]: c for c in out["strategy"]["columns"]}
    assert by_name["revenue"]["action"] == "median_fill_flag"
    assert by_name["date"]["action"] == "drop_row"
    assert by_name["category"]["action"] == "keep"
    assert out["strategy"]["deduplicate"] is True


def test_strategy_tool_normalizes_proposal():
    out = _parse(cleaning_strategy_tool.run(
        UNDERSTANDING, REPORT,
        json.dumps({"columns": [{"column": "revenue",
                                 "action": "flag_and_preserve"}],
                    "deduplicate": False})))
    assert out["errors"] == []
    assert out["strategy"]["columns"][0]["action"] == "flag_and_preserve"


def test_strategy_tool_reports_bad_proposal():
    out = _parse(cleaning_strategy_tool.run(
        UNDERSTANDING, REPORT,
        json.dumps({"columns": [{"column": "nope", "action": "keep"}]})))
    assert out["errors"]


def test_fillna_tool_median(tmp_path):
    csv = _write_csv(tmp_path, {"revenue": [10.0, None, 30.0]})
    out = _parse(fillna_tool.run(csv, UNDERSTANDING, "revenue",
                                 "median_fill"))
    assert any(op["op"] == "fillna_median" for op in out["ops"])


def test_fillna_tool_rejects_bad_method(tmp_path):
    csv = _write_csv(tmp_path, {"revenue": [10.0, None, 30.0]})
    out = _parse(fillna_tool.run(csv, UNDERSTANDING, "revenue", "evil"))
    assert "error" in out


def test_flag_column_tool_counts_rows(tmp_path):
    csv = _write_csv(tmp_path, {"revenue": [10.0, None, 30.0]})
    out = _parse(flag_column_tool.run(csv, UNDERSTANDING, "revenue"))
    assert out["flag"] == "revenue_missing_flag"
    assert out["rows_flagged"] == 1


def test_type_caster_tool_casts_by_role(tmp_path):
    csv = _write_csv(tmp_path, {"revenue": ["10", "abc", "30"]})
    out = _parse(type_caster_tool.run(csv, UNDERSTANDING, "revenue"))
    assert out["casts"][0]["detail"].endswith("->float64")


def test_dedup_tool_counts(tmp_path):
    csv = _write_csv(tmp_path, {"a": [1, 1, 2, 3]})
    out = _parse(dedup_tool.run(csv))
    assert out["duplicates_removed"] == 1
    assert out["rows_before"] == 4


def test_iqr_outlier_tool_flag_and_drop(tmp_path):
    csv = _write_csv(tmp_path, {"revenue": [1, 2, 3, 4, 5, 100]})
    flagged = _parse(iqr_outlier_tool.run(csv, UNDERSTANDING, "revenue",
                                          "flag"))
    assert flagged["outliers"] == 1
    dropped = _parse(iqr_outlier_tool.run(csv, UNDERSTANDING, "revenue",
                                          "drop"))
    assert dropped["outliers"] == 1


def test_iqr_outlier_tool_rejects_bad_mode(tmp_path):
    csv = _write_csv(tmp_path, {"revenue": [1, 2, 3]})
    out = _parse(iqr_outlier_tool.run(csv, UNDERSTANDING, "revenue",
                                      "delete"))
    assert "error" in out


def test_dq_recheck_passed_on_clean(tmp_path):
    csv = _write_csv(tmp_path, {
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "category": ["A", "A", "B"],
        "revenue": [10.0, 20.0, 30.0]})
    out = _parse(dq_recheck_tool.run(csv, UNDERSTANDING, PROFILE, CONTEXT))
    assert out["status"] == "passed"


def test_dq_recheck_needs_repair_on_duplicates(tmp_path):
    csv = _write_csv(tmp_path, {
        "date": ["2024-01-01", "2024-01-01", "2024-01-03"],
        "category": ["A", "A", "B"],
        "revenue": [10.0, 10.0, 30.0]})
    out = _parse(dq_recheck_tool.run(csv, UNDERSTANDING, PROFILE))
    assert out["status"] == "needs_repair"
    assert out["report"]["duplicates"] == 1
