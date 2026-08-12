"""Unit tests for the stage-3 @tool wrappers (A2.3): partial JSON inputs
must never crash, outputs must be JSON-serializable."""
from __future__ import annotations

import json
from pathlib import Path

from shared.tools.data_quality import (
    business_rules_checker_tool,
    deterministic_repair_tool,
    duplicate_detector_tool,
    invalid_value_checker_tool,
    missingness_analyzer_tool,
    referential_integrity_tool,
    schema_checker_tool,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sales_demo.csv"

PROFILE = json.dumps({
    "file_name": "sales_demo.csv", "file_hash": "sha256:fixture",
    "row_count": 202, "column_count": 7,
    "columns": ["order_id", "date", "product", "category", "revenue",
                "quantity", "customer_email"],
    "column_types": {c: "string" for c in
                     ["order_id", "date", "product", "category",
                      "customer_email"]}
    | {"revenue": "float64", "quantity": "int64"},
    "missing_values": {"revenue": 2}, "nunique": {}, "sample": []})

UNDERSTANDING = json.dumps({
    "detected_domain": "sales", "domain_confidence": 0.8,
    "columns": [
        {"name": "order_id", "role": "identifier", "dtype": "string",
         "nunique": 200, "nullable": False},
        {"name": "date", "role": "dimension", "dtype": "string",
         "nunique": 15, "nullable": False},
        {"name": "product", "role": "dimension", "dtype": "string",
         "nunique": 6, "nullable": False},
        {"name": "category", "role": "dimension", "dtype": "string",
         "nunique": 3, "nullable": False},
        {"name": "revenue", "role": "measure", "dtype": "float64",
         "nunique": 197, "nullable": True},
        {"name": "quantity", "role": "measure", "dtype": "int64",
         "nunique": 9, "nullable": False},
        {"name": "customer_email", "role": "free_text", "dtype": "string",
         "nunique": 200, "nullable": False},
    ],
    "measures": ["revenue", "quantity"], "identifiers": ["order_id"]})

CONTEXT = json.dumps({"file_name": "sales_demo.csv", "generic_mode": True})


def _parse_tool_result(call) -> object:
    return json.loads(call)


def test_schema_tool_with_partial_inputs():
    out = _parse_tool_result(
        schema_checker_tool.run(PROFILE, UNDERSTANDING))
    assert isinstance(out, list)


def test_invalid_tool_needs_only_understanding():
    out = _parse_tool_result(
        invalid_value_checker_tool.run(UNDERSTANDING, str(FIXTURE)))
    details = [(i["column"], i["detail"]) for i in out]
    assert ("revenue", "negative") in details


def test_business_rules_tool_needs_only_context():
    out = _parse_tool_result(
        business_rules_checker_tool.run(CONTEXT, str(FIXTURE)))
    assert isinstance(out, list)


def test_missingness_tool_json_safe():
    out = _parse_tool_result(
        missingness_analyzer_tool.run(UNDERSTANDING, str(FIXTURE)))
    assert out["by_column"]["revenue"]["missing"] == 2


def test_duplicates_tool_examples_json_safe():
    out = _parse_tool_result(duplicate_detector_tool.run(str(FIXTURE)))
    assert out["duplicates"] == 2
    assert all(isinstance(row, dict) for row in out["examples"])


def test_referential_tool_with_partial_inputs():
    out = _parse_tool_result(
        referential_integrity_tool.run(UNDERSTANDING, str(FIXTURE)))
    assert isinstance(out, list)


def test_repair_tool_json_safe():
    out = _parse_tool_result(
        deterministic_repair_tool.run(UNDERSTANDING, str(FIXTURE)))
    assert out["duplicates_removed"] == 2
    assert out["repair_applied"]
