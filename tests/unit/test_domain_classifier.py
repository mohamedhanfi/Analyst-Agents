"""Unit tests for build_domain_facts + domain_classifier_tool (§2.2)."""
from __future__ import annotations

import json

from shared.core.understanding import build_domain_facts
from shared.schemas import DataProfile
from shared.tools.understanding import domain_classifier_tool


def make_profile(sample_rows: int = 3) -> DataProfile:
    sample = [{"order_id": i, "city": "Cairo", "revenue": 10.5 * i}
              for i in range(sample_rows)]
    return DataProfile(
        file_name="sales.csv", file_hash="sha256:abc",
        row_count=100, column_count=3,
        columns=["order_id", "city", "revenue"],
        column_types={"order_id": "int64", "city": "object",
                      "revenue": "float64"},
        nunique={"order_id": 100, "city": 12, "revenue": 87},
        sample=sample, validation_status="passed",
    )


def test_domain_facts_include_columns_and_sample():
    out = build_domain_facts(make_profile())
    facts = out["domain_facts"]
    assert facts["row_count"] == 100
    assert [c["name"] for c in facts["columns"]] == ["order_id", "city", "revenue"]
    assert facts["columns"][0]["suggested_role"] == "identifier"
    assert len(facts["sample"]) == 3


def test_domain_decision_skeleton_for_llm():
    out = build_domain_facts(make_profile())
    decision = out["domain_decision"]
    assert decision["detected_domain"] is None
    assert decision["domain_confidence"] is None
    assert decision["entities"] == []


def test_sample_preserved_verbatim():
    out = build_domain_facts(make_profile())
    assert out["domain_facts"]["sample"][2]["revenue"] == 21.0


def test_empty_sample_ok():
    out = build_domain_facts(make_profile(sample_rows=0))
    assert out["domain_facts"]["sample"] == []


def test_domain_classifier_tool_roundtrip():
    payload = json.loads(domain_classifier_tool.run(
        make_profile().model_dump_json()))
    assert payload["domain_facts"]["row_count"] == 100
    assert payload["domain_facts"]["columns"][1]["suggested_role"] == "dimension"
    assert payload["domain_decision"]["entities"] == []
