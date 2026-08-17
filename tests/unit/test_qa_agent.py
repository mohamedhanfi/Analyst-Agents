"""Stage 8 QA agent tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from agents.qa_agent import run_qa, _deterministic_review


def _make_qa_run(tmp_path: Path) -> Path:
    """Create a minimal run dir with all QA-expected artifacts."""
    run_dir = tmp_path / "run_qa"
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "outputs" / "charts").mkdir(parents=True)
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "knowledge").mkdir(parents=True)
    (run_dir / "data" / "processed").mkdir(parents=True)

    # Minimal cleaned CSV
    (run_dir / "data" / "processed" / "cleaned_data.csv").write_text(
        "revenue,quantity,order_id\n100,5,O-1\n200,3,O-2\n150,4,O-3\n",
        encoding="utf-8")

    # Analysis plan with one KPI
    (run_dir / "metadata" / "analysis_plan.json").write_text(
        json.dumps({"candidate_kpis": [
            {"kpi_id": "K1", "name": "Total Revenue",
             "operation": {"function": "sum", "column": "revenue"}},
        ], "statistical_tests": [], "has_temporal_data": False,
        "limitations": []}), encoding="utf-8")

    # Reported KPI matching the plan
    (run_dir / "outputs" / "kpis.json").write_text(
        json.dumps({"kpis": [
            {"kpi_id": "K1", "name": "Total Revenue", "value": 450.0,
             "operation": {"function": "sum", "column": "revenue",
                           "column_a": None, "column_b": None,
                           "method": None, "over_column": None,
                           "period": None, "group_by": None,
                           "filter": None},
             "evidence_id": "EV-001", "computed_by": "pandas"},
        ]}), encoding="utf-8")

    # Evidence registry
    (run_dir / "outputs" / "evidence_registry.json").write_text(
        json.dumps([{"evidence_id": "EV-001",
                     "source": {"aggregation": "sum"}}]),
        encoding="utf-8")

    # Insights + recs
    (run_dir / "outputs" / "insights.json").write_text(
        json.dumps({"insights": [
            {"insight_id": "INS-1", "title": "Revenue is 450",
             "description": "Total revenue across all orders.",
             "claim_type": "DESCRIPTIVE", "confidence": "high",
             "evidence_ids": ["EV-001"]},
        ], "recommendations": [
            {"recommendation_id": "R1", "insight_id": "INS-1",
             "description": "Monitor revenue trend.",
             "potential_impass": "Early signal"},
        ], "warnings": []}), encoding="utf-8")

    # Stats
    (run_dir / "outputs" / "statistical_results.json").write_text(
        json.dumps({"results": []}), encoding="utf-8")

    # Chart metadata
    (run_dir / "metadata" / "chart_metadata.json").write_text(
        json.dumps({"charts": [], "charts_truncated": False}),
        encoding="utf-8")

    # DQ report
    (run_dir / "metadata" / "data_quality_report.json").write_text(
        json.dumps({"summary": {"total_rules": 5, "fail_count": 0}}),
        encoding="utf-8")

    # Profile + understanding
    (run_dir / "metadata" / "data_profile.json").write_text(
        json.dumps({"row_count": 3, "column_count": 3}),
        encoding="utf-8")
    (run_dir / "metadata" / "dataset_understanding.json").write_text(
        json.dumps({"domain": "retail"}),
        encoding="utf-8")

    # Report with all sections
    (run_dir / "report.html").write_text(
        "<html><body>"
        '<div id="s1"></div><div id="s2"></div><div id="s3"></div>'
        '<div id="s4"></div><div id="s5"></div><div id="s6"></div>'
        "</body></html>", encoding="utf-8")

    return run_dir


# ---------------------------------------------------------------------------
# Deterministic review
# ---------------------------------------------------------------------------


def test_deterministic_review_clean(tmp_path: Path) -> None:
    run_dir = _make_qa_run(tmp_path)
    review = _deterministic_review(run_dir)
    assert review["logic_ok"] is True
    assert review["readability_ok"] is True
    assert review["notes"] == []


def test_deterministic_review_bad_ref(tmp_path: Path) -> None:
    run_dir = _make_qa_run(tmp_path)
    # Overwrite insights.json with a broken recommendation ref
    (run_dir / "outputs" / "insights.json").write_text(
        json.dumps({"insights": [], "recommendations": [
            {"recommendation_id": "R1", "insight_id": "INS-GHOST",
             "description": "Do X", "potential_impact": "Y"},
        ], "warnings": []}), encoding="utf-8")
    review = _deterministic_review(run_dir)
    assert review["logic_ok"] is False
    assert any("INS-GHOST" in n for n in review["notes"])


# ---------------------------------------------------------------------------
# run_qa (no crew — deterministic path)
# ---------------------------------------------------------------------------


def test_run_qa_deterministic(tmp_path: Path) -> None:
    run_dir = _make_qa_run(tmp_path)
    result = run_qa(run_dir, use_crew=False)
    assert result["stage"] == "qa"
    assert result["verdict"] in ("APPROVED", "APPROVED_WITH_WARNINGS",
                                  "NEEDS_REVISION")
    assert "score" in result
    assert result["qa_verdict_path"].endswith("qa_verdict.json")


def test_run_qa_writes_verdict_file(tmp_path: Path) -> None:
    run_dir = _make_qa_run(tmp_path)
    run_qa(run_dir, use_crew=False)
    verdict_path = run_dir / "metadata" / "qa_verdict.json"
    assert verdict_path.exists()
    data = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert "verdict" in data
    assert "score" in data
