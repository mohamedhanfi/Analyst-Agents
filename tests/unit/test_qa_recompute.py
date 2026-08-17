"""Stage 8 QA — recomputation + reference validation tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from analysis.qa_recompute import (
    QaCheck,
    compare_kpis,
    recompute_kpis,
    run_all_checks,
    validate_references,
    _op_key,
)


# ---------------------------------------------------------------------------
# _op_key
# ---------------------------------------------------------------------------


def test_op_key_deterministic() -> None:
    op = {"function": "sum", "column": "revenue", "column_a": None,
          "column_b": None, "method": None, "over_column": None,
          "period": None, "group_by": None, "filter": None}
    k1 = _op_key(op)
    k2 = _op_key(op)
    assert k1 == k2
    assert "sum" in k1
    assert "revenue" in k1


def test_op_key_differs_by_function() -> None:
    op1 = {"function": "sum", "column": "revenue", "column_a": None,
           "column_b": None, "method": None, "over_column": None,
           "period": None, "group_by": None, "filter": None}
    op2 = {"function": "mean", "column": "revenue", "column_a": None,
           "column_b": None, "method": None, "over_column": None,
           "period": None, "group_by": None, "filter": None}
    assert _op_key(op1) != _op_key(op2)


# ---------------------------------------------------------------------------
# compare_kpis
# ---------------------------------------------------------------------------


def _make_kpi(kpi_id: str, name: str, value: Any, function: str = "sum",
              column: str = "revenue") -> Dict:
    return {
        "kpi_id": kpi_id, "name": name, "value": value,
        "operation": {
            "function": function, "column": column,
            "column_a": None, "column_b": None, "method": None,
            "over_column": None, "period": None, "group_by": None,
            "filter": None,
        },
        "evidence_id": f"EV-{kpi_id}", "computed_by": "pandas",
    }


def test_compare_kpis_exact_match() -> None:
    reported = [_make_kpi("K1", "Revenue", 1000.0)]
    recomputed = {
        "K1": {"value": 1000.0, "name": "Revenue",
               "operation": {"function": "sum", "column": "revenue",
                             "column_a": None, "column_b": None,
                             "method": None, "over_column": None,
                             "period": None, "group_by": None,
                             "filter": None}},
    }
    checks = compare_kpis(reported, recomputed)
    crits = [c for c in checks if c.severity == "critical"]
    assert len(crits) == 0


def test_compare_kpis_within_tolerance() -> None:
    reported = [_make_kpi("K1", "Revenue", 1000.0)]
    recomputed = {
        "K1": {"value": 1000.05, "name": "Revenue",
               "operation": {"function": "sum", "column": "revenue",
                             "column_a": None, "column_b": None,
                             "method": None, "over_column": None,
                             "period": None, "group_by": None,
                             "filter": None}},
    }
    checks = compare_kpis(reported, recomputed)
    crits = [c for c in checks if c.severity == "critical"]
    assert len(crits) == 0


def test_compare_kpis_beyond_tolerance() -> None:
    reported = [_make_kpi("K1", "Revenue", 1000.0)]
    recomputed = {
        "K1": {"value": 1050.0, "name": "Revenue",
               "operation": {"function": "sum", "column": "revenue",
                             "column_a": None, "column_b": None,
                             "method": None, "over_column": None,
                             "period": None, "group_by": None,
                             "filter": None}},
    }
    checks = compare_kpis(reported, recomputed)
    crits = [c for c in checks if c.severity == "critical"]
    assert len(crits) == 1
    assert "mismatch" in crits[0].check


def test_compare_kpis_empty_recomputed() -> None:
    reported = [_make_kpi("K1", "Revenue", 1000.0)]
    checks = compare_kpis(reported, {})
    crits = [c for c in checks if c.severity == "critical"]
    assert any("no KPIs" in c.message for c in crits)


def test_compare_kpis_null_value() -> None:
    reported = [_make_kpi("K1", "Revenue", None)]
    recomputed = {"K1": {"value": 1000.0, "name": "Revenue", "operation": {}}}
    checks = compare_kpis(reported, recomputed)
    warns = [c for c in checks if c.severity == "warning"]
    assert any("null" in c.message for c in warns)


def test_compare_kpis_no_match() -> None:
    reported = [_make_kpi("K1", "Revenue", 1000.0, function="sum")]
    recomputed = {
        "K2": {"value": 500.0, "name": "Orders",
               "operation": {"function": "count", "column": "order_id",
                             "column_a": None, "column_b": None,
                             "method": None, "over_column": None,
                             "period": None, "group_by": None,
                             "filter": None}},
    }
    checks = compare_kpis(reported, recomputed)
    warns = [c for c in checks if c.severity == "warning"]
    assert any("no matching" in c.message for c in warns)


# ---------------------------------------------------------------------------
# validate_references
# ---------------------------------------------------------------------------


def _make_run_refs(tmp_path: Path, **overrides: Any) -> Path:
    run_dir = tmp_path / "run_refs"
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "outputs" / "charts").mkdir(parents=True)
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "knowledge").mkdir(parents=True)

    evidence = overrides.get("evidence", [
        {"evidence_id": "EV-001", "source": {"aggregation": "sum"}},
    ])
    kpis = overrides.get("kpis", [
        {"kpi_id": "K1", "name": "Revenue", "value": 100,
         "evidence_id": "EV-001", "operation": {"function": "sum",
         "column": "revenue", "column_a": None, "column_b": None,
         "method": None, "over_column": None, "period": None,
         "group_by": None, "filter": None}},
    ])
    insights = overrides.get("ins", [
        {"insight_id": "INS-1", "title": "Test", "description": "Desc",
         "claim_type": "DESCRIPTIVE", "confidence": "high",
         "evidence_ids": ["EV-001"]},
    ])
    recs = overrides.get("recs", [
        {"recommendation_id": "R1", "insight_id": "INS-1",
         "description": "Do X", "basis": "Because",
         "potential_impact": "+10%"},
    ])

    (run_dir / "outputs" / "evidence_registry.json").write_text(
        json.dumps(evidence), encoding="utf-8")
    (run_dir / "outputs" / "kpis.json").write_text(
        json.dumps({"kpis": kpis}), encoding="utf-8")
    (run_dir / "outputs" / "insights.json").write_text(
        json.dumps({"insights": insights, "recommendations": recs,
                     "warnings": []}), encoding="utf-8")
    (run_dir / "metadata" / "chart_metadata.json").write_text(
        json.dumps({"charts": [], "charts_truncated": False}),
        encoding="utf-8")
    # report.html with all sections
    (run_dir / "report.html").write_text(
        "<html><body>"
        '<div id="s1"></div><div id="s2"></div><div id="s3"></div>'
        '<div id="s4"></div><div id="s5"></div><div id="s6"></div>'
        "</body></html>",
        encoding="utf-8")
    return run_dir


def test_validate_references_clean(tmp_path: Path) -> None:
    run_dir = _make_run_refs(tmp_path)
    checks = validate_references(run_dir)
    crits = [c for c in checks if c.severity == "critical"]
    assert len(crits) == 0


def test_validate_references_bad_kpi_evidence(tmp_path: Path) -> None:
    run_dir = _make_run_refs(tmp_path, kpis=[
        {"kpi_id": "K1", "name": "Revenue", "value": 100,
         "evidence_id": "EV-MISSING", "operation": {"function": "sum",
         "column": "revenue", "column_a": None, "column_b": None,
         "method": None, "over_column": None, "period": None,
         "group_by": None, "filter": None}},
    ])
    checks = validate_references(run_dir)
    crits = [c for c in checks if c.severity == "critical"
             and c.check == "kpi_evidence_ref"]
    assert len(crits) == 1


def test_validate_references_bad_insight_evidence(tmp_path: Path) -> None:
    run_dir = _make_run_refs(tmp_path, ins=[
        {"insight_id": "INS-1", "title": "Test", "description": "Desc",
         "claim_type": "DESCRIPTIVE", "confidence": "high",
         "evidence_ids": ["EV-GHOST"]},
    ])
    checks = validate_references(run_dir)
    crits = [c for c in checks if c.severity == "critical"
             and c.check == "insight_evidence_ref"]
    assert len(crits) == 1


def test_validate_references_bad_rec_insight(tmp_path: Path) -> None:
    run_dir = _make_run_refs(tmp_path, recs=[
        {"recommendation_id": "R1", "insight_id": "INS-GHOST",
         "description": "Do X", "basis": "Because",
         "potential_impact": "+10%"},
    ])
    checks = validate_references(run_dir)
    crits = [c for c in checks if c.severity == "critical"
             and c.check == "rec_insight_ref"]
    assert len(crits) == 1


def test_validate_references_missing_report(tmp_path: Path) -> None:
    run_dir = _make_run_refs(tmp_path)
    (run_dir / "report.html").unlink()
    checks = validate_references(run_dir)
    warns = [c for c in checks if c.check == "report_missing"]
    assert len(warns) == 1


def test_validate_references_missing_section(tmp_path: Path) -> None:
    run_dir = _make_run_refs(tmp_path)
    (run_dir / "report.html").write_text(
        "<html><body><div id='s1'></div></body></html>",
        encoding="utf-8")
    checks = validate_references(run_dir)
    warns = [c for c in checks if c.check == "report_section_missing"]
    assert len(warns) >= 1


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------


def test_run_all_checks_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "outputs").mkdir()
    (empty / "metadata").mkdir()
    checks = run_all_checks(empty)
    assert len(checks) > 0
    crits = [c for c in checks if c.severity == "critical"]
    assert len(crits) >= 1
