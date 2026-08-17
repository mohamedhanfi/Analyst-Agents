"""Stage 7 tests — report builder: artifact loading, section rendering, XSS.

The report builder (§2.7) loads all JSON artifacts from a run directory and
renders an HTML report via Jinja2 (autoescape=True).  Every section is
Python-computed; only the executive summary may come from the LLM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from analysis.report_builder import (
    _esc,
    load_artifacts,
    render_business_context,
    render_charts,
    render_dq_summary,
    render_evidence,
    render_insights,
    render_kpis,
    render_limitations,
    render_overview,
    render_recommendations,
    render_report,
    render_stats,
    save_report,
    save_report_result,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
MINIMAL_TPL = FIXTURES / "report_minimal.html"


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------


def _kpi(kpi_id: str, name: str, value: Any, evidence_id: str) -> Dict:
    return {
        "kpi_id": kpi_id, "name": name,
        "operation": {
            "function": "sum", "column": "revenue", "column_a": None,
            "column_b": None, "method": None, "over_column": None,
            "period": None, "basis": None, "as_percent": None,
            "group_by": None, "filter": None, "numerator": None,
            "denominator": None,
        },
        "value": value, "evidence_id": evidence_id, "computed_by": "pandas",
    }


def _stat(test_id: str, test_name: str, variables: List[str],
          statistic: float | None, p_value: float | None,
          evidence_id: str) -> Dict:
    return {
        "test_id": test_id, "category": "correlation",
        "test_name": test_name, "variables": variables,
        "statistic": statistic, "p_value": p_value,
        "ci_low": None, "ci_high": None, "effect_size": None,
        "n": 100, "evidence_id": evidence_id, "extra": {},
    }


def _insight(insight_id: str, title: str, confidence: str,
             claim_type: str) -> Dict:
    return {
        "insight_id": insight_id, "title": title,
        "description": f"Description of {title}.",
        "claim_type": claim_type, "confidence": confidence,
        "evidence_ids": ["EV-001"], "related_kpis": ["KPI-001"],
    }


def _rec(rec_id: str, description: str, basis: str,
         impact: str) -> Dict:
    return {
        "recommendation_id": rec_id, "insight_id": "INS-001",
        "description": description, "basis": basis,
        "potential_impact": impact,
    }


def _evidence(eid: str, agg: str = "sum",
              lineage: List[str] | None = None) -> Dict:
    return {
        "evidence_id": eid,
        "source": {
            "aggregation": agg,
            "lineage": lineage or ["revenue"],
            "result": 1.0,
        },
    }


def _make_run(tmp_path: Path, **overrides: Any) -> Path:
    """Create a minimal run directory with mock JSON artifacts."""
    run_dir = tmp_path / "run_test"
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "outputs" / "charts").mkdir(parents=True)
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "knowledge").mkdir(parents=True)

    kpis = overrides.get("kpis", [_kpi("KPI-001", "Revenue", 1234567.89,
                                       "EV-001")])
    stats = overrides.get("stats", [_stat("ST-001", "pearson",
                                          ["revenue", "quantity"],
                                          0.85, 0.001, "EV-009")])
    insights = overrides.get("insights", [
        _insight("INS-001", "Revenue is growing", "high", "descriptive"),
    ])
    recs = overrides.get("recommendations", [
        _rec("REC-001", "Double down on growth", "Strong revenue signal",
             "+15% expected"),
    ])
    evidence = overrides.get("evidence", [_evidence("EV-001"),
                                          _evidence("EV-009", "correlation")])
    charts = overrides.get("charts", [])
    ctx = overrides.get("business_context", {
        "file_name": "sales.csv",
        "goal_summary": "Understand revenue drivers",
        "context_confidence": 0.8,
        "generic_mode": False,
        "answers": {"What is the goal?": "Revenue analysis"},
    })
    profile = overrides.get("data_profile", {
        "row_count": 200, "column_count": 10,
    })
    understanding = overrides.get("understanding", {
        "domain": "retail", "entities": ["product", "category"],
        "columns": [{"name": "revenue", "role": "measure"},
                     {"name": "category", "role": "dimension"}],
        "limitations": ["No customer-level data"],
    })
    dq = overrides.get("dq_report", {
        "status": "passed", "duplicates": 0,
        "invalid": {}, "missingness": {"overall_rate": 0.01},
        "issues": [],
    })

    (run_dir / "outputs" / "kpis.json").write_text(
        json.dumps({"kpis": kpis}), encoding="utf-8")
    (run_dir / "outputs" / "statistical_results.json").write_text(
        json.dumps({"results": stats}), encoding="utf-8")
    (run_dir / "outputs" / "insights.json").write_text(
        json.dumps({"insights": insights, "recommendations": recs,
                     "warnings": []}), encoding="utf-8")
    (run_dir / "outputs" / "evidence_registry.json").write_text(
        json.dumps(evidence), encoding="utf-8")
    (run_dir / "metadata" / "chart_metadata.json").write_text(
        json.dumps({"charts": charts, "charts_truncated": False}),
        encoding="utf-8")
    (run_dir / "knowledge" / "business_context.json").write_text(
        json.dumps(ctx), encoding="utf-8")
    (run_dir / "metadata" / "data_profile.json").write_text(
        json.dumps(profile), encoding="utf-8")
    (run_dir / "metadata" / "dataset_understanding.json").write_text(
        json.dumps(understanding), encoding="utf-8")
    (run_dir / "metadata" / "data_quality_report.json").write_text(
        json.dumps(dq), encoding="utf-8")
    (run_dir / "metadata" / "cleaning_result.json").write_text(
        json.dumps({"attempt": 1, "rows_before": 202, "rows_after": 200,
                     "status": "passed"}),
        encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# load_artifacts
# ---------------------------------------------------------------------------


def test_load_artifacts_complete(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    arts = load_artifacts(run_dir)
    assert isinstance(arts, dict)
    for key in ("kpis", "stats", "insights", "recommendations",
                "evidence", "charts", "business_context", "data_profile",
                "understanding", "dq_report", "cleaning_result"):
        assert key in arts, f"Missing key: {key}"
    assert len(arts["kpis"]) == 1
    assert arts["kpis"][0]["kpi_id"] == "KPI-001"


def test_load_artifacts_missing_files(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_run"
    empty_dir.mkdir()
    arts = load_artifacts(empty_dir)
    assert arts["kpis"] == []
    assert arts["stats"] == []
    assert arts["insights"] == []
    assert arts["business_context"] == {}


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def test_render_kpis_empty() -> None:
    html = render_kpis([])
    assert "No KPIs" in html


def test_render_kpis_values() -> None:
    kpis = [_kpi("KPI-001", "Revenue", 1234567.89, "EV-001")]
    html = render_kpis(kpis)
    assert "1,234,568" in html
    assert "Revenue" in html
    assert '<div class="card metric' in html


def test_render_kpis_escapes_html() -> None:
    kpis = [_kpi("KPI-001", '<script>alert("xss")</script>', 100, "EV-001")]
    html = render_kpis(kpis)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_render_stats_table() -> None:
    stats = [_stat("ST-001", "pearson", ["revenue", "quantity"],
                   0.85, 0.001, "EV-009")]
    html = render_stats(stats)
    assert "pearson" in html
    assert "0.8500" in html
    assert "0.0010" in html
    assert "EV-009" in html
    assert "<table" in html


def test_render_stats_empty() -> None:
    html = render_stats([])
    assert "No statistical tests" in html


def test_render_insights_confidence() -> None:
    insights = [
        _insight("INS-001", "High insight", "high", "descriptive"),
        _insight("INS-002", "Low insight", "low", "comparative"),
    ]
    html = render_insights(insights)
    assert "c-high" in html
    assert "c-low" in html
    assert "High insight" in html
    assert "Low insight" in html


def test_render_insights_empty() -> None:
    html = render_insights([])
    assert "No insights" in html


def test_render_recommendations() -> None:
    recs = [_rec("REC-001", "Do something", "Because X", "+10%")]
    html = render_recommendations(recs)
    assert "Do something" in html
    assert "Because X" in html
    assert "+10%" in html


def test_render_evidence() -> None:
    evidence = [_evidence("EV-001", "sum", ["revenue"])]
    html = render_evidence(evidence)
    assert "EV-001" in html
    assert "sum" in html
    assert "revenue" in html


def test_render_charts_empty() -> None:
    html = render_charts([], Path("."))
    assert "No charts" in html


def test_render_business_context() -> None:
    ctx = {"file_name": "test.csv", "goal_summary": "Find patterns",
           "context_confidence": 0.9, "generic_mode": False,
           "answers": {"Q1": "A1"}}
    html = render_business_context(ctx)
    assert "test.csv" in html
    assert "Find patterns" in html
    assert "Q1" in html
    assert "A1" in html


def test_render_dq_summary() -> None:
    dq = {"status": "needs_repair", "duplicates": 2,
          "invalid": {"revenue": ["negative"]},
          "missingness": {"overall_rate": 0.05},
          "issues": [{"severity": "high", "message": "bad data"}]}
    html = render_dq_summary(dq)
    assert "needs_repair" in html
    assert "2" in html
    assert "revenue" in html
    assert "negative" in html


def test_render_overview() -> None:
    profile = {"row_count": 500, "column_count": 12}
    understanding = {"domain": "finance", "entities": ["account"],
                     "columns": [{"name": "amount", "role": "measure"}]}
    html = render_overview(profile, understanding)
    assert "500" in html
    assert "12" in html
    assert "finance" in html
    assert "amount" in html


def test_render_limitations_low_confidence() -> None:
    ctx = {"context_confidence": 0.3}
    html = render_limitations({}, ctx)
    assert "Low business context" in html


def test_render_limitations_from_understanding() -> None:
    understanding = {"limitations": ["No temporal data"]}
    html = render_limitations(understanding, {})
    assert "No temporal data" in html


# ---------------------------------------------------------------------------
# XSS defense in depth
# ---------------------------------------------------------------------------


def test_xss_in_kpi_values() -> None:
    kpis = [_kpi("KPI-001", '<img src=x onerror=alert(1)>', 100, "EV-001")]
    html = render_kpis(kpis)
    assert "<img src=x" not in html
    assert "&lt;img" in html


def test_xss_in_insight_title() -> None:
    insights = [_insight("INS-001",
                         '<script>document.cookie</script>', "high",
                         "descriptive")]
    html = render_insights(insights)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# Full render (end-to-end)
# ---------------------------------------------------------------------------


def test_render_report_minimal(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    html = render_report(run_dir, template_path=MINIMAL_TPL)
    assert isinstance(html, str)
    assert len(html) > 100
    for section_id in ("s1", "s5", "s6", "s8", "s9", "s11"):
        assert f'id="{section_id}"' in html, f"Missing section {section_id}"
    assert "run_test" in html


def test_render_report_exec_summary(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    summary = "Revenue grew 27% driven by basket size."
    html = render_report(run_dir, exec_summary=summary,
                         template_path=MINIMAL_TPL)
    assert summary in html


def test_render_report_empty_run(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "outputs").mkdir()
    (empty_dir / "metadata").mkdir()
    (empty_dir / "knowledge").mkdir()
    html = render_report(empty_dir, template_path=MINIMAL_TPL)
    assert isinstance(html, str)
    assert "No KPIs" in html


# ---------------------------------------------------------------------------
# save_report / save_report_result
# ---------------------------------------------------------------------------


def test_save_report_creates_file(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    path = save_report(run_dir, "<html>test</html>")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "<html>test</html>"


def test_save_report_result_json(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    save_report_result(run_dir, "rendered", str(run_dir / "report.html"),
                       "en")
    result_path = run_dir / "metadata" / "report_result.json"
    assert result_path.exists()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["status"] == "rendered"
    assert data["locale"] == "en"
    assert isinstance(data["sections"], list)
