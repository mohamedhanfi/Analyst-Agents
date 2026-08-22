"""Unit tests for agents/analysis.run_analysis (deterministic path).

Stages 1-4 are run deterministically in a temp dir (no API key), then
run_analysis exercises the §2.5 compute layer: kpis.json,
statistical_results.json, chart_metadata.json, evidence_registry.json — every
value/chart evidence-minted, Python-computed over all rows.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from agents.analysis import run_analysis
from agents.cleaning_agent import run_cleaning
from agents.data_quality import run_data_quality
from agents.ingestion_agent import run_ingestion
from agents.understanding_agent import run_understanding
from shared.utils import load_config


@pytest.fixture
def cfg():
    return load_config(require_key=False)


def _business_answers():
    return iter(["Track revenue", "sales", "Top products?", "Set targets"])


def _sales_rows(n=7, negatives=()):
    rows = []
    for i in range(n):
        rows.append({
            "date": f"2024-01-{i + 1:02d}",
            "product": "A" if i % 2 == 0 else "B",
            "revenue": -50.0 if i in negatives else 100.0 + i * 10,
            "quantity": i + 1,
        })
    return rows


def _build_stage4(tmp_path, cfg, df, name="r1"):
    csv = tmp_path / "sales.csv"
    df.to_csv(csv, index=False)
    provider = lambda _prompt: next(_business_answers())
    run_dir = tmp_path / name
    s1 = run_ingestion(str(csv), run_dir=run_dir, cfg=cfg,
                       answer_provider=provider)
    assert s1["status"] == "passed"
    s2 = run_understanding(run_dir, cfg=cfg)
    assert s2["status"] == "passed"
    s3 = run_data_quality(run_dir, cfg=cfg)
    assert s3["status"] in ("passed", "needs_repair")
    s4 = run_cleaning(run_dir, cfg=cfg)
    assert s4["status"] == "passed"
    return run_dir


def test_analysis_produces_all_artifacts(tmp_path, cfg):
    run_dir = _build_stage4(tmp_path, cfg, pd.DataFrame(_sales_rows()))
    s = run_analysis(run_dir, cfg=cfg)
    assert s["status"] == "passed"
    assert s["kpi_count"] > 0
    assert s["statistical_test_count"] > 0
    assert s["chart_count"] > 0
    assert s["evidence_count"] > 0
    assert isinstance(s["charts_truncated"], bool)

    assert (run_dir / "outputs" / "kpis.json").is_file()
    assert (run_dir / "outputs" / "statistical_results.json").is_file()
    assert (run_dir / "metadata" / "chart_metadata.json").is_file()
    assert (run_dir / "outputs" / "evidence_registry.json").is_file()


def test_analysis_kpis_all_carry_evidence(tmp_path, cfg):
    run_dir = _build_stage4(tmp_path, cfg, pd.DataFrame(_sales_rows()))
    run_analysis(run_dir, cfg=cfg)
    kpis = json.loads((run_dir / "outputs" / "kpis.json")
                      .read_text(encoding="utf-8"))["kpis"]
    assert len(kpis) > 0
    assert all(k["computed_by"] == "pandas" for k in kpis)
    assert all(k["evidence_id"] for k in kpis)


def test_analysis_statistical_results_evidence(tmp_path, cfg):
    run_dir = _build_stage4(tmp_path, cfg, pd.DataFrame(_sales_rows()))
    run_analysis(run_dir, cfg=cfg)
    stats = json.loads((run_dir / "outputs" / "statistical_results.json")
                       .read_text(encoding="utf-8"))["results"]
    assert len(stats) > 0
    assert all(r["evidence_id"] for r in stats)
    categories = {r["category"] for r in stats}
    assert categories <= {"descriptive", "correlation", "trend", "comparison"}


def test_analysis_chart_metadata_shape(tmp_path, cfg):
    run_dir = _build_stage4(tmp_path, cfg, pd.DataFrame(_sales_rows()))
    run_analysis(run_dir, cfg=cfg)
    payload = json.loads((run_dir / "metadata" / "chart_metadata.json")
                         .read_text(encoding="utf-8"))
    assert "charts_truncated" in payload
    assert payload["charts"] and all(
        c["evidence_id"] for c in payload["charts"])


def test_analysis_evidence_registry_has_lineage(tmp_path, cfg):
    run_dir = _build_stage4(tmp_path, cfg, pd.DataFrame(_sales_rows()))
    run_analysis(run_dir, cfg=cfg)
    entries = json.loads((run_dir / "outputs" / "evidence_registry.json")
                         .read_text(encoding="utf-8"))
    assert len(entries) > 0
    ids = {e["evidence_id"] for e in entries}
    assert len(ids) == len(entries)          # unique ids
    for entry in entries:
        source = entry["source"]
        assert isinstance(source["transformations"], list)
        assert source["file_hash"].startswith("sha256:")
        assert source["aggregation"]


def test_analysis_logs_stage_lifecycle(tmp_path, cfg):
    run_dir = _build_stage4(tmp_path, cfg, pd.DataFrame(_sales_rows()))
    s = run_analysis(run_dir, cfg=cfg)
    lines = [json.loads(l) for l in
             open(s["log_path"], encoding="utf-8").read().splitlines()]
    kinds = [l["kind"] for l in lines]
    assert kinds[0] == "stage_start"
    assert kinds[-1] == "stage_end"
    tools = [l.get("tool") for l in lines if l["kind"] == "tool_call"]
    assert "dsl_executor_tool" in tools
    assert "statistical_suite_tool" in tools
    assert "chart_planner_tool" in tools


def test_analysis_missing_cleaned_csv_fails(tmp_path, cfg):
    run_dir = _build_stage4(tmp_path, cfg, pd.DataFrame(_sales_rows()))
    for name in run_dir.glob("data/processed/cleaned_data*.csv"):
        name.unlink()
    (run_dir / "data" / "processed" / "analysis_ready.csv").unlink()
    s = run_analysis(run_dir, cfg=cfg)
    assert s["status"] == "failed"
    assert "cleaned_data.csv" in s.get("error", "")


def test_analysis_missing_plan_fails(tmp_path, cfg):
    run_dir = _build_stage4(tmp_path, cfg, pd.DataFrame(_sales_rows()))
    (run_dir / "metadata" / "analysis_plan.json").unlink()
    s = run_analysis(run_dir, cfg=cfg)
    assert s["status"] == "failed"
    assert "analysis_plan.json" in s.get("error", "")


def test_analysis_deterministic_repeatable(tmp_path, cfg):
    run_dir = _build_stage4(tmp_path, cfg, pd.DataFrame(_sales_rows()))
    run_analysis(run_dir, cfg=cfg)
    first = (run_dir / "outputs" / "kpis.json").read_text(encoding="utf-8")
    run_analysis(run_dir, cfg=cfg)
    second = (run_dir / "outputs" / "kpis.json").read_text(encoding="utf-8")
    assert json.loads(first) == json.loads(second)
