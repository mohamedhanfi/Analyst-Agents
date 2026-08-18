"""Unit tests for agents/understanding_agent (deterministic path + finalize).

No API key required: use_crew=False runs role rules + domain heuristic +
default whitelist plan through the same orchestration as the crew path, and
_finalize_understanding is exercised with fake crew outputs.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents.understanding_agent import (
    _finalize_understanding,
    run_understanding,
)
from shared.dsl_validator import validate_plan
from shared.schemas import BusinessContext, DataProfile
from shared.utils import load_config


@pytest.fixture
def cfg():
    return load_config(require_key=False)


def make_profile() -> DataProfile:
    return DataProfile(
        file_name="sales.csv", file_hash="sha256:abc",
        row_count=100, column_count=5,
        columns=["order_id", "date", "product", "revenue", "quantity"],
        column_types={"order_id": "int64", "date": "datetime64[ns]",
                      "product": "object", "revenue": "float64",
                      "quantity": "int64"},
        nunique={"order_id": 100, "date": 60, "product": 5,
                 "revenue": 87, "quantity": 30},
        missing_values={"revenue": 2},
        sample=[{"order_id": 1, "date": "2024-01-01", "product": "A",
                 "revenue": 10.5, "quantity": 2}],
        validation_status="passed",
    )


def make_context(generic_mode: bool = False, answers: dict | None = None,
                 goal: str = "") -> BusinessContext:
    return BusinessContext(
        file_name="sales.csv", business_questions=[],
        answers=answers or {}, goal_summary=goal,
        context_confidence=0.0, generic_mode=generic_mode)


def make_run_dir(tmp_path, profile=None, context=None):
    run_dir = tmp_path / "run1"
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "knowledge").mkdir(parents=True)
    profile = profile or make_profile()
    context = context or make_context()
    (run_dir / "metadata" / "data_profile.json").write_text(
        profile.model_dump_json(), encoding="utf-8")
    (run_dir / "knowledge" / "business_context.json").write_text(
        context.model_dump_json(), encoding="utf-8")
    return run_dir


def read_json(run_dir, name):
    return json.loads((run_dir / "metadata" / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Deterministic run
# ---------------------------------------------------------------------------


def test_deterministic_produces_artifacts(tmp_path, cfg):
    run_dir = make_run_dir(tmp_path)
    s = run_understanding(run_dir, cfg=cfg)
    assert s["status"] == "passed"
    assert s["kpi_count"] == 7  # 2 measures x2 + count + growth + correlation

    understanding = read_json(run_dir, "dataset_understanding.json")
    assert understanding["identifiers"] == ["order_id"]
    assert understanding["temporal_columns"] == ["date"]
    assert understanding["measures"] == ["revenue", "quantity"]
    assert understanding["dimensions"] == ["product"]
    assert understanding["columns"][3]["nullable"] is True

    plan = read_json(run_dir, "analysis_plan.json")
    assert validate_plan(plan) == []
    assert len(plan["candidate_kpis"]) == 7


def test_domain_detected_from_context(tmp_path, cfg):
    ctx = make_context(answers={"domain": "sales and revenue tracking"})
    run_dir = make_run_dir(tmp_path, context=ctx)
    s = run_understanding(run_dir, cfg=cfg)
    assert s["detected_domain"] == "sales"
    assert s["domain_confidence"] > 0.0


def test_build_understanding_tasks_accepts_str_run_dir(tmp_path, cfg):
    """Regression: crew mode crashed with ``str / str`` — the tasks builder
    stringified run_dir then loaded profile/context with the string."""
    run_dir = make_run_dir(tmp_path)
    from agents.understanding_agent import (
        build_understanding_agent, build_understanding_tasks)
    agent = build_understanding_agent(cfg)
    tasks = build_understanding_tasks(agent, str(run_dir), cfg)
    assert len(tasks) == 3


def test_generic_mode_domain(tmp_path, cfg):
    run_dir = make_run_dir(tmp_path, context=make_context(generic_mode=True))
    s = run_understanding(run_dir, cfg=cfg)
    assert s["detected_domain"] == "generic"
    assert s["domain_confidence"] == 0.0


def test_missing_profile_fails_gracefully(tmp_path, cfg):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    s = run_understanding(run_dir, cfg=cfg)
    assert s["status"] == "failed"
    assert "ingestion" in s["error"]


def test_logs_stage_lifecycle(tmp_path, cfg):
    run_dir = make_run_dir(tmp_path)
    s = run_understanding(run_dir, cfg=cfg)
    lines = [json.loads(l) for l in
             open(s["log_path"], encoding="utf-8").read().splitlines()]
    kinds = [l["kind"] for l in lines]
    assert kinds[0] == "stage_start"
    assert kinds[-1] == "stage_end"
    assert "tool_call" in kinds
    assert all(l.get("run_id") for l in lines)


# ---------------------------------------------------------------------------
# _finalize_understanding with fake crew outputs
# ---------------------------------------------------------------------------


def fake_result(*raws):
    return SimpleNamespace(tasks_output=[
        SimpleNamespace(raw=r) for r in raws])


def test_finalize_uses_llm_outputs(tmp_path):
    run_dir = make_run_dir(tmp_path)
    result = fake_result(
        '[{"name": "zip_code", "role": "identifier"}]',
        '{"detected_domain": "sales", "domain_confidence": 0.8, '
        '"entities": ["Product"]}',
        json.dumps({"candidate_kpis": [
            {"kpi_id": "KPI-001", "name": "Total Revenue",
             "operation": {"function": "sum", "column": "revenue"}}],
            "statistical_tests": ["descriptive"]}),
    )
    understanding, plan, warnings = _finalize_understanding(run_dir, result)
    assert warnings == []
    assert understanding.detected_domain == "sales"
    assert understanding.domain_confidence == 0.8
    assert understanding.entities == ["Product"]
    assert len(plan.candidate_kpis) == 1
    assert validate_plan(plan) == []


def test_finalize_rejects_bad_llm_plan(tmp_path):
    run_dir = make_run_dir(tmp_path)
    result = fake_result(
        '[{"name": "product", "role": "measure"}]',
        '{"detected_domain": "sales", "domain_confidence": 0.5, '
        '"entities": []}',
        json.dumps({"candidate_kpis": [
            {"kpi_id": "KPI-BAD",
             "operation": {"function": "evil", "column": "x"}}]}),
    )
    understanding, plan, warnings = _finalize_understanding(run_dir, result)
    assert "llm_plan_partially_rejected" in warnings
    assert plan.candidate_kpis == []
    assert "KPI-BAD" in understanding.limitations[0]


def test_finalize_falls_back_on_garbage(tmp_path):
    run_dir = make_run_dir(tmp_path)
    result = fake_result("not json at all", "garbage", "")
    understanding, plan, warnings = _finalize_understanding(run_dir, result)
    assert "llm_plan_failed_default_used" in warnings
    assert "llm_domain_failed_heuristic_used" in warnings
    assert plan.candidate_kpis  # default plan is not empty
    assert validate_plan(plan) == []
