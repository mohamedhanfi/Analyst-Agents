"""Unit tests for agents/cleaning_agent.run_cleaning (deterministic path).

No API key required: stages 1-3 are run deterministically in a temp dir, then
run_cleaning exercises the §2.4 deliverables: cleaned_data.csv +
cleaned_data_attempt_<n>.csv + cleaning_result.json, the recheck loop and the
retry cap (limits.cleaning_max_rechecks -> cleaning_retry_limit_exceeded).
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

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


def _build_stage3(tmp_path, cfg, df, name="r1"):
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
    assert (run_dir / "metadata" / "data_quality_report.json").is_file()
    return run_dir


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


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_cleaning_produces_all_artifacts(tmp_path, cfg):
    run_dir = _build_stage3(tmp_path, cfg,
                            pd.DataFrame(_sales_rows()))
    s = run_cleaning(run_dir, cfg=cfg)
    assert s["status"] == "passed"
    assert s["attempt"] >= 1
    assert s["rows_before"] == 7
    assert s["rows_after"] == 7

    cleaned = (run_dir / "data" / "processed" / "cleaned_data.csv")
    assert cleaned.is_file()
    result = json.loads(
        (run_dir / "metadata" / "cleaning_result.json")
        .read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["rows_before"] == 7
    assert result["rows_after"] == 7

    attempts = list((run_dir / "data" / "processed").glob(
        "cleaned_data_attempt_*.csv"))
    assert len(attempts) == max(0, result["attempt"] - 1)


def test_cleaning_logs_stage_lifecycle(tmp_path, cfg):
    run_dir = _build_stage3(tmp_path, cfg,
                            pd.DataFrame(_sales_rows()))
    s = run_cleaning(run_dir, cfg=cfg)
    lines = [json.loads(l) for l in
             open(s["log_path"], encoding="utf-8").read().splitlines()]
    kinds = [l["kind"] for l in lines]
    assert kinds[0] == "stage_start"
    assert kinds[-1] == "stage_end"
    tools = [l.get("tool") for l in lines if l["kind"] == "tool_call"]
    assert "cleaning_strategy_tool" in tools
    assert "execute_cleaning" in tools
    assert "dq_recheck_tool" in tools
    assert all(l.get("run_id") for l in lines)


# ---------------------------------------------------------------------------
# Negative measure -> drop_negative (§2.4)
# ---------------------------------------------------------------------------


def test_cleaning_drops_negative_measure(tmp_path, cfg):
    run_dir = _build_stage3(tmp_path, cfg,
                            pd.DataFrame(_sales_rows(negatives=(2, 5))))
    s = run_cleaning(run_dir, cfg=cfg)
    assert s["status"] == "passed"
    assert s["rows_before"] == 7
    assert s["rows_after"] == 5

    cleaned = pd.read_csv(
        run_dir / "data" / "processed" / "cleaned_data.csv")
    assert (cleaned["revenue"] >= 0).all()


# ---------------------------------------------------------------------------
# Retry cap
# ---------------------------------------------------------------------------


def test_cleaning_stops_after_retry_cap(tmp_path, cfg):
    rows = _sales_rows()
    rows[0]["date"] = "2100-01-01"  # impossible temporal -> never passes
    run_dir = _build_stage3(tmp_path, cfg, pd.DataFrame(rows))
    s = run_cleaning(run_dir, cfg=cfg)
    assert s["status"] == "failed"
    assert s["final_dq_status"] == "needs_repair"
    assert s["attempt"] == 3

    result = json.loads(
        (run_dir / "metadata" / "cleaning_result.json")
        .read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["attempt"] == 3

    attempts = list((run_dir / "data" / "processed").glob(
        "cleaned_data_attempt_*.csv"))
    assert len(attempts) == 2
