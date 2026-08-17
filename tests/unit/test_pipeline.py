"""Tests for crew/flows.py (flow helpers) and crew/crew.py (pipeline wiring)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from crew.flows import (
    build_verdict,
    check_caps,
    check_cleaning_retry,
    check_dq_gate,
    write_verdict_file,
)


# =========================================================================
# check_dq_gate
# =========================================================================


class TestCheckDqGate:
    def test_passed(self):
        assert check_dq_gate({"status": "passed"}) == "pass"

    def test_needs_repair(self):
        assert check_dq_gate({"status": "needs_repair"}) == "repair"

    def test_missing_status_defaults_to_repair(self):
        assert check_dq_gate({}) == "repair"

    def test_failed_status_is_repair(self):
        assert check_dq_gate({"status": "failed"}) == "repair"


# =========================================================================
# check_cleaning_retry
# =========================================================================


class TestCheckCleaningRetry:
    def test_recheck_passed(self):
        assert check_cleaning_retry({"status": "passed"}, attempt=1, max_rechecks=3) is True

    def test_recheck_failed_within_budget(self):
        assert check_cleaning_retry({"status": "needs_repair"}, attempt=1, max_rechecks=3) is True

    def test_recheck_failed_at_limit(self):
        assert check_cleaning_retry({"status": "needs_repair"}, attempt=3, max_rechecks=3) is False

    def test_recheck_failed_beyond_limit(self):
        assert check_cleaning_retry({"status": "needs_repair"}, attempt=4, max_rechecks=3) is False

    def test_first_attempt_can_pass(self):
        assert check_cleaning_retry({"status": "passed"}, attempt=1, max_rechecks=1) is True


# =========================================================================
# check_caps
# =========================================================================


class TestCheckCaps:
    def _make_log(self, cost: float = 0.0) -> MagicMock:
        log = MagicMock()
        log.cost_usd = cost
        return log

    def test_no_trip(self):
        cfg = {"llm": {"max_cost_usd": 5.0}, "limits": {"max_run_seconds": 1800}}
        result = check_caps(self._make_log(0.0), time.time(), cfg)
        assert result is None

    def test_cost_tripped(self):
        cfg = {"llm": {"max_cost_usd": 5.0}, "limits": {"max_run_seconds": 1800}}
        result = check_caps(self._make_log(5.0), time.time(), cfg)
        assert result == "cost_limit_exceeded"

    def test_cost_over_tripped(self):
        cfg = {"llm": {"max_cost_usd": 5.0}, "limits": {"max_run_seconds": 1800}}
        result = check_caps(self._make_log(10.0), time.time(), cfg)
        assert result == "cost_limit_exceeded"

    def test_runtime_tripped(self):
        cfg = {"llm": {"max_cost_usd": 5.0}, "limits": {"max_run_seconds": 1}}
        result = check_caps(self._make_log(0.0), time.time() - 2, cfg)
        assert result == "run_time_limit_exceeded"

    def test_defaults_when_missing(self):
        log = self._make_log(0.0)
        result = check_caps(log, time.time(), {})
        assert result is None


# =========================================================================
# build_verdict
# =========================================================================


class TestBuildVerdict:
    def test_approved(self):
        qa = {"verdict": "APPROVED", "score": 100.0, "critical_count": 0, "warning_count": 0}
        v = build_verdict(qa)
        assert v["verdict"] == "APPROVED"
        assert v["score"] == 100.0
        assert v["reason_codes"] == []

    def test_approved_with_warnings(self):
        qa = {"verdict": "APPROVED_WITH_WARNINGS", "score": 97.5, "critical_count": 0, "warning_count": 1}
        v = build_verdict(qa)
        assert v["verdict"] == "APPROVED_WITH_WARNINGS"

    def test_needs_revision_from_qa(self):
        qa = {"verdict": "NEEDS_REVISION", "score": 50.0, "critical_count": 2, "warning_count": 3}
        v = build_verdict(qa)
        assert v["verdict"] == "NEEDS_REVISION"
        assert v["critical"] == 2
        assert v["warnings"] == 3

    def test_fallback_forces_needs_revision(self):
        qa = {"verdict": "APPROVED", "score": 100.0}
        v = build_verdict(qa, reason_codes=["cost_limit_exceeded"])
        assert v["verdict"] == "NEEDS_REVISION"
        assert v["reason_codes"] == ["cost_limit_exceeded"]

    def test_multiple_fallback_codes(self):
        qa = {"verdict": "APPROVED", "score": 100.0}
        codes = ["cost_limit_exceeded", "cleaning_retry_limit_exceeded"]
        v = build_verdict(qa, reason_codes=codes)
        assert v["verdict"] == "NEEDS_REVISION"
        assert v["reason_codes"] == codes

    def test_empty_qa_defaults(self):
        v = build_verdict({})
        assert v["verdict"] == "NEEDS_REVISION"
        assert v["score"] == 0.0


# =========================================================================
# write_verdict_file
# =========================================================================


class TestWriteVerdictFile:
    def test_writes_json(self, tmp_path):
        payload = {"verdict": "APPROVED", "score": 100.0, "critical": 0, "warnings": 0, "reason_codes": []}
        path = write_verdict_file(tmp_path, payload)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["verdict"] == "APPROVED"

    def test_creates_metadata_dir(self, tmp_path):
        payload = {"verdict": "APPROVED", "score": 100.0, "critical": 0, "warnings": 0, "reason_codes": []}
        path = write_verdict_file(tmp_path, payload)
        assert path.parent.name == "metadata"
        assert path.parent.is_dir()


# =========================================================================
# Pipeline (mocked agents)
# =========================================================================


def _ok_result(stage: str) -> Dict[str, Any]:
    """Return a minimal 'passed' result for a stage."""
    base = {"stage": stage, "status": "passed", "errors": [], "log_path": "logs/run.jsonl"}
    extras = {
        "ingestion": {"run_id": "run_test_1", "file_name": "test.csv", "file_hash": "sha256:abc",
                       "row_count": 100, "column_count": 5, "pii_columns": [], "sheet_used": None,
                       "generic_mode": True, "context_confidence": 0.0,
                       "extracted_path": "data/extracted/test.csv",
                       "data_profile_path": "metadata/data_profile.json",
                       "business_context_path": "knowledge/business_context.json"},
        "understanding": {"run_id": "run_test_1", "detected_domain": "generic",
                           "domain_confidence": 0.0, "kpi_count": 3,
                           "dataset_understanding_path": "metadata/dataset_understanding.json",
                           "analysis_plan_path": "metadata/analysis_plan.json"},
        "data_quality": {"run_id": "run_test_1", "missingness_rate": 0.0,
                          "missingness_assessment": "none", "duplicates": 0,
                          "invalid_columns": [], "repair_applied": False,
                          "data_quality_report_path": "metadata/data_quality_report.json",
                          "repair_log_path": "metadata/repair_log.json"},
        "cleaning": {"run_id": "run_test_1", "final_dq_status": "passed",
                      "attempt": 1, "rows_before": 100, "rows_after": 100,
                      "duplicates_removed": 0, "flags_created": [], "type_casts": {},
                      "outliers": {},
                      "cleaned_data_path": "data/processed/cleaned_data.csv",
                      "cleaning_result_path": "metadata/cleaning_result.json"},
        "analysis": {"run_id": "run_test_1", "kpi_count": 3,
                      "statistical_test_count": 6, "chart_count": 2,
                      "charts_truncated": False, "evidence_count": 10,
                      "kpis_path": "outputs/kpis.json",
                      "statistical_results_path": "outputs/statistical_results.json",
                      "chart_metadata_path": "metadata/chart_metadata.json",
                      "evidence_registry_path": "outputs/evidence_registry.json"},
        "insights": {"run_id": "run_test_1", "insight_count": 3,
                      "recommendation_count": 3, "warnings": [],
                      "insights_path": "outputs/insights.json"},
        "report": {"run_id": "run_test_1", "report_path": "report.html",
                    "exec_summary_length": 200,
                    "sections": ["executive_summary", "kpis", "stats"]},
        "qa": {"run_id": "run_test_1", "verdict": "APPROVED", "score": 100.0,
                "critical_count": 0, "warning_count": 0, "reason_codes": [],
                "qa_verdict_path": "metadata/qa_verdict.json"},
    }
    base.update(extras.get(stage, {}))
    return base


def _fail_result(stage: str, error: str = "stage failed") -> Dict[str, Any]:
    return {"stage": stage, "status": "failed", "error": error, "errors": [error], "log_path": ""}


class TestPipeline:
    """Tests for crew.crew.run_pipeline with mocked agents."""

    def _mock_modules(self, monkeypatch, overrides: Dict[str, Any] | None = None):
        """Set up mock agent modules. overrides maps stage -> custom result function."""
        overrides = overrides or {}
        results = {}

        def _make_runner(stage: str):
            fn = overrides.get(stage)
            if fn:
                return lambda *a, **kw: fn(stage)
            return lambda *a, **kw: _ok_result(stage)

        monkeypatch.setattr("agents.ingestion_agent.run_ingestion", _make_runner("ingestion"))
        monkeypatch.setattr("agents.understanding_agent.run_understanding", _make_runner("understanding"))
        monkeypatch.setattr("agents.data_quality.run_data_quality", _make_runner("data_quality"))
        monkeypatch.setattr("agents.cleaning_agent.run_cleaning", _make_runner("cleaning"))
        monkeypatch.setattr("agents.analysis.run_analysis", _make_runner("analysis"))
        monkeypatch.setattr("agents.insight_agent.run_insights", _make_runner("insights"))
        monkeypatch.setattr("agents.report_agent.run_report", _make_runner("report"))
        monkeypatch.setattr("agents.qa_agent.run_qa", _make_runner("qa"))

    def test_happy_path(self, tmp_path, monkeypatch):
        """All stages pass → APPROVED."""
        self._mock_modules(monkeypatch)

        from crew.crew import run_pipeline
        result = run_pipeline(
            "tests/fixtures/sales_demo.csv",
            use_crew=False,
            cfg={"pipeline_version": "4.3.0", "llm": {"max_cost_usd": 999}, "limits": {"max_run_seconds": 9999, "cleaning_max_rechecks": 3}},
        )
        assert result["verdict"] == "APPROVED"
        assert result["score"] == 100.0
        assert result["report_path"] is not None
        assert "ingestion" in result["stage_results"]
        assert "qa" in result["stage_results"]

    def test_master_manifest_written(self, tmp_path, monkeypatch):
        """master_manifest.json is created."""
        self._mock_modules(monkeypatch)

        from crew.crew import run_pipeline
        result = run_pipeline(
            "tests/fixtures/sales_demo.csv",
            use_crew=False,
            cfg={"pipeline_version": "4.3.0", "llm": {"max_cost_usd": 999}, "limits": {"max_run_seconds": 9999, "cleaning_max_rechecks": 3}},
        )
        manifest = Path(result["run_dir"]) / "master_manifest.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["verdict"] == "APPROVED"
        assert data["pipeline_version"] == "4.3.0"

    def test_qa_verdict_file_written(self, tmp_path, monkeypatch):
        """metadata/qa_verdict.json is created."""
        self._mock_modules(monkeypatch)

        from crew.crew import run_pipeline
        result = run_pipeline(
            "tests/fixtures/sales_demo.csv",
            use_crew=False,
            cfg={"pipeline_version": "4.3.0", "llm": {"max_cost_usd": 999}, "limits": {"max_run_seconds": 9999, "cleaning_max_rechecks": 3}},
        )
        vpath = Path(result["run_dir"]) / "metadata" / "qa_verdict.json"
        assert vpath.exists()
        data = json.loads(vpath.read_text(encoding="utf-8"))
        assert data["verdict"] == "APPROVED"

    def test_stage_failure_aborts(self, tmp_path, monkeypatch):
        """Stage 5 failure → pipeline aborts with NEEDS_REVISION."""
        self._mock_modules(monkeypatch, overrides={
            "analysis": lambda s: _fail_result("analysis", "computation error"),
        })

        from crew.crew import run_pipeline
        result = run_pipeline(
            "tests/fixtures/sales_demo.csv",
            use_crew=False,
            cfg={"pipeline_version": "4.3.0", "llm": {"max_cost_usd": 999}, "limits": {"max_run_seconds": 9999, "cleaning_max_rechecks": 3}},
        )
        assert result["verdict"] == "NEEDS_REVISION"
        assert "analysis" in result["stage_results"]
        # Later stages should not be present
        assert "insights" not in result["stage_results"]

    def test_ingestion_failure_aborts(self, tmp_path, monkeypatch):
        """Stage 1 failure → pipeline aborts immediately."""
        self._mock_modules(monkeypatch, overrides={
            "ingestion": lambda s: _fail_result("ingestion", "bad file"),
        })

        from crew.crew import run_pipeline
        result = run_pipeline(
            "tests/fixtures/sales_demo.csv",
            use_crew=False,
            cfg={"pipeline_version": "4.3.0", "llm": {"max_cost_usd": 999}, "limits": {"max_run_seconds": 9999, "cleaning_max_rechecks": 3}},
        )
        assert result["verdict"] == "NEEDS_REVISION"
        # Only ingestion should be in stage_results
        assert list(result["stage_results"].keys()) == ["ingestion"]

    def test_cleaning_retry_cap(self, tmp_path, monkeypatch):
        """DQ recheck keeps failing → cleaning_retry_limit_exceeded."""
        dq_call_count = {"n": 0}

        def _dq_handler(stage):
            dq_call_count["n"] += 1
            # First call = initial DQ (pass). Subsequent = rechecks (fail).
            if dq_call_count["n"] == 1:
                return _ok_result("data_quality")
            return _fail_result("data_quality", "still broken")

        self._mock_modules(monkeypatch, overrides={
            "data_quality": _dq_handler,
        })

        from crew.crew import run_pipeline
        result = run_pipeline(
            "tests/fixtures/sales_demo.csv",
            use_crew=False,
            cfg={"pipeline_version": "4.3.0", "llm": {"max_cost_usd": 999}, "limits": {"max_run_seconds": 9999, "cleaning_max_rechecks": 3}},
        )
        assert result["verdict"] == "NEEDS_REVISION"
        # Should have run DQ 4 times: 1 initial + 3 rechecks (all fail)
        assert dq_call_count["n"] == 4

    def test_cost_cap_trips(self, tmp_path, monkeypatch):
        """Cost cap exceeded → NEEDS_REVISION with reason code."""
        self._mock_modules(monkeypatch)

        from crew.crew import run_pipeline
        # Set cost cap to 0 so it trips immediately on the first check
        result = run_pipeline(
            "tests/fixtures/sales_demo.csv",
            use_crew=False,
            cfg={"pipeline_version": "4.3.0", "llm": {"max_cost_usd": 0.0}, "limits": {"max_run_seconds": 9999, "cleaning_max_rechecks": 3}},
        )
        assert result["verdict"] == "NEEDS_REVISION"

    def test_dq_repair_path(self, tmp_path, monkeypatch):
        """DQ returns needs_repair → cleaning runs → recheck passes."""
        dq_first = {"n": 0}

        def _dq_handler(stage):
            dq_first["n"] += 1
            if dq_first["n"] == 1:
                return _ok_result("data_quality")  # initial: passed
            return _ok_result("data_quality")  # recheck: also passes

        self._mock_modules(monkeypatch, overrides={
            "data_quality": _dq_handler,
        })

        from crew.crew import run_pipeline
        result = run_pipeline(
            "tests/fixtures/sales_demo.csv",
            use_crew=False,
            cfg={"pipeline_version": "4.3.0", "llm": {"max_cost_usd": 999}, "limits": {"max_run_seconds": 9999, "cleaning_max_rechecks": 3}},
        )
        assert result["verdict"] == "APPROVED"
        # DQ should have been called twice: initial + 1 recheck that passed
        assert dq_first["n"] == 2

    def test_duration_positive(self, tmp_path, monkeypatch):
        """duration_s should be positive."""
        self._mock_modules(monkeypatch)

        from crew.crew import run_pipeline
        result = run_pipeline(
            "tests/fixtures/sales_demo.csv",
            use_crew=False,
            cfg={"pipeline_version": "4.3.0", "llm": {"max_cost_usd": 999}, "limits": {"max_run_seconds": 9999, "cleaning_max_rechecks": 3}},
        )
        assert result["duration_s"] >= 0

    def test_run_id_and_dir_set(self, tmp_path, monkeypatch):
        """run_id and run_dir should be non-empty strings."""
        self._mock_modules(monkeypatch)

        from crew.crew import run_pipeline
        result = run_pipeline(
            "tests/fixtures/sales_demo.csv",
            use_crew=False,
            cfg={"pipeline_version": "4.3.0", "llm": {"max_cost_usd": 999}, "limits": {"max_run_seconds": 9999, "cleaning_max_rechecks": 3}},
        )
        assert result["run_id"].startswith("run_")
        assert Path(result["run_dir"]).is_dir()
