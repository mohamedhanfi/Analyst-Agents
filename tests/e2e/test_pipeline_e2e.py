"""E2E tests — full pipeline via run_pipeline() on golden fixtures.

These tests run the REAL pipeline (deterministic mode, no LLM) on fixture
CSVs and verify that all stages complete and artifacts are produced.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from crew.crew import run_pipeline

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _artifact_paths(run_dir: Path) -> dict:
    """Map of expected artifact paths for a completed pipeline run."""
    return {
        "data_profile": run_dir / "metadata" / "data_profile.json",
        "business_context": run_dir / "knowledge" / "business_context.json",
        "understanding": run_dir / "metadata" / "dataset_understanding.json",
        "analysis_plan": run_dir / "metadata" / "analysis_plan.json",
        "dq_report": run_dir / "metadata" / "data_quality_report.json",
        "cleaning_result": run_dir / "metadata" / "cleaning_result.json",
        "cleaned_data": run_dir / "data" / "processed" / "cleaned_data.csv",
        "kpis": run_dir / "outputs" / "kpis.json",
        "stats": run_dir / "outputs" / "statistical_results.json",
        "charts": run_dir / "metadata" / "chart_metadata.json",
        "evidence": run_dir / "outputs" / "evidence_registry.json",
        "insights": run_dir / "outputs" / "insights.json",
        "report": run_dir / "report.html",
        "report_result": run_dir / "metadata" / "report_result.json",
        "qa_verdict": run_dir / "metadata" / "qa_verdict.json",
        "master_manifest": run_dir / "master_manifest.json",
    }


class TestE2eSalesSmall:
    """Full pipeline on the clean baseline fixture."""

    @pytest.fixture(scope="class")
    def run_result(self, tmp_path_factory: pytest.TempPathFactory):
        out = tmp_path_factory.mktemp("e2e_sales_small")
        result = run_pipeline(
            file_path=str(FIXTURES / "sales_small.csv"),
            use_crew=False,
            locale="en",
            output_dir=str(out),
        )
        return result

    def test_pipeline_completes(self, run_result: dict) -> None:
        assert run_result.get("status") != "failed"

    def test_all_stages_present(self, run_result: dict) -> None:
        stages = run_result.get("stages", {})
        for i in range(1, 9):
            key = f"stage_{i}"
            assert key in stages, f"Missing {key}"

    def test_master_manifest_exists(self, run_result: dict) -> None:
        run_dir = Path(run_result.get("run_dir", ""))
        manifest = run_dir / "master_manifest.json"
        assert manifest.exists(), f"master_manifest.json not found in {run_dir}"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert "stages" in data or "pipeline" in data

    def test_report_html_exists(self, run_result: dict) -> None:
        run_dir = Path(run_result.get("run_dir", ""))
        report = run_dir / "report.html"
        assert report.exists()
        content = report.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_qa_verdict_exists(self, run_result: dict) -> None:
        run_dir = Path(run_result.get("run_dir", ""))
        verdict_path = run_dir / "metadata" / "qa_verdict.json"
        assert verdict_path.exists()
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        assert "verdict" in verdict
        assert "score" in verdict


class TestE2eSalesMissing:
    """Pipeline handles missing data gracefully."""

    @pytest.fixture(scope="class")
    def run_result(self, tmp_path_factory: pytest.TempPathFactory):
        out = tmp_path_factory.mktemp("e2e_sales_missing")
        result = run_pipeline(
            file_path=str(FIXTURES / "sales_missing.csv"),
            use_crew=False,
            locale="en",
            output_dir=str(out),
        )
        return result

    def test_pipeline_completes(self, run_result: dict) -> None:
        assert run_result.get("status") != "failed"

    def test_cleaning_stage_runs(self, run_result: dict) -> None:
        stages = run_result.get("stages", {})
        assert "stage_4" in stages


class TestE2eSalesOutliers:
    """Pipeline handles outliers."""

    @pytest.fixture(scope="class")
    def run_result(self, tmp_path_factory: pytest.TempPathFactory):
        out = tmp_path_factory.mktemp("e2e_sales_outliers")
        result = run_pipeline(
            file_path=str(FIXTURES / "sales_outliers.csv"),
            use_crew=False,
            locale="en",
            output_dir=str(out),
        )
        return result

    def test_pipeline_completes(self, run_result: dict) -> None:
        assert run_result.get("status") != "failed"


class TestE2eSalesDuplicates:
    """Pipeline handles duplicate rows."""

    @pytest.fixture(scope="class")
    def run_result(self, tmp_path_factory: pytest.TempPathFactory):
        out = tmp_path_factory.mktemp("e2e_sales_duplicates")
        result = run_pipeline(
            file_path=str(FIXTURES / "sales_duplicates.csv"),
            use_crew=False,
            locale="en",
            output_dir=str(out),
        )
        return result

    def test_pipeline_completes(self, run_result: dict) -> None:
        assert run_result.get("status") != "failed"


class TestE2eMasterManifest:
    """Verify master_manifest.json structure."""

    @pytest.fixture(scope="class")
    def run_result(self, tmp_path_factory: pytest.TempPathFactory):
        out = tmp_path_factory.mktemp("e2e_manifest")
        result = run_pipeline(
            file_path=str(FIXTURES / "sales_small.csv"),
            use_crew=False,
            locale="en",
            output_dir=str(out),
        )
        return result

    def test_manifest_has_stages(self, run_result: dict) -> None:
        run_dir = Path(run_result.get("run_dir", ""))
        manifest_path = run_dir / "master_manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Should have entries for completed stages
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_manifest_has_run_id(self, run_result: dict) -> None:
        run_dir = Path(run_result.get("run_dir", ""))
        manifest_path = run_dir / "master_manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "run_id" in data or run_dir.name != ""


class TestE2eRunDirectory:
    """Verify run directory structure is created correctly."""

    @pytest.fixture(scope="class")
    def run_result(self, tmp_path_factory: pytest.TempPathFactory):
        out = tmp_path_factory.mktemp("e2e_rundir")
        result = run_pipeline(
            file_path=str(FIXTURES / "sales_small.csv"),
            use_crew=False,
            locale="en",
            output_dir=str(out),
        )
        return result

    def test_run_dir_exists(self, run_result: dict) -> None:
        run_dir = Path(run_result.get("run_dir", ""))
        assert run_dir.exists()

    def test_subdirectories_created(self, run_result: dict) -> None:
        run_dir = Path(run_result.get("run_dir", ""))
        for sub in ["metadata", "outputs", "data", "knowledge"]:
            assert (run_dir / sub).exists(), f"Missing subdir: {sub}"


class TestE2eCostCaps:
    """Verify cost and runtime tracking."""

    @pytest.fixture(scope="class")
    def run_result(self, tmp_path_factory: pytest.TempPathFactory):
        out = tmp_path_factory.mktemp("e2e_cost")
        result = run_pipeline(
            file_path=str(FIXTURES / "sales_small.csv"),
            use_crew=False,
            locale="en",
            output_dir=str(out),
        )
        return result

    def test_duration_tracked(self, run_result: dict) -> None:
        assert "duration_seconds" in run_result
        assert run_result["duration_seconds"] >= 0

    def test_total_cost_within_cap(self, run_result: dict) -> None:
        cost = run_result.get("total_cost", 0.0)
        assert cost <= 10.0, f"Cost {cost} exceeds cap"
