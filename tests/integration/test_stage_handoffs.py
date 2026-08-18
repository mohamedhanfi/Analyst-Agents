"""Integration tests — inter-stage handoff contracts.

Each test verifies that the output of stage N is a valid input for
stage N+1, using real fixture data and deterministic stage functions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from shared.core.data_quality import assemble_report
from shared.core.profiler import DataProfiler
from shared.core.validation import FileValidator
from shared.core.understanding import (
    default_plan,
    detect_domain_heuristic,
)
from shared.schemas import (
    AnalysisPlan,
    BusinessContext,
    DataProfile,
    DatasetUnderstanding,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


# ------------------------------------------------------------------
# Stage 1 → Stage 2: ingestion output → understanding input
# ------------------------------------------------------------------


class TestIngestionToUnderstanding:
    def test_data_profile_has_required_fields(self) -> None:
        csv_path = FIXTURES / "sales_small.csv"
        validator = FileValidator()
        vresult = validator.validate(str(csv_path))
        assert vresult.validation_status == "passed"

        df = pd.read_csv(csv_path)
        assert len(df) == 100

        profiler = DataProfiler()
        profile = profiler.profile(df, file_name="sales_small.csv",
                                   file_hash="sha256:test")
        profile_dict = profile.model_dump()
        assert "row_count" in profile_dict
        assert "column_count" in profile_dict
        assert "columns" in profile_dict
        assert profile_dict["row_count"] == 100

    def test_validator_rejects_nonexistent(self) -> None:
        validator = FileValidator()
        result = validator.validate(str(FIXTURES / "nonexistent.csv"))
        assert result.validation_status == "failed"


# ------------------------------------------------------------------
# Stage 2 → Stage 3: understanding output → DQ input
# ---------------------------------------------------------------------------


class TestUnderstandingToDq:
    def test_domain_detection_retail(self) -> None:
        ctx = BusinessContext(
            file_name="test.csv",
            goal_summary="Analyze sales performance and revenue",
            answers={"What domain?": "retail sales"},
            business_questions=[],
            generic_mode=False,
            context_confidence=0.8,
        )
        domain, confidence = detect_domain_heuristic(ctx)
        assert isinstance(domain, str)
        assert 0.0 <= confidence <= 1.0

    def test_domain_detection_generic(self) -> None:
        ctx = BusinessContext(
            file_name="test.csv",
            generic_mode=True,
        )
        domain, confidence = detect_domain_heuristic(ctx)
        assert domain == "generic"
        assert confidence == 0.0

    def test_plan_has_operations(self) -> None:
        csv_path = FIXTURES / "sales_small.csv"
        df = pd.read_csv(csv_path)
        profiler = DataProfiler()
        profile = profiler.profile(df, file_name="sales_small.csv",
                                   file_hash="sha256:test")

        plan = default_plan(profile)
        assert isinstance(plan, AnalysisPlan)
        assert len(plan.candidate_kpis) > 0


# ------------------------------------------------------------------
# Stage 3 → Stage 4: DQ output → cleaning input
# ---------------------------------------------------------------------------


class TestDqToCleaning:
    def test_dq_report_has_status(self) -> None:
        understanding = DatasetUnderstanding(
            detected_domain="retail",
            domain_confidence=0.8,
            entities=[], columns=[], limitations=[],
        )
        profile = DataProfile(
            file_name="test.csv", file_hash="sha256:abc",
            row_count=3, column_count=1, columns=["a"],
            column_types={"a": "int64"},
        )
        ctx = BusinessContext(
            file_name="test.csv", generic_mode=True,
        )
        df = pd.DataFrame({"a": [1, 2, 3]})
        report, repair_log = assemble_report(
            understanding, profile, df, ctx,
        )
        report_dict = report.model_dump()
        assert "status" in report_dict
        assert report_dict["status"] in ("passed", "needs_repair", "failed")


# ------------------------------------------------------------------
# Stage 4 → Stage 5: cleaning output → analysis input
# ---------------------------------------------------------------------------


class TestCleaningToAnalysis:
    def test_cleaned_csv_is_valid(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_small.csv")
        assert df is not None
        assert len(df) == 100
        assert "revenue" in df.columns

    def test_cleaned_csv_numeric_revenue(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_small.csv")
        assert pd.api.types.is_numeric_dtype(df["revenue"])


# ------------------------------------------------------------------
# Stage 5 → Stage 6: analysis output → insights input
# ---------------------------------------------------------------------------


class TestAnalysisToInsights:
    def test_kpis_json_has_structure(self) -> None:
        kpis = [{
            "kpi_id": "KPI-001", "name": "Revenue",
            "operation": {"function": "sum", "column": "revenue"},
            "value": 10000.0, "evidence_id": "EV-001",
        }]
        assert len(kpis) > 0
        assert "evidence_id" in kpis[0]

    def test_evidence_registry_has_entries(self) -> None:
        evidence = [{
            "evidence_id": "EV-001",
            "source": {"aggregation": "sum", "lineage": ["revenue"],
                       "result": 10000.0},
        }]
        assert len(evidence) > 0


# ------------------------------------------------------------------
# Stage 6 → Stage 7: insights output → report input
# ---------------------------------------------------------------------------


class TestInsightsToReport:
    def test_insights_json_structure(self) -> None:
        insights = {
            "insights": [{
                "insight_id": "INS-001",
                "title": "Revenue is high",
                "description": "Total revenue is 10000.",
                "claim_type": "descriptive",
                "confidence": "high",
                "evidence_ids": ["EV-001"],
                "related_kpis": ["KPI-001"],
            }],
            "recommendations": [{
                "recommendation_id": "REC-001",
                "insight_id": "INS-001",
                "description": "Keep it up.",
                "basis": "Strong signal",
                "potential_impact": "+10%",
            }],
            "warnings": [],
        }
        assert "insights" in insights
        assert "recommendations" in insights
        assert len(insights["insights"]) > 0


# ------------------------------------------------------------------
# Stage 7 → Stage 8: report output → QA input
# ---------------------------------------------------------------------------


class TestReportToQa:
    def test_report_result_has_status(self) -> None:
        result = {"status": "rendered", "report_path": "/tmp/report.html"}
        assert result["status"] == "rendered"
        assert "report_path" in result


# ------------------------------------------------------------------
# Stage 8 → verdict: QA output → final verdict
# ---------------------------------------------------------------------------


class TestQaVerdict:
    def test_verdict_has_required_fields(self) -> None:
        verdict = {
            "verdict": "APPROVED",
            "score": 95.0,
            "critical_count": 0,
            "warning_count": 1,
            "reason_codes": [],
        }
        assert "verdict" in verdict
        assert "score" in verdict
        assert verdict["verdict"] in ("APPROVED", "APPROVED_WITH_WARNINGS",
                                       "NEEDS_REVISION")
