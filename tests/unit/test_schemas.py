"""Unit tests for shared/schemas — contracts for every JSON artifact.

Round-trip (build -> dump -> re-parse -> equal) is the guarantee that every
stage can write a JSON file that every later stage can read back.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared import schemas as s


def round_trip(model):
    rebuilt = type(model).model_validate_json(model.model_dump_json())
    assert rebuilt == model


def test_dsl_operation_contract():
    op = s.DslOperation(function="sum", column="revenue", group_by=["category"])
    assert op.column == "revenue"
    round_trip(op)


@pytest.mark.parametrize("function", ["eval", "avg", "sum("])
def test_dsl_operation_rejects_unknown_function(function):
    with pytest.raises(ValidationError):
        s.DslOperation(function=function, column="revenue")


@pytest.mark.parametrize("period", ["QoQ", "quarterly"])
def test_dsl_operation_rejects_bad_period(period):
    with pytest.raises(ValidationError):
        s.DslOperation(function="growth", column="revenue",
                       over_column="order_date", period=period)


def test_kpi_result_round_trip():
    kpi = s.KpiResult(kpi_id="KPI-001", name="Total Revenue",
                      operation={"function": "sum", "column": "revenue"},
                      value=2450000.50, evidence_id="EV-001")
    assert kpi.value == 2450000.50
    assert kpi.computed_by == "pandas"
    round_trip(kpi)


def test_kpi_result_accepts_int_and_str_values():
    assert s.KpiResult(kpi_id="K", name="c", operation={"function": "count",
                                                        "column": "id"},
                       value=100).value == 100
    assert s.KpiResult(kpi_id="K", name="m", operation={"function": "min",
                                                        "column": "date"},
                       value="2024-01-01").value == "2024-01-01"


def test_statistical_result_round_trip_with_extra():
    st = s.StatisticalResult(
        test_id="ST-001", category="correlation", test_name="pearson",
        statistic=0.72, p_value=0.01, ci_low=0.5, ci_high=0.9,
        effect_size=0.72, n=100, evidence_id="EV-002", extra={"df": 98})
    assert st.extra == {"df": 98}
    round_trip(st)


def test_chart_metadata_round_trip_and_reliability():
    ch = s.ChartMetadata(chart_id="CH-004", kind="line",
                         reason="1 ordered dim (month) >= 3 points -> line",
                         columns=["order_date", "revenue"],
                         reliability="low_n", evidence_id="EV-007")
    assert ch.reliability == "low_n"
    round_trip(ch)


def test_chart_metadata_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        s.ChartMetadata(chart_id="CH-1", kind="pie_chart", reason="x")


def test_evidence_entry_lineage_round_trip():
    entry = s.EvidenceEntry(
        evidence_id="EV-006",
        source=s.EvidenceSource(file_hash="sha256:abc", sheet="Sales",
                                transformations=["removed_duplicates"],
                                filter="category==Electronics",
                                aggregation="monthly_sum",
                                comparison="Q4 vs Q3", result=27.4))
    assert entry.source.result == 27.4
    round_trip(entry)


def test_insight_round_trip():
    ins = s.Insight(insight_id="INS-001", claim_type="COMPARATIVE",
                    title="electronics fastest in Q3", description="...",
                    confidence="high", evidence_ids=["EV-001", "EV-006"],
                    required_evidence=["group_comparison", "growth_rate"],
                    related_kpis=["KPI-001"])
    assert ins.claim_type == "COMPARATIVE"
    round_trip(ins)


@pytest.mark.parametrize("claim_type", ["CAUSAL", "DESCRIPTIVE", "PREDICTIVE"])
def test_insight_accepts_all_claim_types(claim_type):
    ins = s.Insight(insight_id="INS-1", claim_type=claim_type,
                    title="t", description="d", confidence="low")
    assert ins.claim_type == claim_type


def test_insight_rejects_unknown_claim_type():
    with pytest.raises(ValidationError):
        s.Insight(insight_id="INS-1", claim_type="OBSERVATIONAL",
                  title="t", description="d", confidence="high")


def test_recommendation_round_trip():
    rec = s.Recommendation(recommendation_id="REC-001", insight_id="INS-001",
                           title="Consider testing...", description="hedged")
    round_trip(rec)


def test_report_result_round_trip():
    rep = s.ReportResult(status="rendered",
                         report_path="runs/run_1/report.html",
                         locale="ar-EG", sections=["Summary", "KPIs"])
    assert rep.locale == "ar-EG"
    round_trip(rep)


def test_qa_verdict_round_trip():
    qa = s.QaVerdict(verdict="NEEDS_REVISION", score=70.0, critical=["x"],
                     reason_codes=["cleaning_retry_limit_exceeded"])
    assert qa.verdict == "NEEDS_REVISION"
    round_trip(qa)


def test_qa_verdict_rejects_unknown_verdict():
    with pytest.raises(ValidationError):
        s.QaVerdict(verdict="PASS", score=100.0)
