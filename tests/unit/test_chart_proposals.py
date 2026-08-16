"""Unit tests for the Stage 5b hybrid proposals (planner + tool) — §2.5.

LLM proposes kinds; Python validates (12-kind whitelist + data-shape
feasibility); rejected proposals fall back to the rule table with errors.
"""
from __future__ import annotations

import json

import pytest

from analysis.chart_planner import plan_charts, validate_proposed_kinds
from analysis.evidence import EvidenceRegistry
from shared.schemas import AnalysisPlan, DslOperation, KpiCandidate
from tests.unit.conftest import GROUPS, SALES

NOT_THIN = {"thin_threshold": 0}


def _registry():
    return EvidenceRegistry(run_id="test")


def _kpi(kpi_id="K1", name="revenue growth", function="growth",
         column="revenue", over_column="date", period="YoY", **op_kwargs):
    return KpiCandidate(
        kpi_id=kpi_id, name=name,
        operation=DslOperation(function=function, column=column,
                               over_column=over_column, period=period,
                               **op_kwargs))


def _plan(*kpis):
    return AnalysisPlan(candidate_kpis=list(kpis), has_temporal_data=True)


def _understanding(make_understanding, frame):
    return make_understanding(frame)


def test_valid_proposal_overrides_rule_kind(make_understanding):
    kpi = _kpi()  # YoY growth -> rule_2 line
    understanding = _understanding(make_understanding, SALES)
    proposals = [{"kpi_id": "K1", "kind": "area", "reason": "trend shading"}]
    accepted, errors = validate_proposed_kinds(SALES, _plan(kpi),
                                               understanding, proposals)
    assert errors == []
    assert accepted["K1"]["kind"] == "area"
    charts, _ = plan_charts(SALES, _plan(kpi), understanding, _registry(),
                            **NOT_THIN, proposals=proposals)
    chart = next(c for c in charts if c.kpi_id == "K1")
    assert chart.kind == "area"
    assert "llm_proposed_area" in chart.reason
    assert chart.kpi_id == "K1"


def test_unknown_kind_rejected_with_error(make_understanding):
    kpi = _kpi()
    understanding = _understanding(make_understanding, SALES)
    proposals = [{"kpi_id": "K1", "kind": "radar", "reason": "why not"}]
    accepted, errors = validate_proposed_kinds(SALES, _plan(kpi),
                                               understanding, proposals)
    assert accepted == {}
    assert any("unknown chart kind" in e for e in errors)
    charts, _ = plan_charts(SALES, _plan(kpi), understanding, _registry(),
                            **NOT_THIN, proposals=proposals)
    assert charts[0].kind == "line"  # rule table fallback


def test_unknown_kpi_id_rejected(make_understanding):
    kpi = _kpi()
    understanding = _understanding(make_understanding, SALES)
    proposals = [{"kpi_id": "K9", "kind": "pie", "reason": "?"}]
    accepted, errors = validate_proposed_kinds(SALES, _plan(kpi),
                                               understanding, proposals)
    assert accepted == {}
    assert any("unknown kpi_id" in e for e in errors)


def test_infeasible_proposal_rejected_by_shape(make_understanding):
    frame = SALES[["date", "product", "revenue"]]  # single measure
    kpi = _kpi(function="sum", column="revenue", over_column=None,
               period=None)  # one measure -> scatter impossible
    understanding = _understanding(make_understanding, frame)
    proposals = [{"kpi_id": "K1", "kind": "scatter", "reason": "wanted"}]
    accepted, errors = validate_proposed_kinds(frame, _plan(kpi),
                                               understanding, proposals)
    assert accepted == {}
    assert any("does not fit" in e for e in errors)


def test_proposal_on_thin_frame_downgrades_low_n(make_understanding):
    frame = SALES.head(4)
    kpi = _kpi()
    understanding = _understanding(make_understanding, frame)
    proposals = [{"kpi_id": "K1", "kind": "area", "reason": "trend"}]
    charts, _ = plan_charts(frame, _plan(kpi), understanding, _registry(),
                            proposals=proposals)  # default thin threshold
    chart = charts[0]
    assert chart.kind == "bar"          # downgraded, not area
    assert chart.reliability == "low_n"


def test_malformed_proposals_never_crash(make_understanding):
    kpi = _kpi()
    understanding = _understanding(make_understanding, SALES)
    for bad in (None, "junk", [42], [{"kind": "pie"}], [{"kpi_id": "K1"}]):
        accepted, errors = validate_proposed_kinds(
            SALES, _plan(kpi), understanding, bad if bad is None else [bad])
        assert isinstance(accepted, dict)
        assert isinstance(errors, list)
    charts, _ = plan_charts(SALES, _plan(kpi), understanding, _registry(),
                            **NOT_THIN, proposals=None)
    assert charts[0].kind == "line"


def test_planner_tool_returns_proposal_errors(make_understanding, tmp_path):
    from shared.tools.analysis import chart_planner_tool
    csv_path = tmp_path / "sales.csv"
    SALES.to_csv(csv_path, index=False)
    understanding = _understanding(make_understanding, SALES)
    plan = _plan(_kpi(), _kpi(kpi_id="K2", name="revenue by product",
                              function="sum", column="revenue",
                              over_column=None, period=None,
                              group_by=["product"]))
    raw = chart_planner_tool.run(
        str(csv_path), understanding.model_dump_json(),
        plan.model_dump_json(),
        limits_json=json.dumps({"max_chart_count": 10,
                                "thin_threshold": 0}),
        proposals_json=json.dumps([
            {"kpi_id": "K1", "kind": "area", "reason": "trend"},
            {"kpi_id": "K2", "kind": "radar", "reason": "fancy"},
            {"kpi_id": "K9", "kind": "pie", "reason": "ghost"},
        ]))
    payload = json.loads(raw)
    kinds = {c["kind"] for c in payload["charts"]}
    assert "area" in kinds                 # valid proposal honored
    assert "radar" not in kinds            # unknown kind never planned
    assert any("K2" in e for e in payload["proposal_errors"])
    assert any("K9" in e for e in payload["proposal_errors"])
    assert "charts_truncated" in payload