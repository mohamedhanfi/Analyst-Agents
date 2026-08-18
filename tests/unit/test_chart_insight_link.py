"""Tests for task B — chart diversity & insight-linked selection.

Covers: the insight-claim→kind override layer (B.3) producing 3+ distinct
kinds for a diverse insight set, the novelty_penalty ranking bias (B.2),
and a regression guard that the deterministic planner + ranking still
behave exactly as before when the penalty is off (default 0.0).
"""
from __future__ import annotations

from analysis.chart_planner import (apply_insight_kind_overrides,
                                    plan_charts, rank_candidates)
from analysis.evidence import EvidenceRegistry
from shared.schemas import AnalysisPlan, ChartMetadata, Insight
from tests.unit.conftest import GROUPS, SALES


def _registry():
    return EvidenceRegistry(run_id="test")


def _chart(chart_id, kind, kpi_id=None, columns=None, reason="fixture"):
    return ChartMetadata(chart_id=chart_id, kind=kind, reason=reason,
                         kpi_id=kpi_id, columns=columns or [])


def _insight(insight_id, claim_type, kpi_id, required_evidence=None):
    return Insight(
        insight_id=insight_id, claim_type=claim_type, title="t",
        description="d", confidence="high", evidence_ids=["e1"],
        required_evidence=required_evidence or [],
        related_kpis=[kpi_id])


# ---------------------------------------------------------------------------
# B.3 — insight-linked kind overrides
# ---------------------------------------------------------------------------


def test_insight_linked_kinds_diverse(make_understanding):
    """Golden fixture: comparative + time-trend + correlation claims must
    produce >= 3 distinct chart kinds, not repeats of the same shape."""
    understanding = make_understanding(SALES)
    df = SALES
    insights = [
        _insight("I-1", "COMPARATIVE", "KPI-A",
                 required_evidence=["group_comparison"]),
        _insight("I-2", "DESCRIPTIVE", "KPI-B",
                 required_evidence=["growth_rate"]),  # trend claim
        _insight("I-3", "CORRELATIONAL", "KPI-C",
                 required_evidence=["correlation"]),
    ]
    charts = [
        _chart("c1", "histogram", kpi_id="KPI-A", columns=["category"]),
        _chart("c2", "histogram", kpi_id="KPI-B", columns=["revenue"]),
        _chart("c3", "histogram", kpi_id="KPI-C", columns=["revenue",
                                                           "quantity"]),
    ]
    updated, applied = apply_insight_kind_overrides(
        charts, insights, df, understanding)
    assert len(applied) == 3
    kinds = {c.kind for c in updated}
    assert len(kinds) >= 3
    by_id = {c.chart_id: c for c in updated}
    assert by_id["c1"].kind == "bar"      # COMPARATIVE -> bar
    assert by_id["c2"].kind == "line"     # trend -> line
    assert by_id["c3"].kind == "scatter"  # CORRELATIONAL -> scatter
    assert "insight_linked" in by_id["c1"].reason
    assert "I-3" in by_id["c3"].reason


def test_insight_override_keeps_rule_kind_when_shape_does_not_fit(
        make_understanding):
    """scatter needs >= 2 measures — a correlation claim on a single-measure
    frame must NOT force a scatter that cannot be drawn."""
    frame = SALES[["date", "product", "revenue"]]
    understanding = make_understanding(
        frame, measures=("revenue",), temporal=("date",),
        dimensions=("product",))
    insights = [_insight("I-1", "CORRELATIONAL", "KPI-C",
                         required_evidence=["correlation"])]
    charts = [_chart("c1", "histogram", kpi_id="KPI-C",
                     columns=["revenue"])]
    updated, applied = apply_insight_kind_overrides(
        charts, insights, frame, understanding)
    assert applied == []
    assert updated[0].kind == "histogram"


def test_insight_override_leaves_unrelated_charts_untouched(make_understanding):
    understanding = make_understanding(SALES)
    insights = [_insight("I-1", "COMPARATIVE", "KPI-A")]
    charts = [_chart("c1", "histogram", kpi_id="KPI-A", columns=["category"]),
              _chart("c2", "scatter", kpi_id="KPI-Z", columns=["revenue"])]
    updated, applied = apply_insight_kind_overrides(
        charts, insights, SALES, understanding)
    assert len(applied) == 1
    assert updated[0].kind == "bar"
    assert updated[1].kind == "scatter"  # untouched


# ---------------------------------------------------------------------------
# B.2 — novelty penalty ranking
# ---------------------------------------------------------------------------


def _bars_and_hist():
    return [
        _chart("c1", "bar"), _chart("c2", "bar"), _chart("c3", "bar"),
        _chart("c4", "histogram"),
    ]


def test_novelty_penalty_promotes_diverse_kinds():
    charts = _bars_and_hist()
    # Without a penalty: strength order (bar=80 x3, histogram=60) unchanged.
    plain = [c.kind for c in rank_candidates(charts)]
    assert plain == ["bar", "bar", "bar", "histogram"]
    # With the penalty, repeated bars sink below the unique histogram.
    diverse = [c.kind for c in rank_candidates(charts, novelty_penalty=0.3)]
    assert diverse[0] == "bar"
    assert diverse[1] == "histogram"
    assert diverse[2:] == ["bar", "bar"]


def test_novelty_penalty_deterministic():
    charts = _bars_and_hist() + [_chart("c5", "bar")]
    order1 = [c.chart_id for c in rank_candidates(charts, 0.3)]
    order2 = [c.chart_id for c in rank_candidates(charts, 0.3)]
    assert order1 == order2


# ---------------------------------------------------------------------------
# Regression: deterministic planner unchanged (penalty off by default)
# ---------------------------------------------------------------------------


def test_plan_charts_accepts_novelty_penalty(make_understanding):
    """plan_charts accepts the penalty and never raises on a mixed set."""
    understanding = make_understanding(SALES)
    plan = AnalysisPlan(candidate_kpis=[], statistical_tests=[],
                        has_temporal_data=True)
    charts, truncated = plan_charts(
        SALES, plan, understanding, _registry(), thin_threshold=0,
        max_chart_count=10, novelty_penalty=0.3)
    assert isinstance(charts, list) and len(charts) > 0
    assert isinstance(truncated, bool)
    kinds = [c.kind for c in charts]
    assert len(set(kinds)) >= 2


def test_rank_unchanged_without_penalty():
    charts = _bars_and_hist()
    assert [c.chart_id for c in rank_candidates(charts)] == [
        "c1", "c2", "c3", "c4"]