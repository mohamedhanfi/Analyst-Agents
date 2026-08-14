"""Unit tests for analysis/chart_planner.py — §2.5 rule table (deterministic)."""
from __future__ import annotations

import pandas as pd
import pytest

from analysis.chart_planner import (
    plan_charts,
    rank_candidates,
    truncate,
)
from analysis.evidence import EvidenceRegistry
from shared.schemas import AnalysisPlan, ChartMetadata, DslOperation, KpiCandidate
from tests.unit.conftest import SALES

# Shape rule tests disable thin-downgrade (SALES is only 6 rows); low_n
# behavior is exercised explicitly with the default threshold.
NOT_THIN = {"thin_threshold": 0}


def _registry():
    return EvidenceRegistry(run_id="test")


def _kpi(kpi_id, name, function="sum", column="revenue", **op_kwargs):
    return KpiCandidate(kpi_id=kpi_id, name=name,
                        operation=DslOperation(function=function, column=column,
                                               **op_kwargs))


def _plan(*kpis, has_temporal=True):
    return AnalysisPlan(candidate_kpis=list(kpis), has_temporal_data=has_temporal)


def test_empty_plan_produces_shape_charts(make_understanding):
    results, truncated = plan_charts(SALES, _plan(),
                                     make_understanding(SALES), _registry(),
                                     **NOT_THIN)
    assert truncated is False
    kinds = {c.kind for c in results}
    assert kinds == {"scatter", "histogram"}  # 2 measures -> rule 7 + rule 6


def test_growth_line_rule_2(make_understanding):
    kpi = _kpi("K1", "revenue growth", function="growth",
               column="revenue", over_column="date", period="YoY")
    results, _ = plan_charts(SALES, _plan(kpi),
                             make_understanding(SALES), _registry(),
                             **NOT_THIN)
    line = next(c for c in results if c.kind == "line")
    assert "rule_2" in line.reason
    assert line.columns == ["date", "revenue"]


def test_growth_too_few_points_downgrades_low_n(make_understanding):
    frame = pd.DataFrame({"date": ["2024-01-15", "2024-02-15"],
                          "revenue": [100, 200]})
    understanding = make_understanding(frame, measures=("revenue",))
    kpi = _kpi("K1", "growth", function="growth", column="revenue",
               over_column="date", period="MoM")
    results, _ = plan_charts(frame, _plan(kpi), understanding, _registry())
    growth = next(c for c in results if c.columns == ["date", "revenue"])
    assert growth.kind == "bar"                 # downgraded from line
    assert growth.reliability == "low_n"
    assert "time points" in growth.reason


def test_thin_frame_downgrades_to_low_n_bar(make_understanding):
    kpi = _kpi("K1", "revenue by product", column="revenue",
               group_by=["product"])
    results, _ = plan_charts(SALES, _plan(kpi),
                             make_understanding(SALES), _registry())
    assert all(c.kind == "bar" for c in results)
    assert all(c.reliability == "low_n" for c in results)


def test_single_dimension_two_values_bar_rule_1(make_understanding):
    frame = SALES.copy()
    frame["order_id"] = [f"o{i}" for i in range(len(frame))]
    kpi = _kpi("K1", "revenue by product", column="revenue",
               group_by=["product"])
    understanding = make_understanding(frame)
    results, _ = plan_charts(frame, _plan(kpi), understanding, _registry(),
                             **NOT_THIN)
    bar = next(c for c in results if c.kind == "bar")
    assert "rule_1" in bar.reason


def test_three_to_twelve_values_vertical_bar_rule_3(make_understanding):
    frame = pd.DataFrame({
        "month": [f"2024-{m:02d}" for m in range(1, 13)],
        "revenue": [float(m * 100) for m in range(1, 13)],
    })
    kpi = _kpi("K1", "revenue by month", column="revenue",
               group_by=["month"])
    understanding = make_understanding(frame, measures=("revenue",),
                                       temporal=(), dimensions=("month",))
    results, _ = plan_charts(frame, _plan(kpi), understanding, _registry(),
                             **NOT_THIN)
    bar = next(c for c in results if c.kind == "bar")
    assert "rule_3" in bar.reason


def test_thirteen_to_fifty_values_horizontal_bar_rule_4(make_understanding):
    frame = pd.DataFrame({
        "product": [f"p{i:02d}" for i in range(13)],
        "revenue": [float(i * 10) for i in range(13)],
    })
    kpi = _kpi("K1", "revenue by product", column="revenue",
               group_by=["product"])
    understanding = make_understanding(frame, measures=("revenue",),
                                       temporal=(), dimensions=("product",))
    results, _ = plan_charts(frame, _plan(kpi), understanding, _registry(),
                             **NOT_THIN)
    barh = next(c for c in results if c.kind == "barh")
    assert "rule_4" in barh.reason


def test_more_than_fifty_values_top15_rollup_rule_5(make_understanding):
    frame = pd.DataFrame({
        "product": [f"p{i:03d}" for i in range(51)],
        "revenue": [float(i) for i in range(51)],
    })
    kpi = _kpi("K1", "revenue by product", column="revenue",
               group_by=["product"])
    understanding = make_understanding(frame, measures=("revenue",),
                                       temporal=(), dimensions=("product",))
    results, _ = plan_charts(frame, _plan(kpi), understanding, _registry(),
                             **NOT_THIN)
    barh = next(c for c in results if c.kind == "barh")
    assert "rule_5" in barh.reason
    assert "$rest" in barh.reason


def test_share_doughnut_rule_9(make_understanding):
    kpi = _kpi("K1", "share of revenue by product", column="revenue",
               group_by=["product"])
    understanding = make_understanding(SALES)
    results, _ = plan_charts(SALES, _plan(kpi), understanding, _registry(),
                             **NOT_THIN)
    doughnut = next(c for c in results if c.kind == "doughnut")
    assert "rule_9" in doughnut.reason


def test_correlation_scatter_rule_7(make_understanding):
    kpi = _kpi("K1", "revenue vs quantity", function="correlation",
               column_a="revenue", column_b="quantity")
    results, _ = plan_charts(SALES, _plan(kpi),
                             make_understanding(SALES), _registry(),
                             **NOT_THIN)
    scatter = next(c for c in results if c.kind == "scatter")
    assert "rule_7" in scatter.reason
    assert scatter.columns == ["revenue", "quantity"]


def test_three_measures_heatmap_rule_8(make_understanding):
    frame = SALES.copy()
    frame["margin"] = [10.0, 20.0, 30.0, 5.0, 8.0, 12.0]
    understanding = make_understanding(frame, measures=("revenue", "quantity",
                                                        "margin"))
    results, _ = plan_charts(frame, _plan(), understanding, _registry(),
                             **NOT_THIN)
    heatmap = next(c for c in results if c.kind == "heatmap")
    assert "rule_8" in heatmap.reason
    assert set(heatmap.columns) == {"revenue", "quantity", "margin"}


def test_histogram_rule_6(make_understanding):
    results, _ = plan_charts(SALES, _plan(),
                             make_understanding(SALES), _registry(),
                             **NOT_THIN)
    histograms = [c for c in results if c.kind == "histogram"]
    assert len(histograms) == 2
    assert all("rule_6" in c.reason for c in histograms)


def test_plan_skips_headline_aggregates(make_understanding):
    kpi = _kpi("K1", "total revenue", column="revenue")
    results, _ = plan_charts(SALES, _plan(kpi),
                             make_understanding(SALES), _registry(),
                             **NOT_THIN)
    assert all(c.kind != "bar" for c in results)  # no shape from a headline


def test_every_chart_mints_and_registers_evidence(make_understanding):
    registry = _registry()
    kpi = _kpi("K1", "revenue by product", column="revenue",
               group_by=["product"])
    results, _ = plan_charts(SALES, _plan(kpi),
                             make_understanding(SALES), registry)
    for chart in results:
        assert registry.get(chart.evidence_id) is not None
    assert len(registry) == len(results)


def test_ranking_prefers_stronger_shapes():
    bar = ChartMetadata(chart_id="CH-001", kind="bar", reason="r",
                        columns=["a", "b"], title="t")
    line = ChartMetadata(chart_id="CH-002", kind="line", reason="r",
                         columns=["a", "b"], title="t")
    hist = ChartMetadata(chart_id="CH-003", kind="histogram", reason="r",
                         columns=["a"], title="t")
    assert [c.chart_id for c in rank_candidates([hist, bar, line])] == \
        ["CH-002", "CH-001", "CH-003"]


def test_truncate_drops_lowest_ranked():
    candidates = [
        ChartMetadata(chart_id=f"CH-{i:03d}", kind="histogram", reason="r",
                      columns=["a"], title="t")
        for i in range(1, 6)
    ]
    kept, truncated = truncate(rank_candidates(candidates), 3)
    assert truncated is True
    assert len(kept) == 3
    assert [c.chart_id for c in kept] == ["CH-001", "CH-002", "CH-003"]


def test_truncate_zero_max():
    candidates = [ChartMetadata(chart_id="CH-001", kind="bar", reason="r",
                                columns=["a"], title="t")]
    kept, truncated = truncate(candidates, 0)
    assert kept == [] and truncated is True


def test_max_chart_count_respected(make_understanding):
    kpi = _kpi("K1", "revenue by product", column="revenue",
               group_by=["product"])
    results, truncated = plan_charts(SALES, _plan(kpi),
                                     make_understanding(SALES), _registry(),
                                     **NOT_THIN, max_chart_count=1)
    assert truncated is True
    assert len(results) == 1


def test_plan_is_deterministic(make_understanding):
    kpi = _kpi("K1", "revenue by product", column="revenue",
               group_by=["product"])
    understanding = make_understanding(SALES)
    first, _ = plan_charts(SALES, _plan(kpi), understanding, _registry(),
                           **NOT_THIN)
    second, _ = plan_charts(SALES, _plan(kpi), understanding, _registry(),
                            **NOT_THIN)
    assert [c.chart_id for c in first] == [c.chart_id for c in second]
    assert [c.kind for c in first] == [c.kind for c in second]


def test_chart_ids_unique_across_kpi_and_shape_charts(make_understanding):
    kpis = [
        _kpi("K1", "revenue growth", function="growth", column="revenue",
             over_column="date", period="YoY"),
        _kpi("K2", "revenue by product", column="revenue",
             group_by=["product"]),
        _kpi("K3", "revenue vs quantity", function="correlation",
             column_a="revenue", column_b="quantity"),
    ]
    results, _ = plan_charts(SALES, _plan(*kpis),
                             make_understanding(SALES), _registry(),
                             **NOT_THIN)
    ids = [c.chart_id for c in results]
    assert len(ids) == len(set(ids))
