"""Unit tests for analysis/chart_renderer.py — Stage 5b SVG output (§2.5).

Every kind must render deterministic, escaped SVG with title/desc caption.
"""
from __future__ import annotations

import pandas as pd
import pytest

from analysis import chart_renderer as cr
from shared.schemas import ChartMetadata, DslOperation, KpiResult
from tests.unit.conftest import GROUPS, SALES

KINDS = ["bar", "barh", "lollipop", "line", "area", "doughnut", "pie",
         "histogram", "boxplot", "scatter", "heatmap", "stacked_bar"]


def _kpi(grouped=False, growth=False):
    if growth:
        op = DslOperation(function="growth", column="revenue",
                          over_column="date", period="MoM")
        return [KpiResult(kpi_id="K1", name="revenue growth", operation=op,
                          value=0.2, evidence_id="E1", computed_by="dsl")]
    op = DslOperation(function="sum", column="revenue", over_column="date",
                      group_by=["category"] if grouped else None)
    return [KpiResult(kpi_id="K1", name="revenue by category", operation=op,
                      value=69.0, evidence_id="E1", computed_by="dsl")]


def _chart(kind, columns=None, kpi_id="K1"):
    return ChartMetadata(chart_id="CH-001", kind=kind, title=f"t {kind}",
                         reason="rule", columns=columns or ["revenue"],
                         kpi_id=kpi_id, evidence_id="E1", index=0)


def test_escape_xml_covers_all_special_chars():
    raw = '<a&b>"c\'d'
    escaped = cr.escape_xml(raw)
    assert "&lt;" in escaped and "&amp;" in escaped and "&quot;" in escaped \
        and "&apos;" in escaped
    assert "<" not in escaped.replace("&lt;", "")


def test_fmt_is_deterministic_and_short():
    assert cr._fmt(1000000) == "1,000,000"
    assert cr._fmt(1235.6) == "1,236"
    assert cr._fmt(0.3333333) == "0.333"
    assert cr._fmt(0) == "0"


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_renders_valid_svg(kind):
    if kind == "stacked_bar":
        chart = _chart(kind, ["category", "revenue", "quantity"])
    elif kind in ("scatter", "heatmap"):
        chart = _chart(kind, ["revenue", "quantity"])
    elif kind in ("histogram", "boxplot"):
        chart = _chart(kind, ["revenue"])
    else:
        chart = _chart(kind, ["revenue"])
    kpis = _kpi(grouped=(kind in ("bar", "barh", "lollipop", "doughnut",
                                  "pie")), growth=(kind in ("line", "area")))
    svg = cr.render_chart(chart, SALES, kpis)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "aria-labelledby" in svg
    assert ("<title" in svg) and ("<desc" in svg)
    assert cr._caption(chart) in svg


def test_render_is_deterministic():
    chart = _chart("bar", ["revenue"])
    first = cr.render_chart(chart, GROUPS, _kpi(grouped=True))
    second = cr.render_chart(chart, GROUPS, _kpi(grouped=True))
    assert first == second


def test_grouped_bar_recomputes_from_all_rows():
    chart = _chart("bar", ["revenue"])
    svg = cr.render_chart(chart, GROUPS, _kpi(grouped=True))
    assert svg.count("<rect") >= 2  # one bar per group (X, Y)


def test_growth_line_uses_period_series():
    chart = _chart("line", ["date", "revenue"])
    svg = cr.render_chart(chart, SALES, _kpi(growth=True))
    assert "<polyline" in svg
    assert "<circle" in svg  # markers on points


def test_histogram_and_boxplot_from_raw_column():
    hist = cr.render_chart(_chart("histogram", ["revenue"]), SALES, _kpi())
    assert "<rect" in hist
    box = cr.render_chart(_chart("boxplot", ["revenue"]), SALES, _kpi())
    assert "<rect" in box and "Q1:" in box


def test_heatmap_renders_correlation_grid():
    chart = _chart("heatmap", ["revenue", "quantity"])
    svg = cr.render_chart(chart, SALES, _kpi())
    assert svg.count("<rect") >= 4  # 2x2 cells


def test_no_data_falls_back_without_crashing():
    empty = pd.DataFrame({"revenue": []})
    chart = _chart("bar", ["revenue"])
    svg = cr.render_chart(chart, empty, _kpi())
    assert "no data" in svg


def test_unknown_kind_renders_error_message():
    chart = ChartMetadata.model_construct(
        chart_id="CH-001", kind="klingon", title="t", reason="rule",
        columns=["revenue"], index=0)
    svg = cr.render_chart(chart, SALES, _kpi())
    assert "unknown kind" in svg


def test_labels_are_escaped_inside_svg():
    frame = pd.DataFrame({"category": ["<A&B>", "C"], "revenue": [1.0, 2.0]})
    chart = _chart("bar", ["category", "revenue"])
    svg = cr.render_chart(chart, frame, _kpi(grouped=True))
    assert "<A&B>" not in svg
    assert "&lt;A&amp;B&gt;" in svg


def test_render_all_writes_files_and_returns_paths(tmp_path):
    chart = _chart("bar", ["revenue"])
    paths = cr.render_all([chart], GROUPS, _kpi(grouped=True), tmp_path)
    assert paths["CH-001"] == tmp_path / "CH-001.svg"
    assert paths["CH-001"].exists()
    assert paths["CH-001"].read_text(encoding="utf-8").startswith("<svg")


def test_render_all_survives_bad_chart(tmp_path):
    good = _chart("bar", ["revenue"])
    bad = ChartMetadata.model_construct(
        chart_id="CH-002", kind="klingon", title="t", reason="rule",
        columns=["revenue"], index=1)
    paths = cr.render_all([good, bad], GROUPS, _kpi(grouped=True), tmp_path)
    assert set(paths) == {"CH-001", "CH-002"}
    assert "render error" in paths["CH-002"].read_text(encoding="utf-8") \
        or "unknown kind" in paths["CH-002"].read_text(encoding="utf-8")


def test_shape_chart_without_kpi_renders_from_columns(tmp_path=None):
    chart = ChartMetadata(chart_id="CH-001", kind="histogram",
                          title="revenue distribution", reason="rule_6",
                          columns=["revenue"], kpi_id=None, index=0)
    svg = cr.render_chart(chart, SALES, [])
    assert "<rect" in svg