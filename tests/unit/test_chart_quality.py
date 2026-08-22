"""Unit tests for analysis/chart_quality.py — the quality gate + labels."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from analysis.chart_quality import (assess_svg, check_aggregation,
                                    data_quality_label, run_quality_gate)
from analysis.chart_planner import plan_charts
from analysis.chart_renderer import render_all
from analysis.evidence import EvidenceRegistry
from shared.schemas import (AnalysisPlan, ChartMetadata, DslOperation,
                            KpiCandidate, KpiResult)
from tests.unit.conftest import SALES


def _registry():
    return EvidenceRegistry(run_id="r1", file_hash="sha256:x",
                            transformations=[])


def _plan(kpi):
    return AnalysisPlan(candidate_kpis=[kpi], has_temporal_data=True)


def _kpi(kpi_id="K1", name="revenue", function="sum", column="revenue",
         **op_kwargs):
    return KpiCandidate(kpi_id=kpi_id, name=name,
                        operation=DslOperation(function=function,
                                               column=column, **op_kwargs))


def test_assess_svg_ok_and_broken():
    ok = assess_svg("<svg><rect x='1'/></svg>", has_data=True)
    assert ok["ok"] is True
    broken = assess_svg("<svg><text>render error</text></svg>",
                        has_data=True)
    assert broken["ok"] is False
    empty = assess_svg("<svg><text>no data</text></svg>", has_data=True)
    assert empty["ok"] is False
    # empty placeholder with NO data is acceptable (nothing to draw)
    legit = assess_svg("<svg><text>no data</text></svg>", has_data=False)
    assert legit["ok"] is True


def test_check_aggregation_matches_and_mismatches():
    kpis = [KpiResult(kpi_id="K1", name="revenue by product",
                      operation=DslOperation(function="sum",
                                             column="revenue",
                                             group_by=["product"]),
                      value=float(SALES["revenue"].sum()))]
    chart = ChartMetadata(chart_id="CH-001", kind="pareto",
                          reason="rule_10", columns=["product", "revenue"],
                          title="revenue by product", kpi_id="K1",
                          evidence_id="EV-1")
    result = check_aggregation(chart, SALES, kpis)
    assert result["checked"] is True
    assert result["matches"] is True
    wrong = [KpiResult(kpi_id="K1", name="revenue by product",
                       operation=DslOperation(function="sum",
                                              column="revenue",
                                              group_by=["product"]),
                       value=999999.0)]
    assert check_aggregation(chart, SALES, wrong)["matches"] is False


def test_data_quality_label_from_report(tmp_path):
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    (meta / "data_quality_report.json").write_text(json.dumps({
        "status": "passed", "issues": [], "missingness": {"rate": 0.02},
    }), encoding="utf-8")
    (meta / "contract_violations.json").write_text(json.dumps([]),
                                                   encoding="utf-8")
    assert data_quality_label(tmp_path)["label"] == "ok"
    (meta / "data_quality_report.json").write_text(json.dumps({
        "status": "needs_repair",
        "issues": [{"severity": "high", "detail": "negative"}],
        "missingness": {"rate": 0.0},
    }), encoding="utf-8")
    assert data_quality_label(tmp_path)["label"] == "data_warning"


def test_quality_gate_stamps_and_writes(tmp_path, make_understanding):
    kpi = _kpi("K1", "revenue by product", group_by=["product"])
    understanding = make_understanding(SALES)
    registry = _registry()
    charts, _ = plan_charts(SALES, _plan(kpi), understanding, registry,
                            thin_threshold=0)
    kpis = [KpiResult(kpi_id="K1", name="revenue by product",
                      operation=DslOperation(function="sum",
                                             column="revenue",
                                             group_by=["product"]),
                      value=float(SALES["revenue"].sum()))]
    render_all(charts, SALES, kpis, tmp_path / "charts")
    payload = run_quality_gate(tmp_path, charts, SALES, kpis)
    assert payload["summary"]["passed"] >= 1
    assert all(c.quality == "pass" for c in charts)
    assert (tmp_path / "metadata" / "chart_quality.json").is_file()
    saved = json.loads((tmp_path / "metadata" / "chart_quality.json")
                       .read_text(encoding="utf-8"))
    assert saved["data_quality_label"]["label"] in ("ok", "data_warning",
                                                    "unknown")
    assert saved["summary"]["passed"] == payload["summary"]["passed"]


def test_quality_gate_fails_missing_svg(tmp_path, make_understanding):
    kpi = _kpi("K1", "revenue by product", group_by=["product"])
    understanding = make_understanding(SALES)
    charts, _ = plan_charts(SALES, _plan(kpi), understanding, _registry(),
                            thin_threshold=0)
    payload = run_quality_gate(tmp_path, charts, SALES, [])
    assert payload["summary"]["failed"] == len(charts)
    assert all(c.quality == "fail" for c in charts)