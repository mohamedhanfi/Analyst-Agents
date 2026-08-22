"""Stage 5b — chart quality gate + data-quality confidence labels.

Every chart is checked before it is trusted in the report:

* SVG integrity — file exists, well-formed root, and the renderer did not
  emit its own "render error" / "no data" fallbacks while data existed.
* Aggregation match — for plain sum/count charts tied to a KPI, the
  rendered group totals must equal the KPI value (within tolerance), so a
  chart never contradicts the headline number.
* Data-quality label — the DQ report (missingness, repair, contract
  violations, outlier flags) stamps each chart with a confidence label so
  the reader sees "this number sits on shaky data".

Output is metadata/chart_quality.json + per-chart quality stamps on the
ChartMetadata objects (which land in chart_metadata.json). Deterministic,
never raises.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from analysis.chart_renderer import _aggregate_pairs, _raw_numeric
from shared.schemas import ChartMetadata, KpiResult

_AGG_TOLERANCE = 0.001          # 0.1% relative tolerance on aggregation match
_AGG_CHECKABLE = ("sum", "count")


def assess_svg(svg: str, has_data: bool) -> Dict[str, Any]:
    """Well-formedness + content checks on a rendered SVG string."""
    checks: Dict[str, Any] = {
        "has_svg_root": svg.lstrip().startswith("<svg"),
        "renderer_error": "render error" in svg,
        "empty_placeholder": "no data" in svg and has_data,
        "has_graphics": any(tag in svg for tag in
                            ("<rect", "<circle", "<line", "<polyline",
                             "<polygon", "<path", "<text")),
    }
    ok = (checks["has_svg_root"] and not checks["renderer_error"]
          and not checks["empty_placeholder"] and checks["has_graphics"])
    return {"ok": ok, "checks": checks}


def _chart_totals(chart: ChartMetadata, df: pd.DataFrame,
                  kpis: List[KpiResult] | None = None
                  ) -> Tuple[float | None, int]:
    """(sum of rendered groups, point count) for bar-like charts."""
    try:
        from analysis.chart_renderer import _operation
        op = _operation(chart, kpis or [])
    except Exception:  # noqa: BLE001 -- no operation -> no totals
        return None, 0
    if op is None:
        return None, 0
    if chart.kind in ("histogram", "boxplot", "scatter", "heatmap"):
        series = _raw_numeric(df, (chart.columns or [None])[0]) \
            if chart.columns else []
        return None, len(series)
    if chart.kind in ("bar", "barh", "lollipop", "doughnut", "pie",
                      "line", "area", "stacked_bar", "pareto", "waterfall"):
        labels, values = _aggregate_pairs(df, op, chart)
        return (sum(values) if values else None), len(values)
    return None, 0


def check_aggregation(chart: ChartMetadata, df: pd.DataFrame,
                      kpis: List[KpiResult]) -> Dict[str, Any]:
    """The chart's rendered totals must match its KPI value.

    Only meaningful for sum/count aggregates where group totals sum up to
    the headline number; everything else is skipped (not a failure).
    """
    kpi = next((k for k in kpis if k.kpi_id == chart.kpi_id), None)
    if kpi is None or kpi.value is None:
        return {"checked": False, "reason": "no_kpi_match"}
    op = (kpi.operation.model_dump() if hasattr(kpi.operation, "model_dump")
          else dict(kpi.operation))
    if op.get("function") not in _AGG_CHECKABLE or not op.get("group_by"):
        return {"checked": False, "reason": "not_sum_count_groupable"}
    total, _ = _chart_totals(chart, df, kpis)
    if total is None:
        return {"checked": False, "reason": "no_renderable_totals"}
    expected = float(kpi.value)
    if expected == 0:
        matches = total == 0
    else:
        matches = abs(total - expected) <= abs(expected) * _AGG_TOLERANCE
    return {"checked": True, "matches": matches,
            "expected": expected, "actual": total}


def data_quality_label(run_dir: Path) -> Dict[str, Any]:
    """Confidence label from the DQ artifacts (never raises)."""
    meta = Path(run_dir) / "metadata"
    dq: Dict[str, Any] = {}
    try:
        dq = json.loads((meta / "data_quality_report.json")
                        .read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"label": "unknown", "detail": "no DQ report available"}
    violations = 0
    try:
        violations = len(json.loads((meta / "contract_violations.json")
                                    .read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        pass
    issues = dq.get("issues") or []
    high = [i for i in issues if i.get("severity") == "high"]
    missing_rate = float((dq.get("missingness") or {}).get("rate", 0.0) or 0)
    outliers = [i for i in issues
                if str(i.get("detail", "")).startswith("outliers_")]
    if high:
        label, detail = "data_warning", \
            f"{len(high)} high-severity DQ issues"
    elif violations:
        label, detail = "data_warning", \
            f"{violations} contract violations"
    elif missing_rate >= 0.1:
        label, detail = "data_warning", \
            f"missingness rate {missing_rate:.1%}"
    elif outliers:
        label, detail = "data_warning", f"{len(outliers)} outlier flags"
    else:
        label, detail = "ok", "no significant DQ findings"
    return {"label": label, "detail": detail,
            "contract_violations": violations}


def check_chart(chart: ChartMetadata, df: pd.DataFrame,
                kpis: List[KpiResult], charts_dir: Path,
                quality_label: Dict[str, Any]) -> Dict[str, Any]:
    """Run every check on one chart; returns the verdict record."""
    svg_path = charts_dir / f"{chart.chart_id}.svg"
    if not svg_path.is_file():
        return {"chart_id": chart.chart_id, "verdict": "fail",
                "reason": "svg_missing", "reasons": ["svg_missing"],
                "kind": chart.kind}
    svg = svg_path.read_text(encoding="utf-8")
    agg = check_aggregation(chart, df, kpis)
    points: int = 0
    try:
        _, points = _chart_totals(chart, df, kpis)
    except Exception:  # noqa: BLE001 -- point count is informational
        points = 0
    svg_check = assess_svg(svg, has_data=points > 0)

    reasons: List[str] = []
    if not svg_check["ok"]:
        reasons.append("svg_failed_checks")
    if agg.get("checked") and not agg.get("matches"):
        reasons.append(
            f"aggregation_mismatch (kpi {agg.get('expected')} vs "
            f"chart {agg.get('actual')})")
    if chart.reliability == "low_n":
        reasons.append("low_n")
    if quality_label.get("label") == "data_warning":
        reasons.append(f"data_quality:{quality_label.get('detail', '')}")

    verdict = "pass"
    if any(r.startswith("svg_") or r.startswith("aggregation_")
           for r in reasons):
        verdict = "fail"
    elif reasons:
        verdict = "warn"

    return {
        "chart_id": chart.chart_id,
        "kind": chart.kind,
        "verdict": verdict,
        "reasons": reasons,
        "svg_checks": svg_check["checks"],
        "aggregation": agg,
        "points": points,
        "reliability": chart.reliability,
        "data_quality_label": quality_label.get("label"),
    }


def run_quality_gate(run_dir: Path, charts: List[ChartMetadata],
                     df: pd.DataFrame,
                     kpis: List[KpiResult]) -> Dict[str, Any]:
    """Gate every chart; stamp verdicts onto the ChartMetadata objects."""
    charts_dir = Path(run_dir) / "charts"
    label = data_quality_label(run_dir)
    results: List[Dict[str, Any]] = []
    for chart in charts:
        record = check_chart(chart, df, kpis, charts_dir, label)
        results.append(record)
        chart.quality = record["verdict"]
        chart.quality_reason = "; ".join(record["reasons"]) or None

    summary = {
        "passed": sum(1 for r in results if r["verdict"] == "pass"),
        "warned": sum(1 for r in results if r["verdict"] == "warn"),
        "failed": sum(1 for r in results if r["verdict"] == "fail"),
    }
    payload = {
        "data_quality_label": label,
        "summary": summary,
        "charts": results,
    }
    meta_dir = Path(run_dir) / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "chart_quality.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload