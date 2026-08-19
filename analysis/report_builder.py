"""Stage 7 — Report builder: deterministic HTML rendering from run artifacts.

Loads all JSON artifacts from a completed run, builds a Jinja2 context, and
renders the full HTML report via the approved template.  The LLM writes only
the executive summary — every other section is Python-computed.

Golden rule: Python renders; LLM summarises only.  Jinja2 autoescape=True;
no raw cell content ever reaches HTML.

CLI: python -m analysis.report_builder <run_dir>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

import pandas as pd

from analysis.chart_renderer import (PALETTE, _fd_bins, _hist, _linreg,
                                     _prepare, _to_float)
from shared.formatting import fmt as _fmt
from shared.schemas import ChartMetadata, KpiResult

log = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "resources"
_TEMPLATE_NAME = "report_template.html"
_SECTIONS = [
    "executive_summary", "business_context", "dq_summary",
    "data_overview", "kpis", "stats", "charts",
    "insights", "recommendations", "limitations", "evidence",
    "run_comparison",
]


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def _load_json(path: Path, default: Any = None) -> Any:
    """Load a JSON file; return *default* when missing or malformed."""
    if not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default if default is not None else {}


def load_artifacts(run_dir: Path) -> Dict[str, Any]:
    """Load every JSON artifact from a run directory.

    Returns a flat dict with keys matching the expected Jinja2 context.
    Missing files produce empty defaults — the report always renders.
    """
    run_dir = Path(run_dir)
    out = run_dir / "outputs"
    meta = run_dir / "metadata"
    know = run_dir / "knowledge"

    kpis_raw = _load_json(out / "kpis.json", {"kpis": []})
    stats_raw = _load_json(out / "statistical_results.json", {"results": []})
    insights_raw = _load_json(out / "insights.json",
                              {"insights": [], "recommendations": [],
                               "warnings": []})
    evidence_raw = _load_json(out / "evidence_registry.json", [])
    charts_raw = _load_json(meta / "chart_metadata.json",
                            {"charts": [], "charts_truncated": False})
    context_raw = _load_json(know / "business_context.json")
    profile_raw = _load_json(meta / "data_profile.json")
    understanding_raw = _load_json(meta / "dataset_understanding.json")
    dq_raw = _load_json(meta / "data_quality_report.json")
    cleaning_raw = _load_json(meta / "cleaning_result.json")

    return {
        "kpis": kpis_raw.get("kpis", []) if isinstance(kpis_raw, dict)
        else kpis_raw if isinstance(kpis_raw, list) else [],
        "stats": stats_raw.get("results", []) if isinstance(stats_raw, dict)
        else stats_raw if isinstance(stats_raw, list) else [],
        "insights": insights_raw.get("insights", [])
        if isinstance(insights_raw, dict) else [],
        "recommendations": insights_raw.get("recommendations", [])
        if isinstance(insights_raw, dict) else [],
        "insight_warnings": insights_raw.get("warnings", [])
        if isinstance(insights_raw, dict) else [],
        "evidence": evidence_raw if isinstance(evidence_raw, list) else [],
        "charts": charts_raw.get("charts", [])
        if isinstance(charts_raw, dict) else [],
        "charts_truncated": charts_raw.get("charts_truncated", False)
        if isinstance(charts_raw, dict) else False,
        "business_context": context_raw if isinstance(context_raw, dict)
        else {},
        "data_profile": profile_raw if isinstance(profile_raw, dict) else {},
        "understanding": understanding_raw
        if isinstance(understanding_raw, dict) else {},
        "dq_report": dq_raw if isinstance(dq_raw, dict) else {},
        "cleaning_result": cleaning_raw
        if isinstance(cleaning_raw, dict) else {},
        "run_comparison": _load_json(out / "run_comparison.json")
        if isinstance(_load_json(out / "run_comparison.json"), dict) else {},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(text: Any) -> str:
    """Escape text for safe HTML embedding (defense in depth)."""
    from html import escape
    return escape(str(text))


# ---------------------------------------------------------------------------
# Section renderers (produce HTML fragments)
# ---------------------------------------------------------------------------


# 5.5: business-relevance ranking for the report's headline KPI cards —
# primary measures outrank generic numeric columns, magnitude breaks ties.
_PRIMARY_MEASURE_KEYWORDS = (
    "revenue", "sales", "amount", "total", "profit", "cost", "price",
    "qty", "quantity", "count", "spend", "value", "margin", "volume",
)


def _kpi_relevance(kpi: Dict[str, Any]) -> tuple[float, float]:
    """(relevance, magnitude) sort key — higher relevance first."""
    op = kpi.get("operation") or {}
    column = str(op.get("column") or op.get("column_a") or "").lower()
    name = str(kpi.get("name", "")).lower()
    relevance = 0.0
    if any(kw in column for kw in _PRIMARY_MEASURE_KEYWORDS):
        relevance += 2.0
    if any(kw in name for kw in _PRIMARY_MEASURE_KEYWORDS):
        relevance += 1.0
    if str(op.get("function", "")) in ("sum", "count"):
        relevance += 0.5
    value = kpi.get("value")
    try:
        magnitude = abs(float(value))
    except (TypeError, ValueError):
        magnitude = 0.0
    return relevance, magnitude


def render_kpis(kpis: List[Dict[str, Any]]) -> str:
    """Render top KPI metric cards (up to 4), ranked by business relevance
    (5.5) rather than plan order or statistical variance alone."""
    if not kpis:
        return '<p class="text-secondary">No KPIs computed.</p>'
    parts: List[str] = []
    ranked = sorted(kpis, key=_kpi_relevance, reverse=True)
    for kpi in ranked[:4]:
        val = kpi.get("value")
        formatted = _fmt(float(val)) if val is not None else "N/A"
        name = kpi.get("name", kpi.get("kpi_id", "KPI"))
        parts.append(
            f'<div class="card metric h-100"><div class="card-body">'
            f'<div class="lbl"><span>{_esc(name)}</span></div>'
            f'<div class="val">{_esc(formatted)}</div>'
            f'</div></div>'
        )
    cols = "".join(f'<div class="col-sm-6 col-xl-3">{p}</div>'
                   for p in parts)
    return f'<div class="row g-4">{cols}</div>'


def render_stats(stats: List[Dict[str, Any]]) -> str:
    """Render statistical results as an HTML table."""
    if not stats:
        return '<p class="text-secondary">No statistical tests run.</p>'
    rows: List[str] = []
    for s in stats:
        p_str = (f"{s['p_value']:.4f}"
                 if s.get("p_value") is not None else "—")
        stat_str = (f"{s['statistic']:.4f}"
                    if s.get("statistic") is not None else "—")
        effect = (f"{s['effect_size']:.4f}"
                  if s.get("effect_size") is not None else "—")
        eid = s.get("evidence_id", "")
        rows.append(
            f"<tr><td>{_esc(s.get('test_name', ''))}</td>"
            f"<td>{_esc(', '.join(s.get('variables', [])))}</td>"
            f"<td class='text-end'>{stat_str}</td>"
            f"<td class='text-end'>{p_str}</td>"
            f"<td class='text-end'>{effect}</td>"
            f"<td><code>{_esc(eid)}</code></td></tr>"
        )
    body = "\n".join(rows)
    return (
        '<div class="table-responsive"><table '
        'class="table table-hover align-middle mb-0">'
        '<thead><tr><th>Test</th><th>Variables</th>'
        '<th class="text-end">Statistic</th>'
        '<th class="text-end">p-value</th>'
        '<th class="text-end">Effect size</th>'
        '<th>Evidence</th></tr></thead>'
        f'<tbody>{body}</tbody></table></div>'
    )


# ---------------------------------------------------------------------------
# Charts — interactive (Chart.js canvas) with a static SVG fallback
# ---------------------------------------------------------------------------

# Kinds with a Chart.js mapping. boxplot/heatmap/lollipop (and unknown
# kinds) stay as static SVG images — they would need extra chart.js plugins.
_INTERACTIVE_KINDS = frozenset({
    "bar", "barh", "line", "area", "doughnut", "pie",
    "histogram", "stacked_bar", "scatter",
})


def _load_chart_df(run_dir: Path) -> Any | None:
    """The same cleaned frame the analysis stage rendered the SVGs from."""
    cleaned = Path(run_dir) / "data" / "processed" / "cleaned_data.csv"
    if not cleaned.is_file():
        return None
    try:
        return pd.read_csv(cleaned, encoding="utf-8-sig")
    except Exception:  # noqa: BLE001 -- report must still render
        return None


def _dataset_label(meta: ChartMetadata) -> str:
    return meta.columns[0] if meta.columns else (meta.title or "")


def _empty_scatter_cfg() -> Dict[str, Any]:
    return {
        "type": "scatter",
        "data": {"labels": [], "datasets": [
            {"label": "", "data": [], "backgroundColor": PALETTE[0]}]},
        "options": {"responsive": True, "maintainAspectRatio": False,
                    "plugins": {"legend": {"display": True}}},
    }


def _js_config(chart: Dict[str, Any], df: Any,
               kpi_objs: List[Any]) -> Dict[str, Any] | None:
    """Deterministic Chart.js config for one chart. Aggregation reuses
    chart_renderer's own data preparation so the interactive canvas shows
    exactly what the static SVG shows. Returns None when the kind has no
    mapping or the data is empty (the static SVG is shown instead)."""
    kind = chart.get("kind")
    if kind not in _INTERACTIVE_KINDS:
        return None
    try:
        meta = ChartMetadata(**chart)
    except Exception:  # noqa: BLE001 -- never fail the report
        return None
    prepared = _prepare(meta, df, kpi_objs)
    labels = prepared.get("labels") or []
    values = prepared.get("values") or []
    base_options: Dict[str, Any] = {
        "responsive": True, "maintainAspectRatio": False,
        "plugins": {"legend": {"display": False}},
    }

    if kind == "bar":
        if not labels:
            return None
        return {
            "type": "bar",
            "data": {"labels": labels,
                     "datasets": [{"label": _dataset_label(meta),
                                   "data": values,
                                   "backgroundColor": list(PALETTE)}]},
            "options": {**base_options,
                        "scales": {"y": {"beginAtZero": True}}},
        }
    if kind == "barh":
        if not labels:
            return None
        return {
            "type": "bar",
            "data": {"labels": labels,
                     "datasets": [{"label": _dataset_label(meta),
                                   "data": values,
                                   "backgroundColor": list(PALETTE)}]},
            "options": {**base_options, "indexAxis": "y",
                        "scales": {"x": {"beginAtZero": True}}},
        }
    if kind in ("line", "area"):
        if not labels:
            return None
        filled = kind == "area"
        return {
            "type": "line",
            "data": {"labels": labels,
                     "datasets": [{"label": _dataset_label(meta),
                                   "data": values,
                                   "borderColor": PALETTE[0],
                                   "backgroundColor": PALETTE[0] + "40",
                                   "borderWidth": 2.5,
                                   "pointRadius": 3,
                                   "fill": filled}]},
            "options": {**base_options},
        }
    if kind in ("pie", "doughnut"):
        if not labels:
            return None
        return {
            "type": kind,
            "data": {"labels": labels,
                     "datasets": [{"data": values,
                                   "backgroundColor": list(PALETTE),
                                   "borderColor": "#ffffff",
                                   "borderWidth": 1.5}]},
            "options": {**base_options,
                        "plugins": {"legend": {"display": True}}},
        }
    if kind == "histogram":
        series = prepared.get("series") or []
        if not series:
            return None
        bins = _fd_bins(series)
        counts, edges = _hist(series, bins)
        bin_labels = [f"{_fmt(edges[i])}–{_fmt(edges[i + 1])}"
                      for i in range(len(counts))]
        return {
            "type": "bar",
            "data": {"labels": bin_labels,
                     "datasets": [{"label": meta.columns[0]
                                   if meta.columns else meta.title,
                                   "data": counts,
                                   "backgroundColor": PALETTE[0] + "BF"}]},
            "options": {**base_options,
                        "scales": {"y": {"beginAtZero": True}}},
        }
    if kind == "stacked_bar":
        dimension = meta.columns[0] if meta.columns else None
        measures = [c for c in meta.columns[1:] if c in df.columns]
        if not dimension or dimension not in df.columns or len(measures) < 2:
            return None
        grouped = df.groupby(dimension, dropna=False)
        rows = sorted(grouped, key=lambda pair: str(pair[0]))
        if not rows:
            return None
        datasets = [
            {"label": measure,
             "data": [max(0.0, _to_float(group[measure].sum()) or 0.0)
                      for _, group in rows],
             "backgroundColor": PALETTE[j % len(PALETTE)]}
            for j, measure in enumerate(measures)]
        return {
            "type": "bar",
            "data": {"labels": [str(key) for key, _ in rows],
                     "datasets": datasets},
            "options": {**base_options,
                        "plugins": {"legend": {"display": True}},
                        "scales": {"x": {"stacked": True},
                                   "y": {"stacked": True,
                                         "beginAtZero": True}}},
        }
    if kind == "scatter":
        cols = [c for c in meta.columns if c in df.columns][:2]
        if len(cols) != 2:
            return None
        pairs = [(x, y) for x, y in
                 ((_to_float(a), _to_float(b))
                  for a, b in zip(df[cols[0]], df[cols[1]]))
                 if x is not None and y is not None]
        if len(pairs) < 2:
            return None
        shown = min(len(pairs), 1500)
        idx = [round(i * (len(pairs) - 1) / (shown - 1))
               for i in range(shown)] if shown > 1 else [0]
        pts = [[pairs[i][0], pairs[i][1]] for i in idx]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        slope, intercept = _linreg(xs, ys)
        x0, x1 = min(xs), max(xs)
        cfg = _empty_scatter_cfg()
        cfg["data"]["labels"] = None
        cfg["data"]["datasets"] = [
            {"label": f"{cols[0]} → {cols[1]}", "data": pts,
             "backgroundColor": PALETTE[0], "pointRadius": 3,
             "pointHoverRadius": 5},
            {"label": "trend", "data": [[x0, intercept + slope * x0],
                                        [x1, intercept + slope * x1]],
             "type": "line", "borderColor": PALETTE[1],
             "borderDash": [4, 3], "borderWidth": 2, "pointRadius": 0,
             "fill": False, "showLine": True},
        ]
        cfg["options"]["plugins"]["datalabels"] = False
        return cfg
    return None


def _has_chart_data(cfg: Dict[str, Any]) -> bool:
    for dataset in cfg["data"]["datasets"]:
        if dataset.get("data"):
            return True
    return False


_JS_INIT = """\
<script>
window.__REPORT_CHARTS__ = %s;
(function(){
  function fmt(v){
    if (v === null || v === undefined || typeof v !== 'number' || isNaN(v)) return '';
    if (Number.isInteger(v)) return v.toLocaleString('en-US');
    var a = Math.abs(v);
    if (a >= 1000) return Math.round(v).toLocaleString('en-US');
    if (v === 0) return '0';
    if (a >= 1) return v.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    var s = v.toFixed(3);
    while (s.indexOf('.') >= 0 && s.charAt(s.length - 1) === '0') s = s.slice(0, -1);
    if (s.charAt(s.length - 1) === '.') s = s.slice(0, -1);
    return s;
  }
  function fallback(id){
    var el = document.getElementById('chart-' + id);
    var img = document.getElementById('img-' + id);
    if (el) el.style.display = 'none';
    if (img) img.style.display = '';
  }
  function init(){
    var CFG = window.__REPORT_CHARTS__ || {};
    var hasChart = (typeof Chart !== 'undefined');
    Object.keys(CFG).forEach(function(id){
      if (!hasChart) { fallback(id); return; }
      var el = document.getElementById('chart-' + id);
      if (!el) return;
      var c = CFG[id];
      c.options = c.options || {};
      c.options.plugins = c.options.plugins || {};
      c.options.plugins.tooltip = c.options.plugins.tooltip || {};
      c.options.plugins.tooltip.callbacks = c.options.plugins.tooltip.callbacks || {};
      c.options.plugins.tooltip.callbacks.label = function(ctx){
        var v = ctx.parsed;
        if (v && v.y !== undefined) v = v.y; else if (v && v.x !== undefined) v = v.x;
        var head = ctx.label ? ctx.label + ' \\u00b7 ' : '';
        var name = ctx.dataset && ctx.dataset.label ? ctx.dataset.label + ': ' : '';
        return head + name + fmt(v);
      };
      if (typeof ChartDataLabels !== 'undefined' && c.options.plugins.datalabels !== false) {
        var n = c.data && c.data.labels ? c.data.labels.length : 0;
        if (n > 0 && n <= 12) {
          c.options.plugins.datalabels = c.options.plugins.datalabels || {};
          c.options.plugins.datalabels.formatter = function(v, ctx){
            if (c.type === 'pie' || c.type === 'doughnut') {
              var tot = ctx.dataset.data.reduce(function(a, b){ return a + b; }, 0) || 1;
              return fmt(100 * v / tot) + '%';
            }
            return fmt(v);
          };
          c.options.plugins.datalabels.color = '#333333';
          c.options.plugins.datalabels.anchor = 'end';
          c.options.plugins.datalabels.align = 'end';
          if (c.type === 'pie' || c.type === 'doughnut') {
            c.options.plugins.datalabels.anchor = 'center';
            c.options.plugins.datalabels.align = 'center';
          }
        } else {
          c.options.plugins.datalabels = { display: false };
        }
      }
      try { new Chart(el, c); }
      catch (e) { fallback(id); }
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
</script>"""


def render_charts(charts: List[Dict[str, Any]], run_dir: Path,
                  kpis: List[Dict[str, Any]] | None = None) -> str:
    """Interactive Chart.js figures (canvas) with a static SVG fallback.

    Kinds without a Chart.js mapping (boxplot/heatmap/lollipop), runs where
    the cleaned frame is unavailable, and empty charts stay as static
    <img> — the report never shows a blank figure.
    """
    if not charts:
        return '<p class="text-secondary">No charts generated.</p>'
    try:
        kpi_objs = [KpiResult(**k) for k in (kpis or [])]
    except Exception:  # noqa: BLE001 -- missing kpis never kill the report
        kpi_objs = []
    df = _load_chart_df(run_dir)
    figures: List[str] = []
    configs: Dict[str, Any] = {}
    charts_dir = run_dir / "outputs" / "charts"
    for ch in charts:
        chart_id = ch.get("chart_id", "")
        path = ch.get("chart_path")
        if not path:
            path = f"outputs/charts/{chart_id}.svg"
        full = run_dir / path
        if not full.exists():
            full = charts_dir / f"{chart_id}.svg"
        # src is relative to the run dir so the report is portable:
        # it works from disk and when served over HTTP
        try:
            src = full.resolve().relative_to(run_dir.resolve()).as_posix()
        except ValueError:
            src = str(path).replace("\\", "/")
        title = ch.get("title", chart_id)
        reliability = ch.get("reliability", "")
        eid = ch.get("evidence_id", "")
        caption = _esc(title)
        if reliability:
            caption += f" ({_esc(reliability)})"
        if eid:
            caption += f" — {_esc(eid)}"
        cfg = _js_config(ch, df, kpi_objs) if df is not None else None
        if cfg is not None and _has_chart_data(cfg):
            configs[chart_id] = cfg
            figures.append(
                f'<figure class="mb-3 chart-figure">'
                f'<canvas id="chart-{_esc(chart_id)}" class="chart-wrap" '
                f'role="img" aria-label="{_esc(title)}"></canvas>'
                f'<img id="img-{_esc(chart_id)}" src="{_esc(src)}" '
                f'alt="{_esc(title)}" class="chart-static img-fluid rounded" '
                f'loading="lazy" />'
                f'<figcaption>{caption}</figcaption>'
                f'</figure>'
            )
        else:
            figures.append(
                f'<figure class="mb-3">'
                f'<img src="{_esc(src)}" alt="{_esc(title)}" '
                f'class="img-fluid rounded" loading="lazy" />'
                f'<figcaption>{caption}</figcaption>'
                f'</figure>'
            )
    if not configs:
        return "\n".join(figures)
    # JSON is embedded inside <script> — escape "<" so a label can never
    # break out of the script element (</script> / <!-- attacks).
    payload = json.dumps(configs).replace("<", "\\u003c")
    return "\n".join(figures) + "\n" + _JS_INIT.replace("%s", payload, 1)


def render_insights(insights: List[Dict[str, Any]]) -> str:
    """Render insight cards with confidence badges."""
    if not insights:
        return '<p class="text-secondary">No insights generated.</p>'
    cards: List[str] = []
    for ins in insights:
        conf = ins.get("confidence", "medium")
        badge_cls = {"high": "c-high", "medium": "c-med",
                     "low": "c-low"}.get(conf, "c-med")
        title = ins.get("title", "")
        desc = ins.get("description", "")
        claim = ins.get("claim_type", "")
        cards.append(
            f'<div class="col-lg-6"><div class="card h-100">'
            f'<div class="card-body">'
            f'<div class="d-flex justify-content-between align-items-start">'
            f'<h5 class="card-title fs-6">{_esc(title)}</h5>'
            f'<span class="badge {badge_cls}" style="font-size:.66rem;">'
            f'{_esc(conf.title())}</span></div>'
            f'<p class="card-text small">{_esc(desc)}</p>'
            f'<p class="text-secondary small fst-italic mb-0">'
            f'Claim: {_esc(claim)}</p>'
            f'</div></div></div>'
        )
    grid = "\n".join(cards)
    return f'<div class="row g-4">{grid}</div>'


def render_recommendations(recs: List[Dict[str, Any]]) -> str:
    """Render recommendations as a table."""
    if not recs:
        return '<p class="text-secondary">No recommendations.</p>'
    rows: List[str] = []
    for i, rec in enumerate(recs, 1):
        desc = rec.get("description", "")
        basis = rec.get("basis", "")
        impact = rec.get("potential_impact", "")
        rows.append(
            f"<tr><td class='ps-4 fw-semibold' style='color:var(--navy2);'>"
            f"{i}</td><td>{_esc(desc)}</td>"
            f"<td class='text-secondary small'>{_esc(basis)}</td>"
            f"<td class='fw-semibold text-end pe-4'>"
            f"{_esc(impact)}</td></tr>"
        )
    body = "\n".join(rows)
    return (
        '<div class="card"><div class="table-responsive">'
        '<table class="table table-hover align-middle mb-0">'
        '<thead><tr><th class="ps-4">#</th><th>Recommendation</th>'
        '<th>Basis</th>'
        '<th class="text-end pe-4">Potential impact</th>'
        f'</tr></thead><tbody>{body}</tbody></table></div></div>'
    )


def render_evidence(evidence: List[Dict[str, Any]]) -> str:
    """Render evidence appendix table."""
    if not evidence:
        return '<p class="text-secondary">No evidence recorded.</p>'
    rows: List[str] = []
    for ev in evidence[:50]:
        eid = ev.get("evidence_id", "")
        source = ev.get("source", {})
        agg = source.get("aggregation", "")
        lineage = ", ".join(source.get("lineage", []))
        rows.append(
            f"<tr><td><code>{_esc(eid)}</code></td>"
            f"<td>{_esc(agg)}</td>"
            f"<td class='text-secondary small'>{_esc(lineage)}</td></tr>"
        )
    body = "\n".join(rows)
    return (
        '<div class="table-responsive">'
        '<table class="table table-hover align-middle mb-0">'
        '<thead><tr><th>Evidence ID</th><th>Aggregation</th>'
        f'<th>Lineage</th></tr></thead><tbody>{body}</tbody></table></div>'
    )


def render_run_comparison(cmp: Dict[str, Any]) -> str:
    """"vs previous run" callout (§8): KPI deltas against the last cached
    run of the same source file. Numbers only — Python-computed."""
    if not cmp or not cmp.get("compared_kpis"):
        return ""
    prev_id = cmp.get("previous_run_id", "")
    rows = []
    for k in cmp["compared_kpis"][:15]:
        cur = k.get("current")
        prev = k.get("previous")
        if not isinstance(cur, (int, float)) or not isinstance(
                prev, (int, float)):
            continue
        if prev == 0:
            delta = ""
        else:
            pct = (cur - prev) / abs(prev) * 100
            arrow = "▲" if pct > 0 else "▼"
            delta = f"{arrow} {abs(pct):.1f}%"
        rows.append(
            f"<tr><td>{_esc(str(k.get('name') or k.get('kpi_id')))}</td>"
            f"<td>{_fmt(cur)}</td><td>{_fmt(prev)}</td>"
            f"<td>{delta}</td></tr>"
        )
    if not rows:
        return ""
    body = "\n".join(rows)
    return (
        '<div class="card mb-4 shadow-sm">'
        '<div class="card-header fw-semibold">'
        '📊 vs Previous Run</div>'
        '<div class="card-body">'
        '<p class="text-secondary small">Compared against the most recent '
        f"cached run of the same source file"
        f"{f' (<code>{_esc(prev_id)}</code>)' if prev_id else ''}.</p>"
        '<div class="table-responsive"><table class="table table-hover '
        'align-middle mb-0"><thead><tr><th>KPI</th><th>Current</th>'
        f'<th>Previous</th><th>Change</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div></div></div>"
    )


def render_business_context(ctx: Dict[str, Any]) -> str:
    """Render business context summary."""
    if not ctx:
        return '<p class="text-secondary">No business context.</p>'
    parts: List[str] = []
    file_name = ctx.get("file_name", "")
    goal = ctx.get("goal_summary", "")
    confidence = ctx.get("context_confidence", 0)
    generic = ctx.get("generic_mode", False)
    if file_name:
        parts.append(f"<p><strong>File:</strong> {_esc(file_name)}</p>")
    if goal:
        parts.append(f"<p><strong>Goal:</strong> {_esc(goal)}</p>")
    parts.append(
        f"<p><strong>Context confidence:</strong> "
        f"{_fmt(confidence) if confidence else 'N/A'}</p>"
    )
    if generic:
        parts.append(
            '<p class="text-warning"><em>Generic mode — limited business '
            'context gathered.</em></p>'
        )
    answers = ctx.get("answers", {})
    if answers:
        rows = []
        for q, a in answers.items():
            if a:
                rows.append(
                    f"<tr><td>{_esc(q)}</td><td>{_esc(a)}</td></tr>"
                )
        if rows:
            parts.append(
                '<div class="table-responsive mt-2">'
                '<table class="table table-sm">'
                '<thead><tr><th>Question</th><th>Answer</th></tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>'
            )
    return "\n".join(parts)


def render_dq_summary(dq: Dict[str, Any]) -> str:
    """Render data quality summary."""
    if not dq:
        return '<p class="text-secondary">No DQ report available.</p>'
    status = dq.get("status", "unknown")
    status_cls = ("text-success" if status == "passed"
                  else "text-warning")
    parts: List[str] = [
        f"<p><strong>Status:</strong> "
        f"<span class='{status_cls} fw-bold'>{_esc(status)}</span></p>"
    ]
    dupes = dq.get("duplicates", 0)
    if dupes:
        parts.append(f"<p>Duplicates found: {dupes}</p>")
    invalid = dq.get("invalid", {})
    if invalid:
        issues = []
        for col, kinds in invalid.items():
            issues.append(
                f"{_esc(col)}: {', '.join(str(k) for k in kinds)}"
            )
        parts.append(
            "<p><strong>Invalid values:</strong></p><ul>"
            + "".join(f"<li>{i}</li>" for i in issues)
            + "</ul>"
        )
    missingness = dq.get("missingness", {})
    if missingness:
        overall = missingness.get("overall_rate", 0)
        if overall > 0:
            parts.append(
                f"<p>Overall missingness: {_fmt(overall * 100)}%</p>"
            )
    issues_list = dq.get("issues", [])
    if issues_list:
        parts.append("<p><strong>Issues:</strong></p><ul>")
        for issue in issues_list[:10]:
            parts.append(f"<li>{_esc(str(issue))}</li>")
        parts.append("</ul>")
    return "\n".join(parts)


def render_overview(profile: Dict[str, Any],
                    understanding: Dict[str, Any]) -> str:
    """Render data overview (rows, columns, roles, domain)."""
    parts: List[str] = []
    rows = profile.get("row_count") or understanding.get("row_count")
    cols = profile.get("column_count") or understanding.get("column_count")
    if rows is not None:
        parts.append(f"<p><strong>Rows:</strong> {_fmt(rows)}</p>")
    if cols is not None:
        parts.append(f"<p><strong>Columns:</strong> {cols}</p>")
    domain = understanding.get("domain", "")
    if domain:
        parts.append(f"<p><strong>Domain:</strong> {_esc(domain)}</p>")
    entities = understanding.get("entities", [])
    if entities:
        parts.append(
            f"<p><strong>Entities:</strong> "
            f"{_esc(', '.join(str(e) for e in entities))}</p>"
        )
    columns = understanding.get("columns", [])
    if columns:
        parts.append("<p><strong>Column roles:</strong></p>")
        role_rows = []
        for col in columns[:30]:
            if isinstance(col, dict):
                name = col.get("name", "")
                role = col.get("role", "")
            else:
                name = str(col)
                role = ""
            role_rows.append(
                f"<tr><td>{_esc(name)}</td><td>{_esc(role)}</td></tr>"
            )
        parts.append(
            '<div class="table-responsive">'
            '<table class="table table-sm">'
            '<thead><tr><th>Column</th><th>Role</th></tr></thead>'
            f'<tbody>{"".join(role_rows)}</tbody></table></div>'
        )
    if not parts:
        return '<p class="text-secondary">No data overview.</p>'
    return "\n".join(parts)


def render_limitations(understanding: Dict[str, Any],
                       ctx: Dict[str, Any]) -> str:
    """Render limitations from context confidence and plan."""
    parts: List[str] = []
    confidence = ctx.get("context_confidence", 0)
    if confidence and confidence < 0.5:
        parts.append(
            "<li>Low business context confidence "
            f"({_fmt(confidence)}). "
            "Interpretations may miss domain nuances.</li>"
        )
    limitations = understanding.get("limitations", [])
    for lim in limitations:
        parts.append(f"<li>{_esc(lim)}</li>")
    if not parts:
        parts.append("<li>No significant limitations identified.</li>")
    return "<ul>" + "\n".join(parts) + "</ul>"


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def _build_context(run_dir: Path, exec_summary: str,
                   locale: str) -> Dict[str, Any]:
    """Build the full Jinja2 context dict from loaded artifacts."""
    from datetime import date as _date

    arts = load_artifacts(run_dir)
    ctx = arts["business_context"]

    # --- Bug fix 0a: wrap exec summary in the exec-panel card ---
    if exec_summary.strip():
        exec_summary_html = (
            '<div class="card exec-panel mb-4"><div class="card-body">'
            '<div class="d-flex justify-content-between align-items-start">'
            '<h5 class="card-title text-white mb-2">Executive Summary</h5>'
            '<span class="badge">Summary</span></div>'
            f'<p class="mb-0">{_esc(exec_summary)}</p>'
            '</div></div>'
        )
    else:
        exec_summary_html = (
            '<p class="text-secondary fst-italic">'
            'No executive summary provided.</p>'
        )

    # --- Bug fix 0b: masthead metadata from business_context ---
    report_title = (
        ctx.get("report_title")
        or ctx.get("goal_summary")
        or ctx.get("objective")
        or "Insight Forge Report"
    )
    prepared_for = ctx.get("prepared_for") or "Executive Team"
    report_date = ctx.get("report_date") or _date.today().isoformat()

    return {
        "run_id": run_dir.name,
        "exec_summary": exec_summary,
        "exec_summary_html": exec_summary_html,
        "locale": locale,
        "report_title": _esc(report_title),
        "report_subtitle": _esc(ctx.get("report_subtitle", "")),
        "prepared_for": _esc(prepared_for),
        "report_date": _esc(report_date),
        "kpis_html": render_kpis(arts["kpis"]),
        "stats_html": render_stats(arts["stats"]),
        "charts_html": render_charts(arts["charts"], run_dir,
                                     arts["kpis"]),
        "insights_html": render_insights(arts["insights"]),
        "recommendations_html": render_recommendations(
            arts["recommendations"]),
        "evidence_html": render_evidence(arts["evidence"]),
        "business_context_html": render_business_context(
            arts["business_context"]),
        "dq_summary_html": render_dq_summary(arts["dq_report"]),
        "overview_html": render_overview(arts["data_profile"],
                                         arts["understanding"]),
        "limitations_html": render_limitations(arts["understanding"],
                                               arts["business_context"]),
        "run_comparison_html": render_run_comparison(
            arts["run_comparison"]),
        "kpis": arts["kpis"],
        "stats": arts["stats"],
        "charts": arts["charts"],
        "insights": arts["insights"],
        "recommendations": arts["recommendations"],
        "evidence": arts["evidence"],
        "charts_truncated": arts["charts_truncated"],
    }


def render_report(run_dir: str | Path,
                  exec_summary: str = "",
                  locale: str = "en",
                  template_path: str | Path | None = None) -> str:
    """Render the full HTML report from run artifacts.

    Parameters
    ----------
    run_dir : path to the run directory
    exec_summary : 3-5 sentence executive summary (LLM-written or empty)
    locale : locale code (default ``"en"``)
    template_path : override template file (for testing)

    Returns
    -------
    Complete HTML string ready to write to report.html.
    """
    run_dir = Path(run_dir)
    ctx = _build_context(run_dir, exec_summary, locale)

    if template_path is not None:
        tpl_path = Path(template_path)
        env = Environment(
            autoescape=select_autoescape(["html"]),
            loader=FileSystemLoader(str(tpl_path.parent)),
        )
        tpl = env.get_template(tpl_path.name)
    else:
        env = Environment(
            autoescape=select_autoescape(["html"]),
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        )
        tpl = env.get_template(_TEMPLATE_NAME)

    return tpl.render(**ctx)


def save_report(run_dir: str | Path, html: str) -> Path:
    """Write report.html into the run root."""
    run_dir = Path(run_dir)
    report_path = run_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def save_report_result(run_dir: str | Path, status: str,
                       report_path: str | None = None,
                       locale: str = "en",
                       sections: List[str] | None = None,
                       error: str | None = None) -> None:
    """Write metadata/report_result.json."""
    run_dir = Path(run_dir)
    meta_dir = run_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": status,
        "report_path": report_path,
        "locale": locale,
        "sections": sections or _SECTIONS,
        "error": error,
    }
    (meta_dir / "report_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="report_builder",
        description="Render Insight Forge HTML report from run artifacts.",
    )
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument("--locale", default="en", help="Report locale")
    parser.add_argument("--summary", default="",
                        help="Executive summary text (or empty)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist", flush=True)
        return 1

    html = render_report(run_dir, exec_summary=args.summary,
                         locale=args.locale)
    report_path = save_report(run_dir, html)
    save_report_result(run_dir, "rendered", str(report_path),
                       args.locale)
    print(f"Report written to {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
