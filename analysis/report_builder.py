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

from shared.formatting import fmt as _fmt

log = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "resources"
_TEMPLATE_NAME = "report_template.html"
_SECTIONS = [
    "executive_summary", "business_context", "dq_summary",
    "data_overview", "kpis", "stats", "charts",
    "insights", "recommendations", "limitations", "evidence",
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


def render_kpis(kpis: List[Dict[str, Any]]) -> str:
    """Render top KPI metric cards (up to 4)."""
    if not kpis:
        return '<p class="text-secondary">No KPIs computed.</p>'
    parts: List[str] = []
    for kpi in kpis[:4]:
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


def render_charts(charts: List[Dict[str, Any]], run_dir: Path) -> str:
    """Embed each chart as an <img> with a caption."""
    if not charts:
        return '<p class="text-secondary">No charts generated.</p>'
    figures: List[str] = []
    charts_dir = run_dir / "outputs" / "charts"
    for ch in charts:
        chart_id = ch.get("chart_id", "")
        path = ch.get("chart_path")
        if not path:
            path = f"outputs/charts/{chart_id}.svg"
        full = run_dir / path
        if not full.exists():
            full = charts_dir / f"{chart_id}.svg"
        src = str(path).replace("\\", "/")
        title = ch.get("title", chart_id)
        reliability = ch.get("reliability", "")
        eid = ch.get("evidence_id", "")
        caption = _esc(title)
        if reliability:
            caption += f" ({_esc(reliability)})"
        if eid:
            caption += f" — {_esc(eid)}"
        figures.append(
            f'<figure class="mb-3">'
            f'<img src="{_esc(src)}" alt="{_esc(title)}" '
            f'class="img-fluid rounded" loading="lazy" />'
            f'<figcaption>{caption}</figcaption>'
            f'</figure>'
        )
    return "\n".join(figures)


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
        "charts_html": render_charts(arts["charts"], run_dir),
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
