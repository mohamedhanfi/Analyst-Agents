"""CrewAI @tool wrappers — Stage 7 Report Generation (§2.7).

Wraps the deterministic report-builder functions.  Every tool is a thin
JSON-in/JSON-out adapter around pure Python rendering; the LLM never
touches HTML directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crewai.tools import tool

from analysis.report_builder import (
    _build_context,
    load_artifacts,
    render_charts,
    render_evidence,
    render_insights,
    render_kpis,
    render_recommendations,
    render_report,
    render_stats,
    save_report,
    save_report_result,
)


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


# ------------------------------------------------------------------
# Tool 1 — Load artifacts
# ------------------------------------------------------------------


@tool("load_report_artifacts_tool")
def load_report_artifacts_tool(run_dir: str) -> str:
    """Load all JSON artefacts from a run directory and return them as a
    JSON blob.  Call this first to understand what data is available
    before rendering any section.

    Returns a dict with keys: kpis, stats, insights, recommendations,
    evidence, charts, charts_truncated, business_context, data_profile,
    understanding, dq_report, cleaning_result.
    """
    arts = load_artifacts(Path(run_dir))
    return _json(arts)


# ------------------------------------------------------------------
# Tool 2 — Render a single section
# ------------------------------------------------------------------


def _render_exec_summary(run_dir: str) -> str:
    """Return the executive summary HTML from the run's context."""
    ctx = _build_context(Path(run_dir))
    return ctx.get("exec_summary", "") or (
        '<p class="text-secondary">No executive summary provided.</p>')


_SECTION_RENDERERS = {
    "executive_summary": lambda ctx, run_dir: _render_exec_summary(run_dir),
    "kpis": lambda ctx, run_dir: render_kpis(ctx.get("kpis", [])),
    "stats": lambda ctx, run_dir: render_stats(ctx.get("stats", [])),
    "charts": lambda ctx, run_dir: render_charts(
        ctx.get("charts", []), Path(run_dir)),
    "insights": lambda ctx, run_dir: render_insights(
        ctx.get("insights", [])),
    "recommendations": lambda ctx, run_dir: render_recommendations(
        ctx.get("recommendations", [])),
    "evidence": lambda ctx, run_dir: render_evidence(
        ctx.get("evidence", [])),
}


@tool("render_report_section_tool")
def render_report_section_tool(section_id: str, run_dir: str) -> str:
    """Render a single report section by id and return HTML.

    Valid section ids:
      executive_summary, kpis, stats, charts, insights,
      recommendations, evidence.

    Returns the HTML fragment for that section.
    """
    if section_id not in _SECTION_RENDERERS:
        return json.dumps(
            {"error": f"unknown section_id {section_id!r}",
             "valid": sorted(_SECTION_RENDERERS.keys())},
            ensure_ascii=False, indent=2)
    ctx = _build_context(Path(run_dir))
    html = _SECTION_RENDERERS[section_id](ctx, run_dir)
    return _json({"section_id": section_id, "html": html})


# ------------------------------------------------------------------
# Tool 3 — Render full report
# ------------------------------------------------------------------


@tool("render_full_report_tool")
def render_full_report_tool(run_dir: str, locale: str = "en") -> str:
    """Render the full HTML report from run artifacts.

    Returns a JSON object with keys: html (the full HTML string),
    length (character count), sections (list of section ids rendered).
    """
    html = render_report(Path(run_dir), locale=locale)
    return _json({
        "html": html,
        "length": len(html),
        "sections": [
            "executive_summary", "kpis", "stats", "charts",
            "insights", "recommendations", "evidence",
        ],
    })


# ------------------------------------------------------------------
# Tool 4 — Save report
# ------------------------------------------------------------------


@tool("save_report_tool")
def save_report_tool(run_dir: str, html: str) -> str:
    """Write the HTML report to ``<run_dir>/report.html`` and record
    metadata/report_result.json.  Returns a JSON object with keys:
    report_path, status, sections.
    """
    path = save_report(Path(run_dir), html)
    save_report_result(Path(run_dir), "rendered",
                       report_path=str(path))
    return _json({
        "report_path": str(path),
        "status": "rendered",
        "sections": [
            "executive_summary", "kpis", "stats", "charts",
            "insights", "recommendations", "evidence",
        ],
    })
