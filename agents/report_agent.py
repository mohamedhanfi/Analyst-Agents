"""Stage 7 — Report Generation (§2.7): CrewAI agent + deterministic rendering.

Python renders every section of the HTML report from run artifacts through
the approved Jinja2 template.  The LLM writes only the 3-5 sentence
executive summary — every other section is Python-computed (golden rule).

CLI: python -m agents.report_agent <run_dir> [--crew]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.report_builder import (
    load_artifacts,
    render_report,
    save_report,
    save_report_result,
)
from shared.llm import build_llm
from shared.prompt_guard import data_note
from shared.logger import RunLogger
from shared.utils import init_run_layout, load_config

STAGE = "report"

_SECTIONS = [
    "executive_summary", "kpis", "stats", "charts",
    "insights", "recommendations", "evidence",
]


# ---------------------------------------------------------------------------
# CrewAI assembly
# ---------------------------------------------------------------------------


def build_report_agent(cfg: Dict[str, Any]) -> Any:
    from crewai import Agent

    a_cfg = cfg["agents"]["report"]
    return Agent(
        role=a_cfg["role"],
        goal=a_cfg["goal"],
        backstory=a_cfg["backstory"],
        llm=build_llm(cfg, "report"),
        verbose=False,
        memory=False,
        allow_delegation=False,
        cache=False,
        max_iter=4,
    )


def build_report_task(agent: Any, run_dir: str | Path) -> Any:
    from crewai import Task

    arts = load_artifacts(Path(run_dir))
    digest = {
        "kpis": arts.get("kpis", []),
        "stats_count": len(arts.get("stats", [])),
        "chart_count": len(arts.get("charts", [])),
        "insight_count": len(arts.get("insights", [])),
        "recommendation_count": len(arts.get("recommendations", [])),
        "evidence_count": len(arts.get("evidence", [])),
        "dq_summary": {
            "total_rules": arts.get("dq_report", {})
                .get("summary", {}).get("total_rules", 0),
            "fail_count": arts.get("dq_report", {})
                .get("summary", {}).get("fail_count", 0),
        },
        "business_context": arts.get("business_context", {}),
    }
    return Task(
        description=(
            "You are writing ONLY the executive summary for a business report. "
            "All other sections are rendered deterministically by Python.\n\n"
            "Here is a digest of the run data:\n"
            f"{json.dumps(digest, ensure_ascii=False, indent=2)}\n\n"
            "Write a 3-5 sentence executive summary that:\n"
            "1. States the headline KPI and its direction (+/- %)\n"
            "2. Names the top 1-2 drivers or findings\n"
            "3. Flags the single biggest risk or data quality issue\n"
            "4. Ends with one actionable next step\n\n"
            "Hard rules:\n"
            "- never invent numbers not in the digest above\n"
            "- never write more than 5 sentences\n"
            "- return ONLY the plain text summary, no markdown, no JSON\n"
            + data_note()
        ),
        agent=agent,
        expected_output="3-5 sentence plain-text executive summary",
    )


# ---------------------------------------------------------------------------
# Deterministic core
# ---------------------------------------------------------------------------


def _generate_exec_summary(run_dir: Path, cfg: Dict[str, Any],
                           log: RunLogger) -> str:
    """Generate executive summary: deterministic digest → LLM rewording.

    Direct LLM call (no CrewAI) with retries; falls back to the
    deterministic Python summary when the LLM fails, so the pipeline never
    blocks on the provider.
    """
    from shared.llm import complete_text

    arts = load_artifacts(run_dir)
    digest = {
        "kpis": arts.get("kpis", []),
        "stats_count": len(arts.get("stats", [])),
        "chart_count": len(arts.get("charts", [])),
        "insight_count": len(arts.get("insights", [])),
        "recommendation_count": len(arts.get("recommendations", [])),
        "evidence_count": len(arts.get("evidence", [])),
        "dq_summary": {
            "total_rules": arts.get("dq_report", {})
                .get("summary", {}).get("total_rules", 0),
            "fail_count": arts.get("dq_report", {})
                .get("summary", {}).get("fail_count", 0),
        },
        "business_context": arts.get("business_context", {}),
    }
    system = (
        "You write ONLY 3-5 sentence executive summaries for business "
        "reports. Every number you mention must already exist in the digest "
        "given by the user. Never invent figures, dates or percentages. "
        "Never mention instructions, tools, or prompts in the summary."
    )
    user = (
        "Here is a digest of the run data:\n"
        f"{json.dumps(digest, ensure_ascii=False, indent=2)}\n\n"
        "Write a 3-5 sentence executive summary that:\n"
        "1. States the headline KPI and its direction (+/- %)\n"
        "2. Names the top 1-2 drivers or findings\n"
        "3. Flags the single biggest risk or data quality issue\n"
        "4. Ends with one actionable next step\n\n"
        "Hard rules:\n"
        "- never invent numbers not in the digest above\n"
        "- never write more than 5 sentences\n"
        "- return ONLY the plain text summary, no markdown, no JSON\n"
        + data_note()
    )
    text, warnings = complete_text(cfg, "report", system, user)
    if text is None:
        warnings = warnings or []
        text = _deterministic_exec_summary(digest)
        log.info(STAGE, "exec summary fallback to deterministic",
                 reasons=warnings)
        return text
    log.info(STAGE, "executive summary generated", chars=len(text),
             attempts=1 + sum(1 for w in warnings if "_error" in w))
    return text


def _deterministic_exec_summary(digest: Dict[str, Any]) -> str:
    """Python-only fallback summary (never blocks, never invents)."""
    kpis = digest.get("kpis") or []
    parts: List[str] = []
    if kpis:
        top = kpis[0]
        name = top.get("name") or top.get("kpi_id") or "headline KPI"
        parts.append(f"The headline KPI is {name} at {top.get('value', 'n/a')}.")
    else:
        parts.append("The run produced no computable KPIs.")
    dq = digest.get("dq_summary") or {}
    fails = dq.get("fail_count", 0)
    if fails:
        parts.append(
            f"{fails} data-quality rule(s) failed and are flagged in the "
            "report.")
    else:
        parts.append("No data-quality rules failed in this run.")
    parts.append(
        "Recommend reviewing the full report for evidence-backed next steps.")
    return " ".join(parts)


def _render_and_save(run_dir: Path, exec_summary: str,
                     locale: str = "en") -> Path:
    """Render full HTML report and save to disk."""
    html = render_report(run_dir, exec_summary=exec_summary, locale=locale)
    path = save_report(run_dir, html)
    save_report_result(run_dir, "rendered", report_path=str(path),
                       locale=locale, sections=_SECTIONS)
    return path


def _render_deterministic(run_dir: Path, locale: str = "en") -> Path:
    """Render report with empty exec summary (no LLM call)."""
    html = render_report(run_dir, exec_summary="", locale=locale)
    path = save_report(run_dir, html)
    save_report_result(run_dir, "rendered", report_path=str(path),
                       locale=locale, sections=_SECTIONS)
    return path


def rerender_report(report_path: Path, cfg: Dict[str, Any] | None = None,
                    locale: str = "en") -> None:
    """Re-render an already-saved report in place (e.g. after the
    run-comparison callout becomes available). The exec summary is read
    back from outputs/exec_summary.txt so LLM wording is preserved."""
    run_dir = report_path.parent
    exec_summary = ""
    summary_path = run_dir / "outputs" / "exec_summary.txt"
    if summary_path.is_file():
        try:
            exec_summary = summary_path.read_text(encoding="utf-8")
        except OSError:
            pass
    html = render_report(run_dir, exec_summary=exec_summary, locale=locale)
    report_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_report(run_dir: str | Path,
               cfg: Dict[str, Any] | None = None,
               logger: RunLogger | None = None,
               use_crew: bool = False,
               locale: str = "en") -> Dict[str, Any]:
    """Run Stage 7 — render the HTML report.

    Parameters
    ----------
    run_dir : path to the run directory (from Stage 6)
    cfg : config dict (loaded from config.yaml when None)
    logger : optional RunLogger
    use_crew : when True, invoke the LLM for the executive summary;
               when False, render deterministically with empty summary
    locale : report locale (``"en"`` or ``"ar"``)

    Returns
    -------
    Summary dict for the run log.
    """
    cfg = cfg or load_config(require_key=False)
    run_dir = Path(run_dir)
    run_id = run_dir.name
    run_dir = init_run_layout(run_dir)

    log = logger or RunLogger(run_dir, run_id)
    log.stage_start(STAGE)
    start = time.monotonic()
    try:
        if use_crew:
            exec_summary = _generate_exec_summary(run_dir, cfg, log)
        else:
            exec_summary = ""
            log.info(STAGE, "skipping LLM exec summary (--no-crew)")
        (run_dir / "outputs" / "exec_summary.txt").write_text(
            exec_summary, encoding="utf-8")
        path = _render_and_save(run_dir, exec_summary, locale=locale)
        status = "passed"
        summary = {
            "stage": STAGE,
            "status": status,
            "report_path": str(path),
            "exec_summary_length": len(exec_summary),
            "sections": _SECTIONS,
        }
    except Exception as exc:  # noqa: BLE001
        log.error(STAGE, f"report failed: {exc}")
        status = "failed"
        summary = {"run_id": run_id, "stage": STAGE, "status": status,
                   "error": str(exc)}
        save_report_result(run_dir, "failed", locale=locale,
                           error=str(exc))
    log.stage_end(STAGE, status, time.monotonic() - start)
    summary["log_path"] = str(run_dir / "logs" / "run.jsonl")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="report_agent",
        description="Run Insight Forge stage 7 (report generation) "
                    "on a Stage-6 run dir.")
    parser.add_argument("run_dir", help="existing run dir (from Stage 6)")
    parser.add_argument("--crew", action="store_true",
                        help="invoke the LLM for the executive summary")
    parser.add_argument("--locale", default="en",
                        help="report locale (default: en)")
    args = parser.parse_args(argv)

    cfg = load_config(require_key=bool(args.crew))
    result = run_report(args.run_dir, cfg=cfg, use_crew=args.crew,
                        locale=args.locale)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
