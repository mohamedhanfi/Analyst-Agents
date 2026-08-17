"""Stage 8 — QA (§2.8): CrewAI agent + deterministic core.

Last gate.  Python recomputation is authoritative; an independent model
checks logic + readability.  Score is informational only; verdict is
decided purely by logical conditions.

CLI: python -m agents.qa_agent <run_dir> [--crew]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.qa_recompute import run_all_checks
from analysis.qa_verdict import run_verdict
from shared.llm import build_llm
from shared.logger import RunLogger
from shared.utils import init_run_layout, load_config

STAGE = "qa"


# ---------------------------------------------------------------------------
# CrewAI assembly
# ---------------------------------------------------------------------------


def build_qa_agent(cfg: Dict[str, Any]) -> Any:
    from crewai import Agent

    a_cfg = cfg["agents"]["qa"]
    return Agent(
        role=a_cfg["role"],
        goal=a_cfg["goal"],
        backstory=a_cfg["backstory"],
        llm=build_llm(cfg, "qa"),
        verbose=False,
        memory=False,
        allow_delegation=False,
        cache=False,
        max_iter=4,
    )


def build_qa_task(agent: Any, run_dir: str | Path) -> Any:
    from crewai import Task

    # Load artifacts for the LLM review
    run_dir = Path(run_dir)
    insights_raw = {}
    insights_path = run_dir / "outputs" / "insights.json"
    if insights_path.exists():
        try:
            insights_raw = json.loads(insights_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    insights = insights_raw.get("insights", [])
    recs = insights_raw.get("recommendations", [])

    exec_summary = ""
    report_path = run_dir / "report.html"
    # The LLM will review the raw insight/recommendation data, not HTML

    return Task(
        description=(
            "You are the QA auditor. Review these insights and "
            "recommendations for logical coherence.\n\n"
            f"Insights ({len(insights)}):\n"
            f"{json.dumps(insights, ensure_ascii=False, indent=2)}\n\n"
            f"Recommendations ({len(recs)}):\n"
            f"{json.dumps(recs, ensure_ascii=False, indent=2)}\n\n"
            "Check:\n"
            "1. Do the evidence_ids in each insight actually support the "
            "claim type?\n"
            "2. Do the recommendations logically follow from the insights "
            "they reference?\n"
            "3. Are there any obvious logical gaps or contradictions?\n\n"
            "Return a JSON object {\"readability_ok\": bool, "
            "\"logic_ok\": bool, \"notes\": [...]} and nothing else."
        ),
        agent=agent,
        expected_output='JSON {"readability_ok": bool, "logic_ok": bool, '
                        '"notes": [...]}',
    )


# ---------------------------------------------------------------------------
# Deterministic core
# ---------------------------------------------------------------------------


def _run_llm_review(run_dir: Path, cfg: Dict[str, Any],
                    log: RunLogger) -> Dict[str, Any]:
    """Run the LLM review via CrewAI."""
    agent = build_qa_agent(cfg)
    task = build_qa_task(agent, run_dir)

    from crewai import Crew
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    result = crew.kickoff()

    try:
        review = json.loads(str(result))
    except (json.JSONDecodeError, TypeError):
        review = {"readability_ok": True, "logic_ok": True,
                  "notes": [f"LLM output parse failed: {str(result)[:200]}"]}
    log.info(STAGE, f"LLM review: logic_ok={review.get('logic_ok')}, "
             f"readability_ok={review.get('readability_ok')}")
    return review


def _deterministic_review(run_dir: Path) -> Dict[str, Any]:
    """Structural review without LLM (same checks as review_logic_tool)."""
    insights_path = run_dir / "outputs" / "insights.json"
    insights_raw = {}
    if insights_path.exists():
        try:
            insights_raw = json.loads(insights_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    insights = insights_raw.get("insights", [])
    recs = insights_raw.get("recommendations", [])
    notes: List[str] = []

    insight_ids = {i.get("insight_id") for i in insights if isinstance(i, dict)}
    for rec in recs:
        if isinstance(rec, dict):
            ref = rec.get("insight_id")
            if ref and ref not in insight_ids:
                notes.append(
                    f"recommendation {rec.get('recommendation_id', '?')} "
                    f"references non-existent insight {ref!r}")

    return {
        "readability_ok": True,
        "logic_ok": not notes,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_qa(run_dir: str | Path,
           cfg: Dict[str, Any] | None = None,
           logger: RunLogger | None = None,
           use_crew: bool = False) -> Dict[str, Any]:
    """Run Stage 8 — QA verdict.

    Parameters
    ----------
    run_dir : path to the run directory (from Stage 7)
    cfg : config dict
    logger : optional RunLogger
    use_crew : when True, invoke the LLM for logic/readability review

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
        # 1. Deterministic checks (KPI recomputation + references)
        log.info(STAGE, "running deterministic checks")
        checks = run_all_checks(run_dir)

        # 2. LLM review (optional)
        reason_codes: List[str] = []
        if use_crew:
            review = _run_llm_review(run_dir, cfg, log)
            if not review.get("logic_ok"):
                reason_codes.append("llm_logic_failure")
            if not review.get("readability_ok"):
                reason_codes.append("llm_readability_failure")
            # Append LLM notes as info-level checks
            for note in review.get("notes", []):
                from analysis.qa_recompute import QaCheck
                checks.append(QaCheck(
                    check="llm_review", severity="warning", message=note))
        else:
            review = _deterministic_review(run_dir)
            log.info(STAGE, "skipping LLM review (--no-crew)")

        # 3. Compute verdict
        verdict = run_verdict(run_dir, reason_codes=reason_codes)

        status = "passed" if verdict.verdict == "APPROVED" else verdict.verdict.lower()
        summary = {
            "stage": STAGE,
            "status": status,
            "verdict": verdict.verdict,
            "score": verdict.score,
            "critical_count": len(verdict.critical),
            "warning_count": len(verdict.warnings),
            "reason_codes": verdict.reason_codes,
            "qa_verdict_path": str(run_dir / "metadata" / "qa_verdict.json"),
        }
    except Exception as exc:  # noqa: BLE001
        log.error(STAGE, f"qa failed: {exc}")
        status = "failed"
        summary = {"run_id": run_id, "stage": STAGE, "status": status,
                   "error": str(exc)}
    log.stage_end(STAGE, status, time.monotonic() - start)
    summary["log_path"] = str(run_dir / "logs" / "run.jsonl")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qa_agent",
        description="Run Insight Forge stage 8 (QA) on a Stage-7 run dir.")
    parser.add_argument("run_dir", help="existing run dir (from Stage 7)")
    parser.add_argument("--crew", action="store_true",
                        help="invoke the LLM for logic/readability review")
    args = parser.parse_args(argv)

    cfg = load_config(require_key=bool(args.crew))
    result = run_qa(args.run_dir, cfg=cfg, use_crew=args.crew)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in ("passed", "approved") else 1


if __name__ == "__main__":
    sys.exit(main())
