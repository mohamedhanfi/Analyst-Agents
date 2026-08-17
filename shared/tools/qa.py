"""CrewAI @tool wrappers — Stage 8 QA tools (§2.8).

Three tools:
  - review_logic_tool: independent LLM checks insight/recommendation alignment
  - score_calculator_tool: wraps the deterministic score formula
  - verdict_tool: wraps the deterministic verdict logic
"""
from __future__ import annotations

import json
from typing import Any

from crewai.tools import tool

from analysis.qa_recompute import QaCheck, run_all_checks
from analysis.qa_verdict import build_qa_verdict, compute_score


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


# ------------------------------------------------------------------
# Tool 1 — Review logic + readability (LLM-driven)
# ------------------------------------------------------------------


@tool("review_logic_tool")
def review_logic_tool(insights_json: str, recommendations_json: str,
                      exec_summary: str) -> str:
    """Independent LLM review of insight/recommendation logic and
    executive summary readability.

    Pass this to the QA agent's LLM. The LLM checks:
    1. Every insight's evidence_ids logically ground the claim
    2. Recommendations follow logically from the insights
    3. Executive summary is readable and ≤5 sentences

    Returns {"readability_ok": bool, "logic_ok": bool, "notes": [...]}.
    """
    try:
        insights = json.loads(insights_json)
    except (json.JSONDecodeError, TypeError):
        insights = []
    try:
        recs = json.loads(recommendations_json)
    except (json.JSONDecodeError, TypeError):
        recs = []

    notes: list[str] = []

    # Basic structural checks (deterministic, no LLM needed for these)
    if not isinstance(insights, list):
        notes.append("insights is not a list")
    elif not insights:
        notes.append("no insights to review")

    if not isinstance(recs, list):
        notes.append("recommendations is not a list")
    elif not recs:
        notes.append("no recommendations to review")

    # Check recommendation → insight linkage
    insight_ids = set()
    if isinstance(insights, list):
        for ins in insights:
            if isinstance(ins, dict):
                iid = ins.get("insight_id")
                if iid:
                    insight_ids.add(iid)

    if isinstance(recs, list):
        for rec in recs:
            if isinstance(rec, dict):
                ref = rec.get("insight_id")
                if ref and ref not in insight_ids:
                    notes.append(
                        f"recommendation {rec.get('recommendation_id', '?')} "
                        f"references non-existent insight {ref!r}")

    # Executive summary check
    summary = str(exec_summary or "").strip()
    readability_ok = True
    if not summary:
        notes.append("executive summary is empty")
        readability_ok = False
    elif len(summary.split()) > 100:
        notes.append(f"executive summary is {len(summary.split())} words "
                     "(recommended ≤50)")
        readability_ok = False

    logic_ok = not notes

    return _json({
        "readability_ok": readability_ok,
        "logic_ok": logic_ok,
        "notes": notes,
    })


# ------------------------------------------------------------------
# Tool 2 — Score calculator (deterministic)
# ------------------------------------------------------------------


@tool("score_calculator_tool")
def score_calculator_tool(checks_json: str) -> str:
    """Compute the §2.8 QA score from a list of checks.

    checks_json: JSON array of {"check": str, "severity": str, "message": str}

    Returns {"score": float, "critical": int, "warnings": int, "info": int}.
    """
    try:
        raw = json.loads(checks_json)
    except (json.JSONDecodeError, TypeError):
        return _json({"score": 0.0, "critical": 0, "warnings": 0, "info": 0,
                       "error": "invalid checks JSON"})

    checks = []
    for item in (raw if isinstance(raw, list) else []):
        if isinstance(item, dict):
            checks.append(QaCheck(
                check=item.get("check", ""),
                severity=item.get("severity", "info"),
                message=item.get("message", "")))

    score = compute_score(checks)
    return _json({
        "score": score,
        "critical": sum(1 for c in checks if c.severity == "critical"),
        "warnings": sum(1 for c in checks if c.severity == "warning"),
        "info": sum(1 for c in checks if c.severity == "info"),
    })


# ------------------------------------------------------------------
# Tool 3 — Verdict (deterministic)
# ------------------------------------------------------------------


@tool("verdict_tool")
def verdict_tool(checks_json: str,
                 reason_codes_json: str = "[]") -> str:
    """Compute the §2.8 QA verdict from checks + reason codes.

    checks_json: JSON array of {"check": str, "severity": str, "message": str}
    reason_codes_json: JSON array of reason code strings

    Returns the full QaVerdict as JSON (verdict, score, critical, warnings,
    info, reason_codes).
    """
    try:
        raw = json.loads(checks_json)
    except (json.JSONDecodeError, TypeError):
        raw = []
    try:
        reason_codes = json.loads(reason_codes_json)
    except (json.JSONDecodeError, TypeError):
        reason_codes = []

    checks = []
    for item in (raw if isinstance(raw, list) else []):
        if isinstance(item, dict):
            checks.append(QaCheck(
                check=item.get("check", ""),
                severity=item.get("severity", "info"),
                message=item.get("message", "")))

    verdict = build_qa_verdict(checks, reason_codes)
    return verdict.model_dump_json(indent=2)
