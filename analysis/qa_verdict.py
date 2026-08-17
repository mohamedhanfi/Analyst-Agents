"""Stage 8 QA — deterministic score + verdict logic (§2.8).

Score is informational only.  The verdict is decided purely by logical
conditions (critical → NEEDS_REVISION, etc.).  No LLM in the decision.

CLI: python -m analysis.qa_verdict <run_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from analysis.qa_recompute import QaCheck, run_all_checks
from shared.schemas import QaVerdict


# ---------------------------------------------------------------------------
# Score (informational only)
# ---------------------------------------------------------------------------


def compute_score(checks: List[QaCheck]) -> float:
    """§2.8 score formula: 100 – (critical×15) – (warnings×2.5) – (info×0.5).

    Floor at 0.  This score is **informational/reporting only** — a high
    score does NOT override NEEDS_REVISION when a critical issue exists.
    """
    critical = sum(1 for c in checks if c.severity == "critical")
    warnings = sum(1 for c in checks if c.severity == "warning")
    info = sum(1 for c in checks if c.severity == "info")
    raw = 100.0 - (critical * 15) - (warnings * 2.5) - (info * 0.5)
    return max(0.0, raw)


# ---------------------------------------------------------------------------
# Verdict (deterministic, §2.8 table)
# ---------------------------------------------------------------------------


def decide_verdict(checks: List[QaCheck],
                   reason_codes: List[str] | None = None) -> str:
    """Apply the §2.8 verdict table.  Returns one of:
    APPROVED | APPROVED_WITH_WARNINGS | NEEDS_REVISION
    """
    reason_codes = reason_codes or []
    has_critical = any(c.severity == "critical" for c in checks)
    has_warning = any(c.severity == "warning" for c in checks)

    # §2.8: any critical → NEEDS_REVISION
    if has_critical:
        return "NEEDS_REVISION"

    # §2.8: fallback used → NEEDS_REVISION
    if any(rc.startswith("fallback") or rc == "cleaning_retry_limit_exceeded"
           or rc.endswith("_limit_exceeded")
           for rc in reason_codes):
        return "NEEDS_REVISION"

    # §2.8: invalid evidence / unresolved DQ → NEEDS_REVISION
    if any(rc.startswith("invalid_evidence") or rc.startswith("unresolved_dq")
           for rc in reason_codes):
        return "NEEDS_REVISION"

    # §2.8: resource limit exceeded → NEEDS_REVISION
    if any(rc.endswith("_limit_exceeded") for rc in reason_codes):
        return "NEEDS_REVISION"

    if has_warning:
        return "APPROVED_WITH_WARNINGS"

    return "APPROVED"


# ---------------------------------------------------------------------------
# Build full QaVerdict
# ---------------------------------------------------------------------------


def build_qa_verdict(checks: List[QaCheck],
                     reason_codes: List[str] | None = None) -> QaVerdict:
    """Assemble a QaVerdict from check results + reason codes."""
    reason_codes = reason_codes or []
    score = compute_score(checks)
    verdict = decide_verdict(checks, reason_codes)

    critical = [c.message for c in checks if c.severity == "critical"]
    warnings = [c.message for c in checks if c.severity == "warning"]
    info = [c.message for c in checks if c.severity == "info"]

    return QaVerdict(
        verdict=verdict,
        score=score,
        critical=critical,
        warnings=warnings,
        info=info,
        reason_codes=reason_codes,
    )


# ---------------------------------------------------------------------------
# Full pipeline: checks → verdict → save
# ---------------------------------------------------------------------------


def run_verdict(run_dir: str | Path,
                reason_codes: List[str] | None = None) -> QaVerdict:
    """Run all deterministic checks and produce the final verdict."""
    run_dir = Path(run_dir)
    checks = run_all_checks(run_dir)
    verdict = build_qa_verdict(checks, reason_codes)
    # Save
    out = run_dir / "metadata"
    out.mkdir(parents=True, exist_ok=True)
    (out / "qa_verdict.json").write_text(
        verdict.model_dump_json(indent=2), encoding="utf-8")
    return verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qa_verdict",
        description="Deterministic QA verdict + score from §2.8")
    parser.add_argument("run_dir", help="run directory")
    args = parser.parse_args(argv)

    verdict = run_verdict(args.run_dir)
    print(verdict.model_dump_json(indent=2))
    return 0 if verdict.verdict == "APPROVED" else 1


if __name__ == "__main__":
    sys.exit(main())
