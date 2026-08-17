"""CrewAI Flow helpers — deterministic branching logic for the pipeline.

Implements:
- DQ gate: passed / needs_repair
- Cleaning recheck loop control
- Hard-cap checks (cost, tokens, runtime)
- Verdict assembly from QA result + fallback reason codes
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# DQ gate
# ---------------------------------------------------------------------------

def check_dq_gate(dq_result: Dict[str, Any]) -> str:
    """Return 'pass' or 'repair' based on the data-quality report status.

    Parameters
    ----------
    dq_result : dict
        The dict returned by ``run_data_quality()``. Must contain a ``status``
        key with value ``'passed'`` or ``'needs_repair'``.

    Returns
    -------
    str
        ``'pass'`` or ``'repair'``.
    """
    status = dq_result.get("status", "failed")
    return "pass" if status == "passed" else "repair"


# ---------------------------------------------------------------------------
# Cleaning recheck
# ---------------------------------------------------------------------------

def check_cleaning_retry(
    recheck_result: Dict[str, Any],
    attempt: int,
    max_rechecks: int,
) -> bool:
    """Return True if cleaning recheck has passed (pipeline may continue).

    Parameters
    ----------
    recheck_result : dict
        The dict returned by a ``run_data_quality()`` recheck.
    attempt : int
        Current attempt number (1-based).
    max_rechecks : int
        Maximum allowed rechecks (from ``config.yaml limits.cleaning_max_rechecks``).

    Returns
    -------
    bool
        True if the recheck passed or we are within the retry budget and should
        continue; False if retries are exhausted.
    """
    if recheck_result.get("status") == "passed":
        return True
    return attempt < max_rechecks


# ---------------------------------------------------------------------------
# Hard-cap checks
# ---------------------------------------------------------------------------

def check_caps(
    log: Any,
    t0: float,
    cfg: Dict[str, Any],
) -> str | None:
    """Check hard resource caps. Returns a reason code if tripped, else None.

    Parameters
    ----------
    log : RunLogger
        The run logger (has ``.cost_usd`` attribute).
    t0 : float
        Pipeline start time (``time.time()`` value).
    cfg : dict
        Loaded config.

    Returns
    -------
    str or None
        Reason code string if a cap was tripped, otherwise None.
    """
    max_cost = cfg.get("llm", {}).get("max_cost_usd", 5.0)
    max_run = cfg.get("limits", {}).get("max_run_seconds", 1800)

    if getattr(log, "cost_usd", 0.0) >= max_cost:
        return "cost_limit_exceeded"
    if (time.time() - t0) >= max_run:
        return "run_time_limit_exceeded"
    return None


# ---------------------------------------------------------------------------
# Verdict assembly
# ---------------------------------------------------------------------------

def build_verdict(
    qa_result: Dict[str, Any],
    reason_codes: List[str] | None = None,
) -> Dict[str, Any]:
    """Assemble the final verdict dict from the QA result and any fallback reasons.

    If fallback reason codes are present, the verdict is forced to
    ``NEEDS_REVISION`` regardless of what QA decided.

    Parameters
    ----------
    qa_result : dict
        The dict returned by ``run_qa()``.
    reason_codes : list of str, optional
        Fallback reason codes from hard-cap or cleaning-retry failures.

    Returns
    -------
    dict
        Verdict payload to write to ``metadata/qa_verdict.json``.
    """
    codes = reason_codes or []
    verdict = qa_result.get("verdict", "NEEDS_REVISION")
    score = qa_result.get("score", 0.0)
    critical = qa_result.get("critical_count", 0)
    warning = qa_result.get("warning_count", 0)

    # Fallback reasons force NEEDS_REVISION
    if codes:
        verdict = "NEEDS_REVISION"

    return {
        "verdict": verdict,
        "score": score,
        "critical": critical,
        "warnings": warning,
        "reason_codes": codes,
    }


def write_verdict_file(
    run_dir: Path,
    verdict_payload: Dict[str, Any],
) -> Path:
    """Write ``metadata/qa_verdict.json``.

    Parameters
    ----------
    run_dir : Path
        The run directory.
    verdict_payload : dict
        Output of :func:`build_verdict`.

    Returns
    -------
    Path
        Path to the written file.
    """
    out_path = run_dir / "metadata" / "qa_verdict.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(verdict_payload, fh, indent=2, ensure_ascii=False)
    return out_path
