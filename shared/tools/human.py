"""CrewAI @tool wrapper — human business-context input (§2.1)."""
from __future__ import annotations

import json
import os
import sys
import time

from crewai.tools import tool

from shared.core.business_context import BusinessContextGatherer
from shared.utils import load_config


def _is_interactive() -> bool:
    """True when stdin is a real TTY (terminal), not piped/redirected."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, OSError):
        return False


@tool("human_input_tool")
def human_input_tool(prompt: str) -> str:
    """Ask the user a business-context question.

    Single prompt, blocking with the configured timeout
    (config limits.human_input_timeout_min). Returns
    {"answer": "..."} or {"timed_out": true, "generic_mode": true}
    when the user does not answer in time (§3.5 Generic Analysis Mode).
    Skips waiting entirely when INSIGHT_FORGE_NO_INPUT is set or
    stdin is non-interactive (piped/redirected).
    """
    if os.environ.get("INSIGHT_FORGE_NO_INPUT") == "1":
        return json.dumps({"timed_out": True, "generic_mode": True},
                          ensure_ascii=False)
    if not _is_interactive():
        return json.dumps({"timed_out": True, "generic_mode": True},
                          ensure_ascii=False)
    cfg = load_config(require_key=False)
    timeout_min = float(cfg["limits"].get("human_input_timeout_min", 5.0))
    gatherer = BusinessContextGatherer(timeout_seconds=int(timeout_min * 60))
    answer = gatherer._ask(prompt, time.monotonic() + timeout_min * 60)
    if answer is None:
        return json.dumps({"timed_out": True, "generic_mode": True},
                          ensure_ascii=False)
    return json.dumps({"answer": answer}, ensure_ascii=False)