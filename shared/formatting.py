"""Shared formatting and parsing utilities.

Pure functions with no side effects, no LLM, no file I/O — safe for any
module to import.
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional


def fmt(value: float) -> str:
    """Short, deterministic number formatting (no trailing float noise).

    Used by chart_renderer, insight_agent, and any future consumer that
    needs human-readable numbers in labels or descriptions.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, int):
        return f"{value:,}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if value == 0:
        return "0"
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def parse_json(raw: str) -> Any:
    """Extract the first JSON object/array from a string that may contain
    prose, markdown fences, or partial wrapping.

    Returns None when nothing parseable is found. Never raises.
    """
    if not raw or not isinstance(raw, str):
        return None
    start = raw.find("{") if "{" in raw else -1
    end = raw.rfind("}")
    try:
        if start == -1 or end <= start:
            return json.loads(raw.strip()) if raw.strip() else None
        return json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
