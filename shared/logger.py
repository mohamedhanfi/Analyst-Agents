"""Structured per-stage logger → runs/<run_id>/logs/ (§5).

Every entry is a JSON line (JSONL): stage, timestamp, kind, latency, tokens,
cost, tool calls, errors, retries — the audit trail for a run.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class RunLogger:
    """Appends JSONL entries under <run_dir>/logs/run.jsonl.

    Safe for sequential use within one run; single writer per run dir.
    """

    def __init__(self, run_dir: str | os.PathLike, run_id: str = ""):
        logs_dir = Path(run_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self._path = logs_dir / "run.jsonl"
        self._run_id = run_id
        self.cost_usd: float = 0.0
        self.tokens_in: int = 0
        self.tokens_out: int = 0

    # -- core ----------------------------------------------------------------

    def _write(self, entry: Dict[str, Any]) -> None:
        entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
        if self._run_id:
            entry.setdefault("run_id", self._run_id)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -- stage lifecycle -----------------------------------------------------

    def stage_start(self, stage: str) -> None:
        self._write({"kind": "stage_start", "stage": stage})

    def stage_end(self, stage: str, status: str, duration_s: float) -> None:
        self._write({"kind": "stage_end", "stage": stage,
                     "status": status, "duration_s": round(duration_s, 3)})

    # -- llm / tools ---------------------------------------------------------

    def llm_call(self, stage: str, model: str,
                 latency_s: float, tokens_in: int, tokens_out: int,
                 cost_usd: float, retries: int = 0,
                 error: Optional[str] = None) -> None:
        self.cost_usd += cost_usd
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self._write({"kind": "llm_call", "stage": stage, "model": model,
                     "latency_s": round(latency_s, 3),
                     "tokens_in": tokens_in, "tokens_out": tokens_out,
                     "cost_usd": round(cost_usd, 6),
                     "retries": retries, "error": error})

    def tool_call(self, stage: str, tool: str, status: str,
                  duration_s: float, note: str = "") -> None:
        self._write({"kind": "tool_call", "stage": stage, "tool": tool,
                     "status": status, "duration_s": round(duration_s, 3),
                     "note": note})

    # -- events --------------------------------------------------------------

    def error(self, stage: str, message: str) -> None:
        self._write({"kind": "error", "stage": stage, "message": message})

    def info(self, stage: str, message: str, **extra: Any) -> None:
        self._write({"kind": "info", "stage": stage,
                     "message": message, **extra})

    def fallback(self, stage: str, reason: str) -> None:
        """Fallback/hard-cap trip: reason codes from §1 (e.g. cost_limit_exceeded)."""
        self._write({"kind": "fallback", "stage": stage, "reason": reason})


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat()