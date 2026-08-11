"""BusinessContextGatherer — asks the user the ingestion questions (§2.1).

Console dialog with a total timeout (config limits.human_input_timeout_min).
If the user does not answer in time (or input is unavailable), the gatherer
returns Generic Analysis Mode: generic_mode=True, context_confidence=0 (§3.5).
Answers are free text — cell content is never involved here.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List

from shared.schemas import BusinessContext

SHEET_QUESTION = (
    "This file contains multiple sheets: {sheets}\n"
    "Which sheet should we analyse? (type the exact name): "
)

DEFAULT_QUESTIONS: List[str] = [
    "What is the main goal of this analysis?",
    "What business domain / industry is this dataset from "
    "(e.g. sales, finance, hr)?",
    "Which business questions should the report answer "
    "(one per line, up to 5, empty line to finish)?",
    "What decisions will be taken based on this report?",
]

InputFunc = Callable[[str], str]


class BusinessContextGatherer:
    def __init__(self, timeout_seconds: int = 300,
                 input_func: InputFunc | None = None):
        self.timeout_seconds = timeout_seconds
        self._input = input_func or input

    # ------------------------------------------------------------------ API

    def gather(self, file_name: str,
               sheet_names: List[str] | None = None) -> BusinessContext:
        deadline = time.monotonic() + self.timeout_seconds

        sheet_used: str | None = None
        if sheet_names and len(sheet_names) > 1:
            sheet_used = self._ask(
                SHEET_QUESTION.format(sheets=", ".join(sheet_names)), deadline)
            if sheet_used is None:
                return self._generic(file_name)

        questions: List[str] = []
        answers: Dict[str, str] = {}
        for question in DEFAULT_QUESTIONS:
            answer = self._ask(question, deadline)
            if answer is None or answer == "":
                if not questions and not answers:
                    return self._generic(file_name)
                break
            if question.startswith("Which business questions"):
                questions = [q for q in answer.splitlines() if q.strip()]
            else:
                answers[question] = answer
            deadline = self._recompute_deadline(deadline)

        goal_summary = answers.get(DEFAULT_QUESTIONS[0], "")
        confidence = self._confidence(answers, sheet_used)

        return BusinessContext(
            file_name=file_name,
            sheet_used=sheet_used,
            business_questions=questions,
            answers=answers,
            goal_summary=goal_summary,
            context_confidence=confidence,
            generic_mode=False,
        )

    # ------------------------------------------------------------- internals

    def _ask(self, prompt: str, deadline: float) -> str | None:
        """Blocking input with per-question timeout; None means timed out."""
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0:
            return None
        box: Dict[str, str] = {}

        def worker() -> None:
            try:
                box["value"] = self._input(prompt + " ")
            except EOFError:
                box["value"] = None

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(remaining)
        if thread.is_alive():
            return None
        return box.get("value", "").strip()

    @staticmethod
    def _recompute_deadline(_deadline: float) -> float:
        return _deadline  # keep the original total deadline

    @staticmethod
    def _confidence(answers: Dict[str, str], sheet_used: str | None) -> float:
        answered = [q for q, v in answers.items() if v]
        if not answered:
            return 0.0
        base = len(answered) / len(DEFAULT_QUESTIONS)
        return min(1.0, round(base + (0.1 if sheet_used else 0.0), 2))

    @staticmethod
    def _generic(file_name: str) -> BusinessContext:
        return BusinessContext(
            file_name=file_name,
            context_confidence=0.0,
            generic_mode=True,
        )

    # ------------------------------------------------------------------ save

    @staticmethod
    def save(context: BusinessContext, run_dir: str | Path) -> Path:
        path = Path(run_dir) / "knowledge" / "business_context.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(context.model_dump(), ensure_ascii=False,
                                   indent=2), encoding="utf-8")
        return path