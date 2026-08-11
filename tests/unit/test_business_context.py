"""Unit tests for shared/core/business_context.BusinessContextGatherer."""
from __future__ import annotations

import json

from shared.core.business_context import BusinessContextGatherer


def _fake_input(lines):
    it = iter(lines)
    return lambda _: next(it)


def test_gather_full_answers():
    answers = [
        "Track revenue growth",
        "sales",
        "Which category grows fastest?\nWhat is the AOV trend?",
        "Set next-quarter targets",
    ]
    ctx = BusinessContextGatherer(timeout_seconds=30,
                                  input_func=_fake_input(answers)) \
        .gather("sales.xlsx")
    assert not ctx.generic_mode
    assert ctx.context_confidence > 0
    assert ctx.business_questions == [
        "Which category grows fastest?", "What is the AOV trend?"]


def test_timeout_generic_mode():
    def slow_input(_):
        import time
        time.sleep(5)
        return "late"
    ctx = BusinessContextGatherer(timeout_seconds=0.2,
                                  input_func=slow_input).gather("f.xlsx")
    assert ctx.generic_mode is True
    assert ctx.context_confidence == 0


def test_sheet_selection_single_skipped():
    answers = ["goal", "sales", "", ""]
    ctx = BusinessContextGatherer(timeout_seconds=30,
                                  input_func=_fake_input(answers)) \
        .gather("f.xlsx", sheet_names=["Sales"])
    assert ctx.sheet_used is None


def test_generic_on_first_empty():
    ctx = BusinessContextGatherer(timeout_seconds=30,
                                  input_func=_fake_input([""])) \
        .gather("f.xlsx")
    assert ctx.generic_mode


def test_generic_on_none_input_return():
    ctx = BusinessContextGatherer(timeout_seconds=30,
                                  input_func=lambda _: None) \
        .gather("f.xlsx")
    assert ctx.generic_mode
    assert ctx.context_confidence == 0


def test_save_writes_json(tmp_path):
    ctx = BusinessContextGatherer(timeout_seconds=30,
                                  input_func=_fake_input(
                                      ["goal", "hr", "Turnover?", "Hiring"])) \
        .gather("hr.xlsx")
    path = BusinessContextGatherer.save(ctx, tmp_path / "run2")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["goal_summary"] == "goal"
    assert payload["context_confidence"] > 0