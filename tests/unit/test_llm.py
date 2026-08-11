"""Unit tests for shared/llm.build_llm — one factory for every agent.

CrewAI's LLM normalizes the model string: the provider prefix is kept in
the config, while llm.model exposes the bare name (e.g.
"deepseek/deepseek-v4-flash" -> "deepseek-v4-flash").
"""
from __future__ import annotations

from shared.llm import DEFAULT_MODEL, build_llm
from shared.utils import load_config


def _cfg(**llm_overrides):
    llm = {"api_key": "dummy"}
    llm.update(llm_overrides)
    return {"llm": llm}


def test_default_model_fallback():
    llm = build_llm(_cfg(), "ghost_agent")
    assert llm.model == DEFAULT_MODEL.split("/", 1)[1]


def test_llm_section_model_used():
    llm = build_llm(_cfg(model="openai/gpt-4o-mini"), "ghost_agent")
    assert llm.model == "gpt-4o-mini"


def test_per_agent_override_wins():
    cfg = _cfg(model="openai/gpt-4o-mini")
    cfg["agents"] = {"ingestion": {"model": "openai/gpt-4o-mini"}}
    llm = build_llm(cfg, "ingestion")
    assert llm.model == "gpt-4o-mini"


def test_no_override_uses_llm_section():
    cfg = load_config(require_key=False)
    cfg["llm"]["api_key"] = "dummy"
    cfg["agents"]["understanding"].pop("model", None)
    llm = build_llm(cfg, "understanding")
    assert llm.model == cfg["llm"]["model"].split("/", 1)[1]
