"""Single LLM factory for every agent (§2, config.yaml agents.*).

One copy of the "config -> CrewAI LLM" wiring instead of a duplicated
build_llm per agent module. Per-agent overrides live in
`cfg["agents"][agent_name].model` (e.g. agents.ingestion.model, and QA must
use a DIFFERENT model than generation — §2.8); anything not overridden
inherits from `cfg["llm"].model`.
"""
from __future__ import annotations

from typing import Any, Dict

from crewai import LLM

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


def build_llm(cfg: Dict[str, Any], agent_name: str) -> LLM:
    llm_cfg = cfg.get("llm", {})
    agent_cfg = cfg.get("agents", {}).get(agent_name, {})
    return LLM(
        model=agent_cfg.get("model") or llm_cfg.get("model", DEFAULT_MODEL),
        base_url=llm_cfg.get("base_url"),
        api_key=llm_cfg.get("api_key"),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        seed=int(llm_cfg.get("seed", 42)),
    )
