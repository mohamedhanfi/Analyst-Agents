"""Single LLM factory for every agent (§2, config.yaml agents.*).

One copy of the "config -> CrewAI LLM" wiring instead of a duplicated
build_llm per agent module. Per-agent overrides live in
`cfg["agents"][agent_name].model` (e.g. agents.ingestion.model, and QA must
use a DIFFERENT model than generation — §2.8); anything not overridden
inherits from `cfg["llm"].model`.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from crewai import LLM

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


def build_llm(cfg: Dict[str, Any], agent_name: str) -> LLM:
    llm_cfg = cfg.get("llm", {})
    agent_cfg = cfg.get("agents", {}).get(agent_name, {})
    kwargs: Dict[str, Any] = dict(
        model=agent_cfg.get("model") or llm_cfg.get("model", DEFAULT_MODEL),
        base_url=llm_cfg.get("base_url"),
        api_key=llm_cfg.get("api_key"),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        seed=int(llm_cfg.get("seed", 42)),
    )
    # §1 hard limits: cap retries from config.
    max_retries = llm_cfg.get("max_retries")
    if max_retries:
        kwargs["max_retries"] = int(max_retries)
    return LLM(**kwargs)


def _llm_kwargs(cfg: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """litellm-compatible kwargs for direct calls (same config as CrewAI)."""
    llm_cfg = cfg.get("llm", {})
    agent_cfg = cfg.get("agents", {}).get(agent_name, {})
    kwargs: Dict[str, Any] = dict(
        model=agent_cfg.get("model") or llm_cfg.get("model", DEFAULT_MODEL),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        seed=int(llm_cfg.get("seed", 42)),
    )
    if llm_cfg.get("base_url"):
        kwargs["api_base"] = llm_cfg["base_url"]
    if llm_cfg.get("api_key"):
        kwargs["api_key"] = llm_cfg["api_key"]
    max_retries = llm_cfg.get("max_retries")
    if max_retries:
        kwargs["num_retries"] = int(max_retries)
    return kwargs


def complete_json(
    cfg: Dict[str, Any],
    agent_name: str,
    system: str,
    user: str,
    schema: Optional[Any] = None,
    validator: Optional[Callable[[Dict[str, Any]], Tuple[bool, List[str]]]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Direct LLM call (no CrewAI) that returns validated JSON.

    Parameters
    ----------
    cfg : full config dict
    agent_name : config key under cfg["agents"] (llm/key selection)
    system : system prompt
    user : user prompt (data must already be wrapped per prompt_guard)
    schema : optional pydantic model — the response must validate
    validator : optional (dict) -> (ok, warnings) semantic validator

    Returns
    -------
    (payload, warnings) — payload is None when every attempt failed;
    warnings always carries the reason(s) so callers can fall back
    deterministically instead of blocking the pipeline.
    """
    import litellm

    llm_cfg = cfg.get("llm", {})
    retries = int(llm_cfg.get("complete_json_retries", 3))
    warnings: List[str] = []
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error = "no response"
    for attempt in range(1, retries + 1):
        try:
            resp = litellm.completion(
                messages=messages, **_llm_kwargs(cfg, agent_name))
            content = resp.choices[0].message.content
            if not content or not str(content).strip():
                last_error = "empty LLM response"
                warnings.append(f"llm_attempt_{attempt}_empty")
                continue
            payload = _extract_json(str(content))
            if payload is None:
                last_error = "LLM output was not JSON"
                warnings.append(f"llm_attempt_{attempt}_not_json")
                continue
            if schema is not None:
                try:
                    payload = schema.model_validate(payload).model_dump()
                except Exception as exc:  # noqa: BLE001 -- schema reject
                    last_error = f"schema validation failed: {exc}"
                    warnings.append(f"llm_attempt_{attempt}_schema_reject")
                    continue
            if validator is not None:
                ok, errs = validator(payload)
                if not ok:
                    last_error = "; ".join(errs) or "semantic reject"
                    warnings.append(f"llm_attempt_{attempt}_semantic_reject")
                    continue
            return payload, warnings
        except Exception as exc:  # noqa: BLE001 -- transient provider errors
            last_error = str(exc)
            warnings.append(f"llm_attempt_{attempt}_error")
            time.sleep(min(2 ** (attempt - 1), 8))
    warnings.append(f"llm_complete_json_failed_{last_error[:120]}")
    return None, warnings


def complete_text(
    cfg: Dict[str, Any],
    agent_name: str,
    system: str,
    user: str,
) -> Tuple[Optional[str], List[str]]:
    """Direct LLM call returning plain text (used for the exec summary).

    Returns (text, warnings); text is None when every attempt failed so the
    caller falls back to the deterministic summary.
    """
    import litellm

    llm_cfg = cfg.get("llm", {})
    retries = int(llm_cfg.get("complete_json_retries", 3))
    warnings: List[str] = []
    last_error = "no response"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for attempt in range(1, retries + 1):
        try:
            resp = litellm.completion(
                messages=messages, **_llm_kwargs(cfg, agent_name))
            content = resp.choices[0].message.content
            text = str(content or "").strip()
            if not text:
                last_error = "empty LLM response"
                warnings.append(f"llm_attempt_{attempt}_empty")
                continue
            return text, warnings
        except Exception as exc:  # noqa: BLE001 -- transient provider errors
            last_error = str(exc)
            warnings.append(f"llm_attempt_{attempt}_error")
            time.sleep(min(2 ** (attempt - 1), 8))
    warnings.append(f"llm_complete_text_failed_{last_error[:120]}")
    return None, warnings


def _extract_json(content: str) -> Optional[Dict[str, Any]]:
    """Tolerate prose + ```json fences around the payload."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None
