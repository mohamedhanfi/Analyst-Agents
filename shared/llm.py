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

import os

# Kill the OpenTelemetry exporters CrewAI/litellm try to reach (spams the
# console with "Service Unavailable" retries) — must be set before the
# SDK initializes, i.e. before crewai is imported below.
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")

from crewai import LLM
import litellm

# Errors surface through exceptions (surfaced in the app), not as red
# console spam — the web app keeps its console quiet.
litellm.suppress_debug_info = True

DEFAULT_MODEL = "openrouter/z-ai/glm-5.2:free"


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
    max_tokens = llm_cfg.get("max_tokens")
    if max_tokens:
        kwargs["max_tokens"] = int(max_tokens)
    return LLM(**kwargs)


def _llm_kwargs(cfg: Dict[str, Any], agent_name: str,
                model: Optional[str] = None) -> Dict[str, Any]:
    """litellm-compatible kwargs for direct calls (same config as CrewAI)."""
    llm_cfg = cfg.get("llm", {})
    agent_cfg = cfg.get("agents", {}).get(agent_name, {})
    kwargs: Dict[str, Any] = dict(
        model=model or agent_cfg.get("model")
        or llm_cfg.get("model", DEFAULT_MODEL),
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
    max_tokens = llm_cfg.get("max_tokens")
    if max_tokens:
        kwargs["max_tokens"] = int(max_tokens)
    return kwargs


def _is_rate_limit(error: str) -> bool:
    """True when *error* looks like an HTTP 429 / quota saturation."""
    low = error.lower()
    return ("429" in error or "ratelimit" in low
            or "rate-limited" in low or "rate limit" in low)


def _backoff_sleep(attempt: int, error: str) -> None:
    """Exponential backoff between attempts; rate limits wait longer.

    Shared free pools (e.g. OpenRouter :free models) saturate upstream for
    tens of seconds — a short fixed retry burns attempts while busy."""
    wait = min(2 ** (attempt - 1), 8)
    if _is_rate_limit(error):
        wait = max(wait, min(15 * attempt, 45))
    time.sleep(wait)


def _model_chain(cfg: Dict[str, Any], agent_name: str) -> List[str]:
    """Primary model first, then ``llm.fallback_models`` (deduplicated).

    When the primary model is saturated or failing all retries, direct
    LLM calls fall through this chain before degrading to the
    deterministic per-stage fallback."""
    llm_cfg = cfg.get("llm", {})
    agent_cfg = cfg.get("agents", {}).get(agent_name, {})
    primary = str(agent_cfg.get("model")
                  or llm_cfg.get("model", DEFAULT_MODEL))
    models = [primary]
    for m in llm_cfg.get("fallback_models") or []:
        m = str(m)
        if m and m not in models:
            models.append(m)
    return models


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
    for model in _model_chain(cfg, agent_name):
        kwargs = _llm_kwargs(cfg, agent_name, model=model)
        for attempt in range(1, retries + 1):
            try:
                resp = litellm.completion(messages=messages, **kwargs)
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
                        warnings.append(
                            f"llm_attempt_{attempt}_schema_reject")
                        continue
                if validator is not None:
                    ok, errs = validator(payload)
                    if not ok:
                        last_error = "; ".join(errs) or "semantic reject"
                        warnings.append(
                            f"llm_attempt_{attempt}_semantic_reject")
                        continue
                return payload, warnings
            except Exception as exc:  # noqa: BLE001 -- transient provider errors
                last_error = str(exc)
                warnings.append(f"llm_attempt_{attempt}_error")
                _backoff_sleep(attempt, last_error)
        warnings.append(f"llm_model_exhausted_{model}")
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
    for model in _model_chain(cfg, agent_name):
        kwargs = _llm_kwargs(cfg, agent_name, model=model)
        for attempt in range(1, retries + 1):
            try:
                resp = litellm.completion(messages=messages, **kwargs)
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
                _backoff_sleep(attempt, last_error)
        warnings.append(f"llm_model_exhausted_{model}")
    warnings.append(f"llm_complete_text_failed_{last_error[:120]}")
    return None, warnings


def test_connection(
    cfg: Dict[str, Any],
    agent_name: str = "ingestion",
    timeout: float = 15.0,
) -> Optional[str]:
    """One quick call to the configured model — a pre-flight check.

    Returns None when the LLM API answered, otherwise a human-readable
    error message (missing key, bad URL, auth failure, provider outage...).
    """
    import litellm

    kwargs = _llm_kwargs(cfg, agent_name)
    kwargs["num_retries"] = 0
    kwargs["timeout"] = float(timeout)
    kwargs["max_tokens"] = 128  # ping only — override config for cheap pre-flight
    try:
        resp = litellm.completion(
            messages=[{"role": "user", "content": "ping"}],
            **kwargs)
        # Connectivity is what matters — a reasoning model may spend the
        # whole budget on reasoning_content and return empty content.
        if resp.choices:
            return None
        return "LLM returned no choices"
    except Exception as exc:  # noqa: BLE001 -- any provider/auth/network error
        msg = str(exc)[:300]
        # A rate-limit reply still proves connectivity AND auth are fine —
        # the provider simply answered "busy". Shared :free pools saturate
        # regularly; blocking run start on that makes the app unusable.
        # Every LLM stage already degrades deterministically per-call.
        if _is_rate_limit(msg):
            return None
        return msg


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
