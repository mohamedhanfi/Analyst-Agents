"""Unit tests for shared/llm.build_llm — one factory for every agent.

CrewAI's LLM normalizes the model string: the provider prefix is kept in
the config, while llm.model exposes the bare name (e.g.
"deepseek/deepseek-v4-flash" -> "deepseek-v4-flash").
"""
from __future__ import annotations

from shared.llm import DEFAULT_MODEL, build_llm
from shared.utils import load_config


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


# ---------------------------------------------------------------------------
# complete_json / complete_text — direct LLM calls with schema validation
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, content: str):
        msg = type("Msg", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]


def _cfg(**llm_overrides):
    llm = {"api_key": "dummy", "complete_json_retries": 3}
    llm.update(llm_overrides)
    return {"llm": llm}


def test_complete_json_valid_payload(monkeypatch):
    from shared.llm import complete_json
    import litellm

    monkeypatch.setattr(
        litellm, "completion",
        lambda messages, **kw: _FakeResp('{"ok": true, "n": 3}'))
    payload, warnings = complete_json(_cfg(), "qa", "sys", "user")
    assert payload == {"ok": True, "n": 3}
    assert not warnings


def test_complete_json_schema_reject_then_retry(monkeypatch):
    from shared.llm import complete_json
    import litellm

    calls = {"n": 0}

    def fake(messages, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResp('{"readability_ok": "definitely", "logic_ok": true}')
        return _FakeResp('{"readability_ok": true, "logic_ok": true}')

    monkeypatch.setattr(litellm, "completion", fake)
    payload, warnings = complete_json(
        _cfg(), "qa", "sys", "user", schema=_MakeJsonSchema())
    assert payload == {"readability_ok": True, "logic_ok": True, "notes": []}
    assert calls["n"] == 2


def test_complete_json_all_fail_returns_none(monkeypatch):
    from shared.llm import complete_json
    import litellm

    monkeypatch.setattr(litellm, "completion",
                        lambda messages, **kw: _FakeResp("not json"))
    payload, warnings = complete_json(_cfg(), "qa", "sys", "user")
    assert payload is None
    assert any("llm_complete_json_failed" in w for w in warnings)


def test_complete_json_tolerates_fenced_json(monkeypatch):
    from shared.llm import complete_json
    import litellm

    monkeypatch.setattr(
        litellm, "completion",
        lambda messages, **kw: _FakeResp(
            "Here you go:\n```json\n{\"logic_ok\": true, "
            "\"readability_ok\": true, \"notes\": []}\n```"))
    payload, warnings = complete_json(
        _cfg(), "qa", "sys", "user", schema=_MakeJsonSchema())
    assert payload == {"logic_ok": True, "readability_ok": True,
                       "notes": []}


def test_complete_text_returns_plain_text(monkeypatch):
    from shared.llm import complete_text
    import litellm

    monkeypatch.setattr(
        litellm, "completion",
        lambda messages, **kw: _FakeResp("Sales grew 12% in Q1."))
    text, warnings = complete_text(_cfg(), "report", "sys", "user")
    assert text == "Sales grew 12% in Q1."
    assert not warnings


def test_complete_text_all_empty_returns_none(monkeypatch):
    from shared.llm import complete_text
    import litellm

    monkeypatch.setattr(litellm, "completion",
                        lambda messages, **kw: _FakeResp("  "))
    text, warnings = complete_text(_cfg(), "report", "sys", "user")
    assert text is None
    assert any("llm_complete_text_failed" in w for w in warnings)


def test_extract_json_with_prose(monkeypatch):
    from shared.llm import _extract_json
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('prefix {"a": 1} suffix') == {"a": 1}
    assert _extract_json("no json here") is None


def _MakeJsonSchema():
    from pydantic import BaseModel, Field

    class QaReview(BaseModel):
        readability_ok: bool
        logic_ok: bool
        notes: list = Field(default_factory=list)

    return QaReview
