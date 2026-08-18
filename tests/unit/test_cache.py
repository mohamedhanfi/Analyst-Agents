"""Unit tests for shared/cache — SQLite result cache + source index."""
from __future__ import annotations

import json

import pytest

from shared import cache


def _cfg(**llm_overrides):
    llm = {"model": "openrouter/x", "temperature": 0.0, "seed": 42}
    llm.update(llm_overrides)
    return {"pipeline_version": "4.3.0", "llm": llm}


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DB_PATH", tmp_path / "index.sqlite3")


def test_key_changes_with_input_and_config():
    k1 = cache.cache_key("abc", _cfg())
    k2 = cache.cache_key("abd", _cfg())
    assert k1 != k2
    k3 = cache.cache_key("abc", _cfg(model="openrouter/y"))
    assert k1 != k3


def test_store_and_get_roundtrip():
    key = cache.cache_key("abc", _cfg())
    assert cache.get_cached(key) is None
    cache.store(key, "run_1", "sales.csv")
    hit = cache.get_cached(key)
    assert hit == {"run_id": "run_1", "created_at": hit["created_at"]}


def test_store_replaces_same_key():
    key = cache.cache_key("abc", _cfg())
    cache.store(key, "run_1", "sales.csv")
    cache.store(key, "run_2", "sales.csv")
    assert cache.get_cached(key)["run_id"] == "run_2"


def test_input_hash_changes_with_content(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("x,1\n", encoding="utf-8")
    b.write_text("x,2\n", encoding="utf-8")
    assert cache.input_hash(a) != cache.input_hash(b)
    assert cache.input_hash(a) == cache.input_hash(a)


def test_find_previous_returns_latest_and_excludes_current(tmp_path):
    cache.store("k1", "run_1", "sales.csv")
    cache.store("k2", "run_2", "sales.csv")
    cache.store("k3", "run_9", "other.csv")
    assert cache.find_previous("sales.csv") == "run_2"
    assert cache.find_previous("sales.csv", exclude_run_id="run_2") == "run_1"
    assert cache.find_previous("other.csv") == "run_9"
    assert cache.find_previous("ghost.csv") is None