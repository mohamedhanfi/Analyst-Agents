"""SQLite result cache + previous-run index (§8 / audit item F).

The cache maps a content+config key to the run_id of a successful
deterministic run, so re-running the same file with the same settings
serves the previous result instead of recomputing. Only APPROVED
deterministic runs are cached (LLM runs are not, per audit).

The same table doubles as the "source_name -> run_id" index used by the
run_comparison callout: the most recent previous run of the same source
file becomes the baseline for the "vs previous run" table in the report.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
DB_PATH = CACHE_DIR / "index.sqlite3"


def _conn() -> sqlite3.Connection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")   # concurrency-safe (§4/§5)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runs("
        " key TEXT PRIMARY KEY,"
        " run_id TEXT NOT NULL,"
        " source_name TEXT NOT NULL,"
        " created_at REAL NOT NULL)")
    return conn


def cache_key(input_hash: str, cfg: Dict[str, Any]) -> str:
    """Key = input hash + pipeline version + LLM settings (deterministic)."""
    payload = {
        "input": input_hash,
        "version": cfg.get("pipeline_version", "4.3.0"),
        "model": cfg.get("llm", {}).get("model"),
        "temperature": cfg.get("llm", {}).get("temperature"),
        "seed": cfg.get("llm", {}).get("seed"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def input_hash(path: str | Path) -> str:
    """sha256 of the file bytes (1 MB chunks) — no full read into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_cached(key: str) -> Optional[Dict[str, Any]]:
    """Return {"run_id", "created_at"} for an exact key hit, else None."""
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT run_id, created_at FROM runs WHERE key = ?",
            (key,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"run_id": row[0], "created_at": row[1]}


def store(key: str, run_id: str, source_name: str) -> None:
    # Monotonic ordering within one process: time.time() has second
    # resolution, so consecutive stores can tie — a fractional offset
    # guarantees created_at strictly increases (stable "latest" lookups).
    store._order += 1
    created_at = time.time() + store._order / 1_000_000.0
    conn = _conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO runs(key, run_id, source_name, created_at)"
            " VALUES (?, ?, ?, ?)",
            (key, run_id, source_name, created_at))
        conn.commit()
    finally:
        conn.close()


store._order = 0


def find_previous(source_name: str,
                  exclude_run_id: Optional[str] = None) -> Optional[str]:
    """Most recent run_id for the same source file (run_comparison)."""
    conn = _conn()
    try:
        if exclude_run_id:
            row = conn.execute(
                "SELECT run_id FROM runs WHERE source_name = ?"
                " AND run_id != ? ORDER BY created_at DESC LIMIT 1",
                (source_name, exclude_run_id)).fetchone()
        else:
            row = conn.execute(
                "SELECT run_id FROM runs WHERE source_name = ?"
                " ORDER BY created_at DESC LIMIT 1",
                (source_name,)).fetchone()
    finally:
        conn.close()
    return row[0] if row else None