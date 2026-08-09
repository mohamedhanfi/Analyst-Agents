"""Config loading (config.yaml + env) and run_id allocation.

Loads `.env`, yaml config, resolves API key from env, and mints unique
run_ids under `runs/` with an atomic mkdir (concurrency-safe, §5).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
RUNS_DIR = ROOT_DIR / "runs"

RUN_SUBDIRS = ("data/raw", "data/extracted", "data/processed",
               "knowledge", "metadata", "outputs/charts", "logs")


def load_config(path: str | os.PathLike | None = None,
                require_key: bool = True) -> Dict[str, Any]:
    """Load config.yaml, merge env values, resolve the API key.

    Raises RuntimeError if require_key and the configured API key env
    var is missing/empty. Tools that are pure computation (file
    validation, reader) pass require_key=False — they need limits only.
    """
    load_dotenv(ROOT_DIR / ".env")
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    llm = cfg.setdefault("llm", {})
    key_env = llm.get("api_key_env", "OPENROUTER_API_KEY")
    api_key = os.getenv(key_env, "").strip()
    if require_key and not api_key:
        raise RuntimeError(
            f"Missing API key: env var '{key_env}' is empty. "
            f"Copy .env.example to .env and fill it in."
        )
    llm["api_key"] = api_key
    return cfg


def allocate_run_id(runs_dir: str | os.PathLike = RUNS_DIR) -> Tuple[str, Path]:
    """Create the next run directory atomically and return (run_id, dir).

    run_id format: run_<YYYYmmdd_HHMMSS>_<seq>. Atomic mkdir makes parallel
    runs safe — unique seq per second.
    """
    base = Path(runs_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for seq in range(1, 10_000):
        run_id = f"run_{stamp}_{seq}"
        candidate = base / run_id
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return run_id, candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate a run_id under {base} (seq ceiling hit)")


def init_run_layout(run_dir: str | os.PathLike) -> Path:
    """Create the standard per-run subfolder tree; returns the run dir."""
    run = Path(run_dir)
    for sub in RUN_SUBDIRS:
        (run / sub).mkdir(parents=True, exist_ok=True)
    return run