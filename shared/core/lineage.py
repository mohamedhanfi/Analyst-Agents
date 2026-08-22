"""Data lineage: the raw -> validated -> repaired -> cleaned -> analysis-ready
chain with per-artifact hashes, row counts and operation logs.

Every stage that transforms the dataset records one lineage step: the
artifact path (relative to the run dir), its sha256, rows before/after and
the operations that produced it. metadata/lineage.json lets any number in
the report be traced back to the source file (auditability, §4 lineage).

The chain is written incrementally: Stage 3 records raw/validated/repaired,
Stage 4 appends cleaned/analysis_ready. Missing steps are tolerated — the
artifact stays readable mid-pipeline.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

STAGES = ("raw", "validated", "repaired", "cleaned", "analysis_ready")

_LINEAGE_PATH = Path("metadata") / "lineage.json"


def file_sha256(path: str | Path) -> str:
    """'sha256:<hex>' of a file on disk (matches shared.utils hashing)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def df_fingerprint(df) -> str:
    """Deterministic in-memory fingerprint: sha256 of the canonical CSV
    serialization (column order and row order are part of the identity)."""
    try:
        payload = df.to_csv(index=False).encode("utf-8")
    except Exception:  # noqa: BLE001 -- a fingerprint must never raise
        return ""
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _rows(path: str | Path) -> Optional[int]:
    """Row count (excluding header) of a CSV, best-effort."""
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            return sum(1 for _ in fh) - 1
    except OSError:
        return None


def step(stage: str,
         artifact: str,
         hash: str = "",
         rows_before: Optional[int] = None,
         rows_after: Optional[int] = None,
         ops: Optional[List[Dict[str, Any]]] = None,
         extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build one lineage entry. `ops` are the transformations that produced
    this artifact from the previous one (deterministic version, affected
    rows, before/after statistics)."""
    return {
        "stage": stage,
        "artifact": artifact,
        "hash": hash,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "ops": ops or [],
        **({"extra": extra} if extra else {}),
    }


def write_lineage(run_dir: str | Path,
                  source_name: str,
                  source_hash: str,
                  steps: List[Dict[str, Any]],
                  source_path: str = "") -> Path:
    """Write metadata/lineage.json (full replace)."""
    run_dir = Path(run_dir)
    payload = {
        "source_file": source_name,
        "source_hash": source_hash,
        "source_path": source_path,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": steps,
    }
    path = run_dir / _LINEAGE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def read_lineage(run_dir: str | Path) -> Dict[str, Any]:
    """Load metadata/lineage.json; empty payload when missing/broken."""
    path = Path(run_dir) / _LINEAGE_PATH
    if not path.exists():
        return {"steps": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"steps": []}


def append_steps(run_dir: str | Path,
                 steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Append steps to an existing lineage file (used by Stage 4)."""
    lineage = read_lineage(run_dir)
    lineage.setdefault("steps", []).extend(steps)
    path = Path(run_dir) / _LINEAGE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lineage, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return lineage


def artifact_step(run_dir: str | Path, stage: str, rel_path: str,
                  ops: Optional[List[Dict[str, Any]]] = None,
                  rows_before: Optional[int] = None,
                  rows_after: Optional[int] = None) -> Dict[str, Any]:
    """Build a step for an on-disk CSV artifact (hash + row counts
    computed from the file itself)."""
    full = Path(run_dir) / rel_path
    return step(
        stage=stage,
        artifact=rel_path,
        hash=file_sha256(full) if full.is_file() else "",
        rows_before=rows_before,
        rows_after=rows_after if rows_after is not None else _rows(full),
        ops=ops,
    )