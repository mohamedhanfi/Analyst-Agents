"""Large-data I/O helpers (improvement-plan item: chunked/large files).

Deterministic, dependency-light:

* ``read_dataframe`` — transparent CSV/Parquet read; files above
  ``LARGE_FILE_MB`` are converted once to a Parquet cache next to the CSV
  (when an engine is installed) so later stages never re-parse the text.
* ``column_stats_chunked`` — missing/nunique/min/max/sum per column in
  fixed-size chunks, so profiling a 10M-row file never loads it whole.
* ``estimate_csv_rows`` — cheap newline count for progress/size reporting.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import pandas as pd

LARGE_FILE_MB = 200.0          # above this -> parquet cache path
LARGE_ROWS = 250_000           # above this -> chunked stats preferred
CHUNK_SIZE = 100_000


def _parquet_engine() -> Optional[str]:
    for engine in ("pyarrow", "fastparquet"):
        try:
            __import__(engine)
            return engine
        except ImportError:
            continue
    return None


def file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def estimate_csv_rows(path: Path) -> Optional[int]:
    """Fast newline count (no pandas) — None when unreadable."""
    try:
        with open(path, "rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def parquet_cache_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".parquet")


def convert_to_parquet(csv_path: Path, chunk_size: int = CHUNK_SIZE
                       ) -> Optional[Path]:
    """One-time CSV -> Parquet conversion (chunked); None when no engine."""
    engine = _parquet_engine()
    if engine is None:
        return None
    target = parquet_cache_path(csv_path)
    try:
        first = True
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size,
                                 encoding="utf-8-sig"):
            mode = "w" if first else "a"
            chunk.to_parquet(target, engine=engine, index=False,
                             append=not first)
            first = False
        return target if target.is_file() else None
    except Exception:  # noqa: BLE001 -- cache is best-effort
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def read_dataframe(path: str | Path, *, force_parquet: bool = True
                   ) -> pd.DataFrame:
    """Read a CSV or Parquet file, transparently using the Parquet cache
    for large files. Never raises on cache failures — falls back to CSV."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    cache = parquet_cache_path(path)
    if force_parquet and file_size_mb(path) >= LARGE_FILE_MB:
        if cache.is_file():
            try:
                return pd.read_parquet(cache)
            except Exception:  # noqa: BLE001 -- stale cache -> rebuild
                pass
        built = convert_to_parquet(path)
        if built is not None:
            return pd.read_parquet(built)
    return pd.read_csv(path, encoding="utf-8-sig")


def iter_csv_chunks(path: str | Path, chunk_size: int = CHUNK_SIZE
                    ) -> Iterator[pd.DataFrame]:
    with pd.read_csv(path, chunksize=chunk_size, encoding="utf-8-sig") as r:
        yield from r


def column_stats_chunked(path: str | Path, chunk_size: int = CHUNK_SIZE
                         ) -> Dict[str, Dict[str, Any]]:
    """Per-column {missing, nunique, min, max, sum} computed in chunks."""
    stats: Dict[str, Dict[str, Any]] = {}
    total_rows = 0
    for chunk in iter_csv_chunks(path, chunk_size):
        total_rows += len(chunk)
        for column in chunk.columns:
            entry = stats.setdefault(column, {
                "missing": 0, "nunique": set(), "min": None, "max": None,
                "sum": 0.0})
            entry["missing"] += int(chunk[column].isna().sum())
            entry["nunique"].update(
                str(v) for v in chunk[column].dropna().unique())
            numeric = pd.to_numeric(chunk[column], errors="coerce")
            if numeric.notna().any():
                lo, hi = float(numeric.min()), float(numeric.max())
                entry["min"] = lo if entry["min"] is None \
                    else min(entry["min"], lo)
                entry["max"] = hi if entry["max"] is None \
                    else max(entry["max"], hi)
                entry["sum"] += float(numeric.sum())
    return {
        "rows": total_rows,
        "columns": {
            column: {"missing": entry["missing"],
                     "nunique": len(entry["nunique"]),
                     "min": entry["min"], "max": entry["max"],
                     "sum": round(entry["sum"], 6)}
            for column, entry in stats.items()
        },
    }


def is_large(path: str | Path) -> bool:
    path = Path(path)
    return file_size_mb(path) >= LARGE_FILE_MB


def read_analysis_dataframe(run_dir: str | Path) -> Optional[pd.DataFrame]:
    """Prefer analysis_ready (parquet cache included), then cleaned."""
    run_dir = Path(run_dir)
    for name in ("analysis_ready.csv", "cleaned_data.csv"):
        candidate = run_dir / "data" / "processed" / name
        if candidate.is_file():
            try:
                return read_dataframe(candidate)
            except Exception:  # noqa: BLE001 -- fall through to next
                continue
    return None