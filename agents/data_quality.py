"""Stage 3 — Data Quality: deterministic pre-cleaning gate (A2.3).

Not a CrewAI agent: plain functions (as the spec requires) invoked from
crew/flows.py (and here from the Flow Review viewer). Runs every check
on the raw extracted CSV against the metadata from Stages 1-2, applies
the deterministic repair, and writes:
    metadata/data_quality_report.json   (DataQualityReport)
    metadata/repair_log.json            (what repair did / did NOT touch)

CLI: python -m agents.data_quality <run_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from shared.core.data_quality import assemble_report
from shared.logger import RunLogger
from shared.schemas import (
    BusinessContext,
    DataProfile,
    DatasetUnderstanding,
)
from shared.utils import init_run_layout, load_config

STAGE = "data_quality"

CHECKS = [
    "schema_checker_tool",
    "invalid_value_checker_tool",
    "business_rules_checker_tool",
    "missingness_analyzer_tool",
    "duplicate_detector_tool",
    "referential_integrity_tool",
    "deterministic_repair_tool",
]


def run_data_quality(run_dir: str | Path,
                     cfg: Dict[str, Any] | None = None,
                     logger: RunLogger | None = None) -> Dict[str, Any]:
    cfg = cfg or load_config(require_key=False)
    run_dir = Path(run_dir)
    run_id = run_dir.name
    run_dir = init_run_layout(run_dir)

    log = logger or RunLogger(run_dir, run_id)
    log.stage_start(STAGE)
    start = time.monotonic()
    try:
        summary = _run(run_dir, cfg, log)
        summary.setdefault("run_id", run_id)
        status = summary.get("status", "failed")
    except Exception as exc:  # noqa: BLE001 -- a failed stage must not crash the run
        log.error(STAGE, f"data quality failed: {exc}")
        status = "failed"
        summary = {"run_id": run_id, "stage": STAGE, "status": status,
                   "error": str(exc)}
    log.stage_end(STAGE, status, time.monotonic() - start)
    summary["log_path"] = str(run_dir / "logs" / "run.jsonl")
    return summary


def _run(run_dir: Path, cfg: Dict[str, Any],
         log: RunLogger) -> Dict[str, Any]:
    profile = _load_profile(run_dir)
    context = _load_context(run_dir)
    understanding = _load_understanding(run_dir)
    extracted = _find_extracted_csv(run_dir)
    if extracted is None:
        raise RuntimeError(
            "No extracted CSV under data/extracted/ - run Stage 1 first.")
    df = pd.read_csv(extracted, encoding="utf-8-sig")

    limits = cfg.get("limits") or {}

    def log_tool(name: str, duration_s: float, status: str) -> None:
        log.tool_call(STAGE, name, status, round(duration_s, 3))

    report, repair_log = assemble_report(
        understanding=understanding, profile=profile, df=df,
        context=context, limits=limits, log_tool=log_tool)

    metadata = run_dir / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "data_quality_report.json").write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8")
    (metadata / "repair_log.json").write_text(
        json.dumps(repair_log, ensure_ascii=False, indent=2),
        encoding="utf-8")

    return {
        "stage": STAGE, "status": report.status,
        "missingness_rate": report.missingness.get("rate", 0.0),
        "missingness_assessment": report.missingness.get("assessment",
                                                         "none"),
        "duplicates": report.duplicates,
        "invalid_columns": sorted(report.invalid.keys()),
        "repair_applied": repair_log["repair_applied"],
        "data_quality_report_path": str(metadata
                                        / "data_quality_report.json"),
        "repair_log_path": str(metadata / "repair_log.json"),
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Artifact IO
# ---------------------------------------------------------------------------


def _load_profile(run_dir: Path) -> DataProfile:
    path = run_dir / "metadata" / "data_profile.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run Stage 1 (ingestion) on this run dir first")
    return DataProfile(**json.loads(path.read_text(encoding="utf-8")))


def _load_context(run_dir: Path) -> BusinessContext:
    path = run_dir / "knowledge" / "business_context.json"
    if not path.exists():
        return BusinessContext(file_name="", generic_mode=True)
    return BusinessContext(**json.loads(path.read_text(encoding="utf-8")))


def _load_understanding(run_dir: Path) -> DatasetUnderstanding:
    path = run_dir / "metadata" / "dataset_understanding.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run Stage 2 (understanding) on this run dir "
            "first")
    return DatasetUnderstanding(**json.loads(
        path.read_text(encoding="utf-8")))


def _find_extracted_csv(run_dir: Path) -> Path | None:
    csvs = sorted((run_dir / "data" / "extracted").glob("*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="data_quality_agent",
        description="Run Insight Forge stage 3 (deterministic data quality "
                    "gate) on a Stage-2 run dir.")
    parser.add_argument("run_dir", help="existing run dir (from Stage 2)")
    args = parser.parse_args(argv)

    summary = run_data_quality(args.run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
