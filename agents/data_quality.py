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


def run_data_quality(run_dir: str | Path,
                     cfg: Dict[str, Any] | None = None,
                     logger: RunLogger | None = None,
                     data_source: str = "extracted") -> Dict[str, Any]:
    """Run stage-3 data-quality checks.

    Parameters
    ----------
    data_source : str
        ``"extracted"`` (default) reads from ``data/extracted/``.
        ``"cleaned"`` reads from ``data/processed/cleaned_data.csv``
        — used for the post-cleaning recheck.
    """
    cfg = cfg or load_config(require_key=False)
    run_dir = Path(run_dir)
    run_id = run_dir.name
    run_dir = init_run_layout(run_dir)

    log = logger or RunLogger(run_dir, run_id)
    log.stage_start(STAGE)
    start = time.monotonic()
    try:
        summary = _run(run_dir, cfg, log, data_source=data_source)
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
         log: RunLogger, data_source: str = "extracted") -> Dict[str, Any]:
    profile = _load_profile(run_dir)
    context = _load_context(run_dir)
    understanding = _load_understanding(run_dir)

    if data_source == "cleaned":
        csv_path = run_dir / "data" / "processed" / "cleaned_data.csv"
        if not csv_path.exists():
            raise RuntimeError(
                "No cleaned CSV under data/processed/ - run Stage 4 first.")
    else:
        csv_path = _find_extracted_csv(run_dir)
        if csv_path is None:
            raise RuntimeError(
                "No extracted CSV under data/extracted/ - run Stage 1 first.")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    limits = cfg.get("limits") or {}

    def log_tool(name: str, duration_s: float, status: str) -> None:
        log.tool_call(STAGE, name, status, round(duration_s, 3))

    is_recheck = data_source == "cleaned"
    report, repair_log, repaired_df = assemble_report(
        understanding=understanding, profile=profile, df=df,
        context=context, limits=limits, log_tool=log_tool,
        skip_repair=is_recheck, with_frame=True)

    metadata = run_dir / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "data_quality_report.json").write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8")
    (metadata / "repair_log.json").write_text(
        json.dumps(repair_log, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # Data contracts (heuristic): the declared shape each column is measured
    # against. Saved for Cleaning's normalization layer + the report.
    contracts: List[Any] = []
    contract_violations: List[Dict[str, Any]] = []
    if not is_recheck:
        from shared.core.contracts import (build_contracts,
                                           save_contracts,
                                           validate_contracts)
        contracts = build_contracts(understanding, df)
        contract_violations = validate_contracts(contracts, df)
        save_contracts(run_dir, contracts)
        (metadata / "contract_violations.json").write_text(
            json.dumps(contract_violations, ensure_ascii=False, indent=2),
            encoding="utf-8")

    validated_path = None
    if not is_recheck:
        # Deep profile (improvement plan 3-5): sentinel-aware missingness,
        # MAD outlier flags, and raw -> repaired impact. Saved once on the
        # raw extraction (not re-checked against cleaned output).
        from shared.core.deep_profile import (deep_missingness_report,
                                              deep_outlier_report,
                                              impact_analysis)
        deep_profile = {
            "missingness": deep_missingness_report(understanding, df),
            "outliers": deep_outlier_report(understanding, df),
            "impact_raw_to_validated": impact_analysis(understanding, df,
                                                       repaired_df),
        }
        (metadata / "deep_profile.json").write_text(
            json.dumps(deep_profile, ensure_ascii=False, indent=2),
            encoding="utf-8")
        # Lineage: the deterministic repair output becomes the official
        # source for Cleaning (repaired = validated_data.csv).
        processed = run_dir / "data" / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        validated_path = processed / "validated_data.csv"
        repaired_df.to_csv(validated_path, index=False,
                           encoding="utf-8-sig")
        _write_lineage(run_dir, profile, csv_path, validated_path, df,
                       repaired_df, repair_log)

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
        "validated_data_path": str(validated_path) if validated_path
        else None,
        "contracts_path": str(metadata / "data_contracts.json")
        if not is_recheck else None,
        "contract_violations_count": len(contract_violations),
        "deep_profile_path": str(metadata / "deep_profile.json")
        if not is_recheck else None,
        "errors": [],
    }


def _write_lineage(run_dir: Path, profile: DataProfile, input_csv: Path,
                   validated_path: Path, df, repaired_df,
                   repair_log: Dict[str, Any]) -> None:
    """Record raw -> validated -> repaired steps in metadata/lineage.json."""
    from shared.core.lineage import (artifact_step, file_sha256,
                                     step, write_lineage)

    repair_ops: List[Dict[str, Any]] = []
    for column, count in (repair_log.get("coerced_to_null") or {}).items():
        repair_ops.append({"op": "coerce_to_null", "column": column,
                           "rows_affected": count})
    for column, detail in (repair_log.get("type_casts") or {}).items():
        repair_ops.append({"op": "type_cast", "column": column,
                           "detail": detail})
    impossible = repair_log.get("impossible_rows_dropped") or {}
    if impossible:
        repair_ops.append({
            "op": "drop_impossible",
            "columns": sorted(impossible.keys()),
            "rows_affected": len({i for rows in impossible.values()
                                  for i in rows}),
        })
    dups = int(repair_log.get("duplicates_removed", 0) or 0)
    if dups:
        repair_ops.append({"op": "dedup", "rows_affected": dups})

    rel_extracted = input_csv.relative_to(run_dir).as_posix() \
        if input_csv.is_relative_to(run_dir) else str(input_csv)
    steps = [
        step(stage="raw", artifact=profile.file_name or "",
             hash=profile.file_hash or "", rows_after=len(df),
             ops=[{"op": "upload", "detail": "source file as provided"}]),
        artifact_step(run_dir, "validated", rel_extracted,
                      rows_before=len(df), rows_after=len(df),
                      ops=[{"op": "parse", "detail": "validated + extracted"}]),
        artifact_step(
            run_dir, "repaired",
            validated_path.relative_to(run_dir).as_posix(),
            rows_before=len(df), rows_after=len(repaired_df),
            ops=repair_ops),
    ]
    write_lineage(run_dir, profile.file_name or "", profile.file_hash or "",
                  steps, source_path=str(input_csv))


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
