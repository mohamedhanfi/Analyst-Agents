"""Stage 4 — Cleaning: CrewAI agent + deterministic path (§2.4).

Consumes a Stage-3 run dir (understanding + profile + business context +
data_quality_report + extracted CSV). The LLM picks the strategy
(cleaning_strategy_tool); Python normalizes it (Python-authoritative),
executes it (execute_strategy), persists the versioned attempt, and re-checks
DQ on the cleaned output; on `needs_repair` it re-runs up to
`limits.cleaning_max_rechecks` (3), then auto-verdicts
`cleaning_retry_limit_exceeded`.

CLI: python -m agents.cleaning_agent <run_dir> [--crew]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from crewai import Agent, Crew, Process, Task

from shared.core.cleaning import (
    assemble_cleaning_result,
    build_strategy,
    execute_strategy,
    normalize_strategy,
    persist_attempt,
)
from shared.core.data_quality import assemble_report
from shared.llm import build_llm
from shared.logger import RunLogger
from shared.prompt_guard import data_note
from shared.schemas import (
    BusinessContext,
    CleaningResult,
    DataProfile,
    DataQualityReport,
    DatasetUnderstanding,
)
from shared.tools import (
    cleaning_strategy_tool,
    dedup_tool,
    dq_recheck_tool,
    fillna_tool,
    flag_column_tool,
    iqr_outlier_tool,
    type_caster_tool,
)
from shared.utils import init_run_layout, load_config

STAGE = "cleaning"

CLEANING_TOOLS = [
    cleaning_strategy_tool,
    fillna_tool,
    flag_column_tool,
    type_caster_tool,
    dedup_tool,
    iqr_outlier_tool,
    dq_recheck_tool,
]


# ---------------------------------------------------------------------------
# CrewAI assembly
# ---------------------------------------------------------------------------


def build_cleaning_agent(cfg: Dict[str, Any]) -> Agent:
    a_cfg = cfg["agents"]["cleaning"]
    return Agent(
        role=a_cfg["role"],
        goal=a_cfg["goal"],
        backstory=a_cfg["backstory"],
        llm=build_llm(cfg, "cleaning"),
        tools=CLEANING_TOOLS,
        verbose=False,
        memory=False,
        allow_delegation=False,
        cache=False,
        max_iter=12,
    )


def build_cleaning_tasks(agent: Agent, run_dir: str | Path,
                         cfg: Dict[str, Any]) -> List[Task]:
    understanding = _load_understanding(Path(run_dir))
    dq_report = _load_dq_report(Path(run_dir))
    understanding_json = understanding.model_dump_json()
    dq_json = dq_report.model_dump_json()

    decide_cleaning_strategy = Task(
        description=(
            f"Decide the cleaning strategy for run '{run_dir}' (§2.4).\n"
            "Step 1: call cleaning_strategy_tool with understanding_json="
            f"'{understanding_json}' and dq_report_json='{dq_json}' (leave "
            "proposed_strategy_json empty). It returns the deterministic "
            "strategy from the role x missingness table.\n"
            "Step 2: review it. You MAY change actions for flagged measures "
            "(e.g. 'drop_negative' for negative revenue, 'iqr' via outliers "
            "{column: 'flag'|'drop'}) by calling cleaning_strategy_tool "
            "again with proposed_strategy_json. Allowed actions: "
            "keep, median_fill, median_fill_flag, mode_fill, unknown_fill, "
            "flag_and_preserve, keep_flag, drop_row, drop_column, "
            "drop_negative.\n"
            "Step 3: keep calling until errors is empty.\n"
            "Return ONLY the final strategy JSON: {\"columns\": "
            "[{\"column\", \"action\"}], \"deduplicate\": bool, "
            "\"outliers\": {column: \"flag\"|\"drop\"}}.\n"
            + data_note()
        ),
        expected_output=(
            'The validated strategy JSON as returned by the tool. No prose.'
        ),
        agent=agent,
    )

    execute_cleaning = Task(
        description=(
            f"Finalize the cleaning strategy for run '{run_dir}'.\n"
            "Python executes it deterministically: role-based type casts, "
            "fills (median/mode/'Unknown' only — never invented values), "
            "`*_missing_flag` features for non-random missingness, dedup, "
            "IQR outlier handling, and drops flagged negatives/duplicates. "
            "Nothing is written until the whole strategy is validated.\n"
            "You may preview single operations with fillna_tool, "
            "flag_column_tool, type_caster_tool, dedup_tool, "
            "iqr_outlier_tool, and refine the strategy via "
            "cleaning_strategy_tool before finalizing.\n"
            "Return ONLY the final strategy JSON (same shape as task 1)."
        ),
        expected_output=(
            'The final strategy JSON. No prose.'
        ),
        agent=agent,
    )

    recheck_data_quality = Task(
        description=(
            f"Re-check data quality after cleaning run '{run_dir}' (§2.4).\n"
            "Python re-runs Agent 3's checks on the cleaned output. If the "
            f"status is not 'passed', the run retries cleaning (max "
            "`limits.cleaning_max_rechecks` = 3 attempts) then auto-verdicts "
            "`cleaning_retry_limit_exceeded`.\n"
            "You may refine the strategy once more via cleaning_strategy_tool "
            "if a signal can be fixed; otherwise leave it unchanged.\n"
            "Return ONLY the JSON {\"status\": \"passed\"|\"needs_repair\", "
            "\"attempt\": <n>}."
        ),
        expected_output=(
            '{"status": "passed|needs_repair", "attempt": <n>}'
        ),
        agent=agent,
    )

    return [decide_cleaning_strategy, execute_cleaning,
            recheck_data_quality]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_cleaning(run_dir: str | Path,
                 cfg: Dict[str, Any] | None = None,
                 logger: RunLogger | None = None,
                 use_crew: bool = False) -> Dict[str, Any]:
    cfg = cfg or load_config(require_key=bool(use_crew))
    run_dir = Path(run_dir)
    run_id = run_dir.name
    run_dir = init_run_layout(run_dir)

    log = logger or RunLogger(run_dir, run_id)
    log.stage_start(STAGE)
    start = time.monotonic()
    try:
        if use_crew:
            summary = _run_crew(run_dir, cfg, log)
        else:
            summary = _run_deterministic(run_dir, cfg, log)
        summary.setdefault("run_id", run_id)
        status = summary.get("status", "failed")
    except Exception as exc:  # noqa: BLE001 -- a failed stage must not crash the run
        log.error(STAGE, f"cleaning failed: {exc}")
        status = "failed"
        summary = {"run_id": run_id, "stage": STAGE, "status": status,
                   "error": str(exc)}
    log.stage_end(STAGE, status, time.monotonic() - start)
    summary["log_path"] = str(run_dir / "logs" / "run.jsonl")
    return summary


def _run_deterministic(run_dir: Path, cfg: Dict[str, Any],
                       log: RunLogger) -> Dict[str, Any]:
    understanding, profile, context, dq_report, df = _load_all(run_dir)
    limits = cfg.get("limits") or {}

    t0 = time.monotonic()
    strategy = build_strategy(understanding, dq_report)
    log.tool_call(STAGE, "cleaning_strategy_tool", "passed",
                  time.monotonic() - t0)

    result, final_status = _clean_with_rechecks(
        run_dir, df, strategy, understanding, profile, context, limits, log)
    _save_result(run_dir, result)

    return _summary(result, run_dir, final_status)


def _run_crew(run_dir: Path, cfg: Dict[str, Any],
              log: RunLogger) -> Dict[str, Any]:
    agent = build_cleaning_agent(cfg)
    tasks = build_cleaning_tasks(agent, str(run_dir), cfg)
    crew = Crew(agents=[agent], tasks=tasks, process=Process.sequential,
                verbose=False, cache=False)
    t0 = time.monotonic()
    result = crew.kickoff(inputs={})
    log.info(STAGE, "crew kickoff finished",
             duration_s=round(time.monotonic() - t0, 3))

    result_obj, final_status, warnings = _finalize_cleaning(
        run_dir, result, cfg, log)
    for warning in warnings:
        log.fallback(STAGE, warning)

    return _summary(result_obj, run_dir, final_status)


def _finalize_cleaning(run_dir: Path, result, cfg: Dict[str, Any],
                       log: RunLogger
                       ) -> Tuple[CleaningResult, str, List[str]]:
    understanding, profile, context, dq_report, df = _load_all(run_dir)
    limits = cfg.get("limits") or {}
    warnings: List[str] = []

    outputs = getattr(result, "tasks_output", None) or []
    raws = [str(getattr(t, "raw", "") or getattr(t, "output", "") or "")
            for t in outputs]
    raw_strategy = _parse_json(raws[0] if len(raws) > 0 else "")

    if isinstance(raw_strategy, dict) and raw_strategy.get("columns"):
        strategy, strategy_errors = normalize_strategy(
            raw_strategy, understanding, dq_report)
        if strategy_errors:
            warnings.append("llm_strategy_partially_rejected")
    else:
        strategy = build_strategy(understanding, dq_report)
        warnings.append("llm_strategy_missing_default_used")

    result_obj, final_status = _clean_with_rechecks(
        run_dir, df, strategy, understanding, profile, context, limits, log)
    _save_result(run_dir, result_obj)
    return result_obj, final_status, warnings


def _clean_with_rechecks(run_dir: Path, df: pd.DataFrame,
                         strategy: Dict[str, Any],
                         understanding: DatasetUnderstanding,
                         profile: DataProfile,
                         context: BusinessContext,
                         limits: Dict[str, Any],
                         log: RunLogger
                         ) -> Tuple[CleaningResult, str]:
    max_rechecks = int((limits or {}).get("cleaning_max_rechecks", 3))
    rows_before = len(df)
    current = df
    attempt = 0
    result = None
    final_status = "failed"

    while attempt < max_rechecks:
        attempt += 1
        t0 = time.monotonic()
        cleaned, ops = execute_strategy(current, strategy, understanding)
        log.tool_call(STAGE, "execute_cleaning", "passed",
                      time.monotonic() - t0)

        persist_attempt(run_dir, cleaned, attempt)

        t0 = time.monotonic()
        final_status, _ = _recheck_data_quality(
            cleaned, understanding, profile, context, limits)
        log.tool_call(STAGE, "dq_recheck_tool", final_status,
                      time.monotonic() - t0)

        dups, casts, flags, outliers = _op_stats(ops)
        status = "passed" if final_status == "passed" else "failed"
        result = assemble_cleaning_result(
            attempt=attempt, rows_before=rows_before, rows_after=len(cleaned),
            duplicates_removed=dups, type_casts=casts, flags_created=flags,
            outliers=outliers, status=status)
        if final_status == "passed":
            break
        current = cleaned

    if final_status != "passed":
        log.fallback(STAGE, "cleaning_retry_limit_exceeded")

    return result, final_status


def _recheck_data_quality(df: pd.DataFrame,
                          understanding: DatasetUnderstanding,
                          profile: DataProfile,
                          context: BusinessContext,
                          limits: Dict[str, Any]) -> Tuple[str, Any]:
    typed, _ = execute_strategy(
        df, {"columns": [], "deduplicate": False, "outliers": {}},
        understanding)
    report, _ = assemble_report(understanding=understanding, profile=profile,
                                df=typed, context=context, limits=limits,
                                skip_repair=True)
    return report.status, report


def _op_stats(ops: List[Dict[str, Any]]):
    dups = sum(o.get("rows_affected", 0) for o in ops
               if o.get("op") == "dedup")
    casts = {o["column"]: o["detail"] for o in ops
             if o.get("op") == "type_cast"}
    flags = sorted({o["flag"] for o in ops if o.get("op") == "flag_column"})
    outliers: Dict[str, int] = {}
    for o in ops:
        op = str(o.get("op", ""))
        if op.startswith("iqr_outlier"):
            outliers[o["column"]] = o.get("rows_affected", 0)
    return dups, casts, flags, outliers


def _summary(result: CleaningResult, run_dir: Path,
             final_dq_status: str) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "status": result.status,
        "final_dq_status": final_dq_status,
        "attempt": result.attempt,
        "rows_before": result.rows_before,
        "rows_after": result.rows_after,
        "duplicates_removed": result.duplicates_removed,
        "flags_created": result.flags_created,
        "type_casts": result.type_casts,
        "outliers": result.outliers,
        "cleaned_data_path": str(run_dir / "data" / "processed"
                                 / "cleaned_data.csv"),
        "cleaning_result_path": str(run_dir / "metadata"
                                    / "cleaning_result.json"),
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Artifact IO
# ---------------------------------------------------------------------------


def _load_profile(run_dir: Path) -> DataProfile:
    path = run_dir / "metadata" / "data_profile.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run Stage 1 (ingestion) first")
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
            f"{path} missing - run Stage 2 (understanding) first")
    return DatasetUnderstanding(**json.loads(path.read_text(encoding="utf-8")))


def _load_dq_report(run_dir: Path) -> DataQualityReport:
    path = run_dir / "metadata" / "data_quality_report.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run Stage 3 (data quality) first")
    return DataQualityReport(**json.loads(path.read_text(encoding="utf-8")))


def _find_extracted_csv(run_dir: Path) -> Path | None:
    csvs = sorted((run_dir / "data" / "extracted").glob("*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def _load_all(run_dir: Path):
    profile = _load_profile(run_dir)
    context = _load_context(run_dir)
    understanding = _load_understanding(run_dir)
    dq_report = _load_dq_report(run_dir)
    extracted = _find_extracted_csv(run_dir)
    if extracted is None:
        raise RuntimeError(
            "No extracted CSV under data/extracted/ - run Stage 1 first.")
    df = pd.read_csv(extracted, encoding="utf-8-sig")
    return understanding, profile, context, dq_report, df


def _save_result(run_dir: Path, result: CleaningResult) -> None:
    metadata = run_dir / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "cleaning_result.json").write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON parsing + CLI
# ---------------------------------------------------------------------------


def _parse_json(raw: str) -> Any:
    start = raw.find("{") if "{" in raw else -1
    end = raw.rfind("}")
    try:
        if start == -1 or end <= start:
            return json.loads(raw.strip()) if raw.strip() else None
        return json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, TypeError):
        return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cleaning_agent",
        description="Run Insight Forge stage 4 (cleaning) on a Stage-3 "
                    "run dir.")
    parser.add_argument("run_dir", help="existing run dir (from Stage 3)")
    parser.add_argument("--crew", action="store_true",
                        help="run via the real CrewAI agent (requires API key)")
    args = parser.parse_args(argv)

    summary = run_cleaning(args.run_dir, use_crew=args.crew)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
