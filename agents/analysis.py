"""Stage 5a — Analysis compute layer: CrewAI agent + deterministic path (§2.5).

Consumes a Stage-4 run dir (understanding + analysis plan + cleaned CSV).
Python computes everything on the FULL dataset: whitelist DSL KPIs
(execute_plan), the statistical suite (run_statistical_suite), and the chart
plan (plan_charts). The LLM only selects which facts deserve a chart and may
re-rank with a reason — the shape and every number stay Python's call
(Python-authoritative). One EvidenceRegistry is shared across all three so
every value/chart carries a unique evidence_id.

CLI: python -m agents.analysis <run_dir> [--crew]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from crewai import Agent, Crew, Process, Task

from analysis.chart_planner import plan_charts
from analysis.chart_renderer import render_all
from analysis.dsl_executor import execute_plan
from analysis.evidence import EvidenceRegistry
from analysis.generic import run_statistical_suite
from shared.formatting import parse_json as _parse_json
from shared.llm import build_llm
from shared.logger import RunLogger
from shared.schemas import (AnalysisPlan, ChartMetadata, CleaningResult,
                            DatasetUnderstanding, KpiResult, StatisticalResult)
from shared.tools import (
    chart_planner_tool,
    chart_renderer_tool,
    dsl_executor_tool,
    evidence_registry_tool,
    statistical_suite_tool,
)
from shared.utils import init_run_layout, load_config

STAGE = "analysis"

ANALYSIS_TOOLS = [
    dsl_executor_tool,
    statistical_suite_tool,
    chart_planner_tool,
    chart_renderer_tool,
    evidence_registry_tool,
]


# ---------------------------------------------------------------------------
# CrewAI assembly
# ---------------------------------------------------------------------------

def build_analysis_agent(cfg: Dict[str, Any]) -> Agent:
    a_cfg = cfg["agents"]["analyst"]
    return Agent(
        role=a_cfg["role"],
        goal=a_cfg["goal"],
        backstory=a_cfg["backstory"],
        llm=build_llm(cfg, "analyst"),
        tools=ANALYSIS_TOOLS,
        verbose=False,
        memory=False,
        allow_delegation=False,
        cache=False,
        max_iter=12,
    )


def build_analysis_tasks(agent: Agent, run_dir: str | Path,
                         cfg: Dict[str, Any]) -> List[Task]:
    understanding = _load_understanding(Path(run_dir))
    plan = _load_plan(Path(run_dir))
    understanding_json = understanding.model_dump_json()
    plan_json = plan.model_dump_json()
    csv_path = str(_find_cleaned_csv(Path(run_dir)))
    limits_json = json.dumps(cfg.get("limits") or {})

    select_kpis = Task(
        description=(
            f"Select the KPIs worth computing for run '{run_dir}' (§2.5).\n"
            "Step 1: read the analysis_plan.json candidate_kpis (the plan is "
            "already DSL-whitelisted — never invent formulas).\n"
            "Step 2: review each candidate against the cleaned data; drop "
            "only candidates that are meaningless (e.g. no signal, constant "
            "column). Prefer keeping the plan as-is.\n"
            "Return ONLY the JSON {\"kpi_ids\": [\"KPI-001\", ...]}."
        ),
        expected_output='{"kpi_ids": ["KPI-001", ...]}',
        agent=agent,
    )

    run_dsl_and_stats = Task(
        description=(
            f"Compute KPIs and the statistical suite for run '{run_dir}'.\n"
            "Python executes every DSL op over ALL rows — never sample.\n"
            "Step 1: call dsl_executor_tool with csv_path, understanding_json, "
            "plan_json.\n"
            "Step 2: call statistical_suite_tool with the same inputs; tests "
            "come from plan.statistical_tests (default when empty).\n"
            "Both return JSON with evidence_id per value. Every number is "
            "Python-computed; report only what the tools returned.\n"
            "Return ONLY the JSON {\"status\": \"computed\", "
            "\"kpi_count\": <n>, \"test_count\": <m>}."
        ),
        expected_output='{"status": "computed", "kpi_count": <n>, "test_count": <m>}',
        agent=agent,
    )

    rank_chart_candidates = Task(
        description=(
            f"Rank the chart candidates for run '{run_dir}' (§2.5).\n"
            "Step 1: call chart_planner_tool with csv_path, understanding_json, "
            "plan_json, limits_json. It returns the deterministic chart plan "
            "(shape decided by the data rule table) + charts_truncated.\n"
            "Step 2: review it. You MAY re-order candidates and write a short "
            "reason per chart; you MUST NOT change any shape/kind — drawing "
            "stays Python's call. Charts with reliability 'low_n' should rank "
            "last.\n"
            "Return ONLY the JSON {\"rank\": [{\"chart_id\", \"reason\"}], "
            "\"charts_truncated\": bool}."
        ),
        expected_output=(
            '{"rank": [{"chart_id": "CH-001", "reason": "..."}], '
            '"charts_truncated": bool}'
        ),
        agent=agent,
    )

    return [select_kpis, run_dsl_and_stats, rank_chart_candidates]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_analysis(run_dir: str | Path,
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
        log.error(STAGE, f"analysis failed: {exc}")
        status = "failed"
        summary = {"run_id": run_id, "stage": STAGE, "status": status,
                   "error": str(exc)}
    log.stage_end(STAGE, status, time.monotonic() - start)
    summary["log_path"] = str(run_dir / "logs" / "run.jsonl")
    return summary


def _run_deterministic(run_dir: Path, cfg: Dict[str, Any],
                       log: RunLogger) -> Dict[str, Any]:
    kpis, stats, charts, truncated, evidence_count = _compute(run_dir, cfg, log)
    return _summary(run_dir, kpis, stats, charts, truncated, evidence_count,
                    errors=[])


def _run_crew(run_dir: Path, cfg: Dict[str, Any],
              log: RunLogger) -> Dict[str, Any]:
    agent = build_analysis_agent(cfg)
    tasks = build_analysis_tasks(agent, run_dir, cfg)
    crew = Crew(agents=[agent], tasks=tasks, process=Process.sequential,
                verbose=False, cache=False)
    t0 = time.monotonic()
    result = crew.kickoff(inputs={})
    log.info(STAGE, "crew kickoff finished",
             duration_s=round(time.monotonic() - t0, 3))

    warnings = _finalize_analysis(run_dir, result, cfg, log)
    for warning in warnings:
        log.fallback(STAGE, warning)

    kpis, stats, charts, truncated, evidence_count = _compute(run_dir, cfg, log)
    return _summary(run_dir, kpis, stats, charts, truncated, evidence_count,
                    errors=warnings)


def _finalize_analysis(run_dir: Path, result, cfg: Dict[str, Any],
                       log: RunLogger) -> List[str]:
    """Inspect the crew output; the compute itself is always Python's.

    Returns fallback warnings when the LLM produced no usable signal. The
    artifacts are re-computed deterministically by the caller, so a bad LLM
    response can never poison numbers/charts.
    """
    outputs = getattr(result, "tasks_output", None) or []
    raws = [str(getattr(t, "raw", "") or getattr(t, "output", "") or "")
            for t in outputs]
    rerank = _parse_json(raws[-1] if len(raws) > 0 else "")
    warnings: List[str] = []
    if not isinstance(rerank, dict) or "rank" not in rerank:
        warnings.append("llm_rerank_missing_default_used")
    else:
        log.info(STAGE, "llm rerank received",
                 chart_count=len(rerank.get("rank", [])))
    return warnings


# ---------------------------------------------------------------------------
# Computation (Python-authoritative)
# ---------------------------------------------------------------------------

def _compute(run_dir: Path, cfg: Dict[str, Any],
             log: RunLogger) -> Tuple[List[KpiResult],
                                      List[StatisticalResult],
                                      List[ChartMetadata],
                                      bool,
                                      int]:
    understanding = _load_understanding(run_dir)
    plan = _load_plan(run_dir)
    cleaned = _find_cleaned_csv(run_dir)
    if cleaned is None:
        raise RuntimeError(
            "No cleaned_data.csv under data/processed/ - run Stage 4 first.")
    df = pd.read_csv(cleaned, encoding="utf-8-sig")
    limits = cfg.get("limits") or {}
    max_chart_count = int(limits.get("max_chart_count", 20) or 20)

    registry = EvidenceRegistry(
        run_id=run_dir.name,
        file_hash=_file_hash(cleaned),
        transformations=_cleaning_transformations(run_dir),
    )

    t0 = time.monotonic()
    kpis = execute_plan(df, plan, registry)
    log.tool_call(STAGE, "dsl_executor_tool", "passed",
                  time.monotonic() - t0, note=f"{len(kpis)} kpis")

    t0 = time.monotonic()
    tests = plan.statistical_tests or None
    stats = run_statistical_suite(df, understanding, registry, tests=tests)
    log.tool_call(STAGE, "statistical_suite_tool", "passed",
                  time.monotonic() - t0, note=f"{len(stats)} results")

    t0 = time.monotonic()
    charts, truncated = plan_charts(df, plan, understanding, registry,
                                    max_chart_count=max_chart_count)
    log.tool_call(STAGE, "chart_planner_tool", "passed",
                  time.monotonic() - t0,
                  note=f"{len(charts)} charts, truncated={truncated}")

    t0 = time.monotonic()
    render_all(charts, df, kpis, run_dir / "charts")
    _fill_chart_paths(charts, run_dir / "charts")
    log.tool_call(STAGE, "chart_renderer_tool", "passed",
                  time.monotonic() - t0,
                  note=f"{len(charts)} svg files")

    _save_outputs(run_dir, kpis, stats, charts, truncated, registry)
    return kpis, stats, charts, truncated, len(registry)


# ---------------------------------------------------------------------------
# Summary + IO
# ---------------------------------------------------------------------------

def _summary(run_dir: Path, kpis: List[KpiResult],
             stats: List[StatisticalResult], charts: List[ChartMetadata],
             truncated: bool, evidence_count: int,
             errors: List[str]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "status": "passed",
        "kpi_count": len(kpis),
        "statistical_test_count": len(stats),
        "chart_count": len(charts),
        "charts_truncated": truncated,
        "evidence_count": evidence_count,
        "kpis_path": str(run_dir / "outputs" / "kpis.json"),
        "statistical_results_path": str(run_dir / "outputs"
                                        / "statistical_results.json"),
        "chart_metadata_path": str(run_dir / "metadata"
                                   / "chart_metadata.json"),
        "evidence_registry_path": str(run_dir / "outputs"
                                     / "evidence_registry.json"),
        "errors": errors,
    }


def _file_hash(csv_path: Path) -> str:
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _cleaning_transformations(run_dir: Path) -> List[str]:
    path = run_dir / "metadata" / "cleaning_result.json"
    if not path.exists():
        return []
    result = CleaningResult(**json.loads(path.read_text(encoding="utf-8")))
    ops: List[str] = []
    if result.duplicates_removed > 0:
        ops.append("removed_duplicates")
    ops.extend(f"type_cast_{col}" for col in result.type_casts)
    ops.extend(f"flag_{flag}" for flag in result.flags_created)
    ops.extend(f"iqr_outlier_{col}" for col in result.outliers)
    return ops


def _fill_chart_paths(charts: List[ChartMetadata], charts_dir: Path) -> None:
    """Record charts/<chart_id>.svg in chart_metadata.json (chart_path)."""
    for chart in charts:
        chart.chart_path = str(charts_dir / f"{chart.chart_id}.svg")


def _save_outputs(run_dir: Path, kpis: List[KpiResult],
                  stats: List[StatisticalResult],
                  charts: List[ChartMetadata], truncated: bool,
                  registry: EvidenceRegistry) -> None:
    outputs = run_dir / "outputs"
    metadata = run_dir / "metadata"
    outputs.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)

    (outputs / "kpis.json").write_text(
        json.dumps({"kpis": [k.model_dump() for k in kpis]},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    (outputs / "statistical_results.json").write_text(
        json.dumps({"results": [s.model_dump() for s in stats]},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    (metadata / "chart_metadata.json").write_text(
        json.dumps({"charts": [c.model_dump() for c in charts],
                    "charts_truncated": truncated},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    registry.save(outputs / "evidence_registry.json")


def _load_understanding(run_dir: Path) -> DatasetUnderstanding:
    path = run_dir / "metadata" / "dataset_understanding.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run Stage 2 (understanding) first")
    return DatasetUnderstanding(**json.loads(path.read_text(encoding="utf-8")))


def _load_plan(run_dir: Path) -> AnalysisPlan:
    path = run_dir / "metadata" / "analysis_plan.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run Stage 2 (understanding) first")
    return AnalysisPlan(**json.loads(path.read_text(encoding="utf-8")))


def _find_cleaned_csv(run_dir: Path) -> Path | None:
    processed = run_dir / "data" / "processed"
    latest = processed / "cleaned_data.csv"
    if latest.exists():
        return latest
    attempts = sorted(processed.glob("cleaned_data_attempt_*.csv"))
    return attempts[-1] if attempts else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="analysis_agent",
        description="Run Insight Forge stage 5a (analysis compute) on a "
                    "Stage-4 run dir.")
    parser.add_argument("run_dir", help="existing run dir (from Stage 4)")
    parser.add_argument("--crew", action="store_true",
                        help="run via the real CrewAI agent (requires API key)")
    args = parser.parse_args(argv)

    summary = run_analysis(args.run_dir, use_crew=args.crew)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
