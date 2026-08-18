"""Stage 2 — Data Understanding: CrewAI agent + deterministic path (§2.2).

Consumes a Stage-1 run dir (data_profile.json + business_context.json +
the 20-row PII-redacted sample — never raw cells). Three sequential tasks
run INSIDE the same agent (planning is a 2nd-class task, not a 9th agent):
classify_column_roles -> detect_domain_and_entities -> build_analysis_plan.

use_crew=False runs the same pipeline deterministically (role rules +
domain keyword heuristic + default whitelist plan) so the path is
unit-testable without an API key. Python stays authoritative: final roles,
domain and plan are validated/rebuilt in _finalize_* before being written.

CLI: python -m agents.understanding_agent <run_dir> [--crew]
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
from pydantic import BaseModel, Field

from shared.core.understanding import (
    ColumnProfiler,
    _is_numeric,
    assemble_understanding,
    build_analysis_plan,
    default_plan,
    detect_domain_heuristic,
)
from shared.llm import build_llm, complete_json
from shared.logger import RunLogger
from shared.prompt_guard import data_note
from shared.schemas import AnalysisPlan, BusinessContext, DataProfile
from shared.tools import (
    column_profiler_tool,
    dsl_plan_builder_tool,
    domain_classifier_tool,
)
from shared.utils import init_run_layout, load_config

STAGE = "understanding"

UNDERSTANDING_TOOLS = [
    column_profiler_tool,
    domain_classifier_tool,
    dsl_plan_builder_tool,
]

# Task A.2: LLM fallback verdict for identifier-like numeric columns in the
# ambiguous band (0.3 <= heuristic score < threshold).
class IdentifierVerdict(BaseModel):
    role: str = Field(pattern="^(identifier|measure|categorical)$")
    confidence: float = 0.0
    reason: str = ""


def _validate_identifier_verdicts(payload: Any) -> Tuple[bool, List[str]]:
    if not isinstance(payload, dict) or not payload:
        return False, ["response must be a non-empty object"]
    errors: List[str] = []
    for name, raw in payload.items():
        try:
            verdict = IdentifierVerdict(**raw) if isinstance(raw, dict) else None
        except Exception as exc:  # noqa: BLE001 -- collected per entry
            errors.append(f"{name}: {exc}")
            continue
        if verdict is None:
            errors.append(f"{name}: verdict must be an object")
        elif not 0.0 <= verdict.confidence <= 1.0:
            errors.append(f"{name}: confidence out of range 0-1")
    return (not errors), errors


# ---------------------------------------------------------------------------
# CrewAI assembly
# ---------------------------------------------------------------------------


def build_understanding_agent(cfg: Dict[str, Any]) -> Agent:
    a_cfg = cfg["agents"]["understanding"]
    return Agent(
        role=a_cfg["role"],
        goal=a_cfg["goal"],
        backstory=a_cfg["backstory"],
        llm=build_llm(cfg, "understanding"),
        tools=UNDERSTANDING_TOOLS,
        verbose=False,
        memory=False,
        allow_delegation=False,
        cache=False,
        max_iter=12,
    )


def build_understanding_tasks(agent: Agent, run_dir: str | Path,
                              cfg: Dict[str, Any]) -> List[Task]:
    run_dir = Path(run_dir)
    profile = _load_profile(run_dir)
    context = _load_context(run_dir)
    profile_json = profile.model_dump_json()

    classify_column_roles = Task(
        description=(
            f"Classify the role of every column of run '{run_dir}' (§2.2).\n"
            "Step 1: call column_profiler_tool with profile_json="
            f"'{profile_json}'. It returns per-column facts + the "
            "rule-based suggested_role.\n"
            "Step 2: review the suggestions. You MAY reclassify a column "
            "ONLY into its alternate_roles or into 'identifier' by "
            "name/semantics (e.g. a numeric zip_code is an identifier). "
            "Never invent columns; never change dtype/nunique/nullable.\n"
            "Return ONLY a JSON list, one object per column:\n"
            '[{"name": "<col>", "role": "<identifier|temporal|measure|'
            'categorical|dimension|free_text>"}]\n'
            "Python is authoritative — the rules win when you do not "
            "reclassify.\n"
            + data_note()
        ),
        expected_output=(
            "The strict JSON list of {name, role} described above. No prose."
        ),
        agent=agent,
    )

    detect_domain_and_entities = Task(
        description=(
            f"Infer the domain and business entities for run '{run_dir}' "
            "(§2.2).\n"
            "Step 1: call domain_classifier_tool with profile_json="
            f"'{profile_json}'. It returns column facts + the redacted "
            "20-row sample + a 'domain_decision' skeleton.\n"
            f"Business context: goal_summary='{context.goal_summary}', "
            f"answers={json.dumps(context.answers, ensure_ascii=False)}, "
            f"generic_mode={context.generic_mode}.\n"
            "Step 2: fill the domain_decision skeleton: detected_domain "
            "(short name like 'sales', 'finance', 'hr', ...), "
            "domain_confidence (0.0-1.0, 0.0 when the context is generic), "
             "entities (business entity names like Product, Customer, "
             "Order). Use the sample only as a redacted hint.\n"
             "Return ONLY the filled domain_decision JSON object.\n"
             + data_note()
        ),
        expected_output=(
            '{"detected_domain": "...", "domain_confidence": 0.0, '
            '"entities": [...]}'
        ),
        agent=agent,
    )

    build_analysis_plan_task = Task(
        description=(
            f"Build the DSL analysis plan for run '{run_dir}' (§2.2).\n"
            "Propose candidate KPIs as DSL operations ONLY (function from "
            "the whitelist: sum, mean, median, count, nunique, min, max, "
            "std, growth, correlation, ratio; ratio has nested numerator/"
            "denominator operations) plus statistical_tests from "
            "descriptive/correlation/trend/anova.\n"
            "Then call dsl_plan_builder_tool with your raw plan JSON — it "
            "validates every operation against the whitelist and returns "
            '{"plan": ..., "errors": [...]}. If errors is non-empty, fix '
            "the plan and call it again until errors is empty.\n"
            "Return ONLY the final validated plan JSON as returned by the "
            "tool.\n"
            + data_note()
        ),
        expected_output=(
            'The validated plan JSON: {"candidate_kpis": [{"kpi_id", "name",'
            ' "operation"}], "statistical_tests": [...], '
            '"has_temporal_data": bool, "limitations": []}'
        ),
        agent=agent,
    )

    return [classify_column_roles, detect_domain_and_entities,
            build_analysis_plan_task]


# ---------------------------------------------------------------------------
# Task A.2 — LLM fallback for the ambiguous identifier band
# ---------------------------------------------------------------------------


def _identifier_threshold(cfg: Dict[str, Any]) -> float:
    return float(cfg.get("understanding", {})
                 .get("identifier_confidence_threshold", 0.7))


def _resolve_ambiguous_identifiers(
        facts, profile: DataProfile, cfg: Dict[str, Any],
        use_crew: bool, log: RunLogger
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """LLM fallback (task A.2): reclassify numeric columns in the ambiguous
    band (0.3 <= heuristic score < threshold) that a naive dtype check would
    misclassify as measures. Crew mode + opt-in config only; goes through the
    same complete_json guardrails (retries, schema/validator, cost ledger).
    Returns (overrides {name: role}, reasons {name: "reason (confidence X)"})."""
    if not use_crew:
        return {}, {}
    u_cfg = cfg.get("understanding", {}) or {}
    if not bool(u_cfg.get("identifier_llm_fallback", True)):
        return {}, {}
    threshold = _identifier_threshold(cfg)
    ambiguous = [
        f for f in facts
        if _is_numeric(f.dtype) and f.suggested_role == "measure"
        and 0.3 <= f.identifier_score < threshold
    ]
    if not ambiguous:
        return {}, {}

    sample_by_col: Dict[str, List[Any]] = {}
    for record in profile.sample or []:
        for f in ambiguous:
            values = sample_by_col.setdefault(f.name, [])
            if len(values) < 10 and f.name in record:
                values.append(record[f.name])

    system = (
        "You decide whether a numeric column is an ID-like value (phone "
        "number, postal code, national id, account number...) or a real "
        "measure. You see only the column name plus up to 10 redacted "
        "sample values — never the raw dataset. For each column return "
        'one object {"<column>": {"role": "identifier"|"measure"|'
        '"categorical", "confidence": 0.0-1.0, "reason": "<short reason>"}}. '
        "Return ONLY one JSON object."
    )
    payload_json = json.dumps(
        {f.name: {"sample": sample_by_col.get(f.name, [])}
         for f in ambiguous}, ensure_ascii=False, indent=2)
    user = (
        "Columns whose numeric role is ambiguous. Decide each one:\n"
        f"{payload_json}\n"
        "Examples: a phone_number with constant digit length -> identifier; "
        "a revenue column with decimals -> measure; an ID-like code -> "
        "identifier.\n"
        + data_note()
    )
    payload, warnings = complete_json(
        cfg, "understanding", system, user,
        validator=_validate_identifier_verdicts)
    for warning in warnings:
        log.fallback(STAGE, warning)
    if not isinstance(payload, dict):
        return {}, {}

    overrides: Dict[str, str] = {}
    reasons: Dict[str, str] = {}
    for name, raw in payload.items():
        if not isinstance(raw, dict):
            continue
        try:
            verdict = IdentifierVerdict(**raw)
        except Exception:  # noqa: BLE001 -- skip malformed entries
            continue
        if verdict.role != "measure":
            overrides[name] = verdict.role
        if verdict.role == "identifier":
            reasons[name] = (f"{verdict.reason} "
                             f"(confidence {verdict.confidence:.2f})")
    if overrides:
        log.info(STAGE, "identifier fallback applied",
                 columns=list(overrides), reasons=reasons)
    return overrides, reasons


def _apply_identifier_reasons(understanding, reasons: Dict[str, str]) -> None:
    """Record LLM identifier-fallback reasons in dataset_understanding.json
    so every identifier decision is auditable (task A.2)."""
    if not reasons:
        return
    for column in understanding.columns:
        if column.role == "identifier" and column.name in reasons:
            column.override_source = "llm"
            column.override_reason = (
                f"identifier fallback: {reasons[column.name]}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_understanding(run_dir: str | Path,
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
        log.error(STAGE, f"understanding failed: {exc}")
        status = "failed"
        summary = {"run_id": run_id, "stage": STAGE, "status": status,
                   "error": str(exc)}
    log.stage_end(STAGE, status, time.monotonic() - start)
    summary["log_path"] = str(run_dir / "logs" / "run.jsonl")
    return summary


def _load_extracted_df(run_dir: Path) -> Optional[pd.DataFrame]:
    """Best-effort raw extracted CSV for the value-shape signals of the
    identifier heuristic (A.1). None -> name-only detection."""
    extracted = run_dir / "data" / "extracted"
    if not extracted.is_dir():
        return None
    files = sorted(extracted.glob("*.csv"))
    if not files:
        return None
    try:
        return pd.read_csv(files[0], encoding="utf-8-sig")
    except Exception:  # noqa: BLE001 -- heuristic is best-effort
        return None


def _run_deterministic(run_dir: Path, cfg: Dict[str, Any],
                       log: RunLogger) -> Dict[str, Any]:
    profile = _load_profile(run_dir)
    context = _load_context(run_dir)

    t0 = time.monotonic()
    df = _load_extracted_df(run_dir)
    facts = ColumnProfiler().profile_columns(
        profile, df=df, identifier_threshold=_identifier_threshold(cfg))
    log.tool_call(STAGE, "column_profiler_tool", "passed",
                  time.monotonic() - t0)

    t0 = time.monotonic()
    domain, confidence = detect_domain_heuristic(context)
    log.tool_call(STAGE, "domain_classifier_tool", "passed",
                  time.monotonic() - t0)

    t0 = time.monotonic()
    plan = default_plan(profile, df=df)
    log.tool_call(STAGE, "dsl_plan_builder_tool", "passed",
                  time.monotonic() - t0)

    # A.2: LLM fallback only in crew mode; deterministic path never calls it.
    overrides, reasons = _resolve_ambiguous_identifiers(
        facts, profile, cfg, use_crew=False, log=log)
    understanding = assemble_understanding(
        profile=profile, facts=facts, role_overrides=overrides,
        domain=(domain, confidence, []), context=context, limitations=[])
    _apply_identifier_reasons(understanding, reasons)
    _save_artifacts(run_dir, understanding, plan)

    return {
        "stage": STAGE, "status": "passed",
        "detected_domain": domain, "domain_confidence": confidence,
        "kpi_count": len(plan.candidate_kpis),
        "dataset_understanding_path": str(run_dir / "metadata"
                                          / "dataset_understanding.json"),
        "analysis_plan_path": str(run_dir / "metadata" / "analysis_plan.json"),
        "errors": [],
    }


def _run_crew(run_dir: Path, cfg: Dict[str, Any],
              log: RunLogger) -> Dict[str, Any]:
    agent = build_understanding_agent(cfg)
    tasks = build_understanding_tasks(agent, str(run_dir), cfg)
    crew = Crew(agents=[agent], tasks=tasks, process=Process.sequential,
                verbose=False, cache=False)
    t0 = time.monotonic()
    result = crew.kickoff(inputs={})
    log.info(STAGE, "crew kickoff finished",
             duration_s=round(time.monotonic() - t0, 3))

    understanding, plan, warnings = _finalize_understanding(
        run_dir, result, cfg=cfg, log=log)
    for warning in warnings:
        log.fallback(STAGE, warning)

    return {
        "stage": STAGE, "status": "passed",
        "detected_domain": understanding.detected_domain,
        "domain_confidence": understanding.domain_confidence,
        "kpi_count": len(plan.candidate_kpis),
        "dataset_understanding_path": str(run_dir / "metadata"
                                          / "dataset_understanding.json"),
        "analysis_plan_path": str(run_dir / "metadata" / "analysis_plan.json"),
        "errors": [],
    }


def _finalize_understanding(run_dir: Path, result,
                            cfg: Optional[Dict[str, Any]] = None,
                            log: Optional[RunLogger] = None) -> tuple[
        Any, AnalysisPlan, List[str]]:
    """Python-authoritative finalize: validate/rebuild every artifact."""
    cfg = cfg or load_config(require_key=False)
    log = log or RunLogger(run_dir, run_dir.name)
    """Python-authoritative finalize: validate/rebuild every artifact."""
    profile = _load_profile(run_dir)
    context = _load_context(run_dir)
    df = _load_extracted_df(run_dir)
    facts = ColumnProfiler().profile_columns(
        profile, df=df, identifier_threshold=_identifier_threshold(cfg))
    warnings: List[str] = []

    outputs = getattr(result, "tasks_output", None) or []
    raws = [str(getattr(t, "raw", "") or getattr(t, "output", "") or "")
            for t in outputs]

    role_overrides = _parse_roles_json(raws[0] if len(raws) > 0 else "")
    domain = _parse_domain_json(raws[1] if len(raws) > 1 else "")
    raw_plan = _parse_json(raws[2] if len(raws) > 2 else "")

    plan, plan_errors = build_analysis_plan(raw_plan)
    if raw_plan is None or not raw_plan.get("candidate_kpis"):
        plan = default_plan(profile, df=df)
        warnings.append("llm_plan_failed_default_used")
    elif plan_errors:
        warnings.append("llm_plan_partially_rejected")

    if domain is None:
        domain = detect_domain_heuristic(context)
        domain = (domain[0], domain[1], [])
        warnings.append("llm_domain_failed_heuristic_used")

    # A.2: LLM fallback for the ambiguous identifier band (crew mode only).
    overrides, reasons = _resolve_ambiguous_identifiers(
        facts, profile, cfg, use_crew=True, log=log)
    role_overrides = {**role_overrides, **overrides}

    understanding = assemble_understanding(
        profile=profile, facts=facts, role_overrides=role_overrides,
        domain=domain, context=context, limitations=plan_errors)
    _apply_identifier_reasons(understanding, reasons)
    _save_artifacts(run_dir, understanding, plan)
    return understanding, plan, warnings


# ---------------------------------------------------------------------------
# Artifact IO
# ---------------------------------------------------------------------------


def _load_profile(run_dir: str | Path) -> DataProfile:
    run_dir = Path(run_dir)
    path = run_dir / "metadata" / "data_profile.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run Stage 1 (ingestion) on this run dir first")
    return DataProfile(**json.loads(path.read_text(encoding="utf-8")))


def _load_context(run_dir: str | Path) -> BusinessContext:
    run_dir = Path(run_dir)
    path = run_dir / "knowledge" / "business_context.json"
    if not path.exists():
        return BusinessContext(file_name="", generic_mode=True)
    return BusinessContext(**json.loads(path.read_text(encoding="utf-8")))


def _save_artifacts(run_dir: Path, understanding, plan) -> None:
    metadata = run_dir / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "dataset_understanding.json").write_text(
        json.dumps(understanding.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8")
    (metadata / "analysis_plan.json").write_text(
        json.dumps(plan.model_dump(exclude_none=True), ensure_ascii=False,
                   indent=2),
        encoding="utf-8")


def _parse_roles_json(raw: str) -> Dict[str, str]:
    data = _parse_json(raw)
    if not isinstance(data, list):
        return {}
    return {str(item.get("name")): str(item.get("role"))
            for item in data if isinstance(item, dict)
            and item.get("name") and item.get("role")}


def _parse_domain_json(raw: str) -> tuple[str, float, List[str]] | None:
    data = _parse_json(raw)
    if not isinstance(data, dict) or not data.get("detected_domain"):
        return None
    confidence = data.get("domain_confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    if not 0.0 <= confidence <= 1.0:
        confidence = 0.0
    entities = data.get("entities")
    if not isinstance(entities, list):
        entities = []
    entities = [str(e) for e in entities]
    return str(data["detected_domain"]), confidence, entities


def _parse_json(raw: str) -> Any:
    start = raw.find("{") if "{" in raw else -1
    end = raw.rfind("}")
    try:
        if start == -1 or end <= start:
            return json.loads(raw.strip()) if raw.strip() else None
        return json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="understanding_agent",
        description="Run Insight Forge stage 2 (data understanding) on a "
                    "Stage-1 run dir.")
    parser.add_argument("run_dir", help="existing run dir (from Stage 1)")
    parser.add_argument("--crew", action="store_true",
                        help="run via the real CrewAI agent (requires API key)")
    args = parser.parse_args(argv)

    summary = run_understanding(args.run_dir, use_crew=args.crew)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
