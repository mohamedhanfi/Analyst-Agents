"""Insight Forge — pipeline orchestrator.

Wires all 8 stages into a single sequential pipeline with:
- DQ gate (passed / needs_repair → cleaning)
- Cleaning recheck loop (max N retries → auto-verdict)
- Hard-cap checks (cost / runtime) before each stage
- QA verdict branching as the final gate
- master_manifest.json at the end

Usage::

    from crew.crew import run_pipeline
    result = run_pipeline("tests/fixtures/sales_demo.csv")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from crew.flows import (
    build_verdict,
    check_caps,
    check_cleaning_retry,
    check_dq_gate,
    write_verdict_file,
)
from shared.logger import RunLogger
from shared.utils import allocate_run_id, init_run_layout, load_config


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    file_path: str | Path,
    use_crew: bool = False,
    locale: str = "en",
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the full 8-stage Insight Forge pipeline.

    Parameters
    ----------
    file_path : str or Path
        Path to the raw CSV / XLSX file.
    use_crew : bool
        If True, LLM-using agents run through CrewAI; if False, deterministic
        fallback only (no API key required).
    locale : str
        Report locale (e.g. ``'en'`` or ``'ar'``).
    cfg : dict, optional
        Pre-loaded config dict. If None, loaded from ``config.yaml``.

    Returns
    -------
    dict
        Pipeline result with keys: ``run_id``, ``run_dir``, ``verdict``,
        ``score``, ``report_path``, ``stage_results``, ``duration_s``.
    """
    t0 = time.time()
    file_path = Path(file_path)

    if cfg is None:
        cfg = load_config(require_key=use_crew)

    # Allocate run directory
    run_id, run_dir = allocate_run_id()
    run_dir = init_run_layout(run_dir)
    log = RunLogger(run_dir, run_id)
    log.stage_start("pipeline")

    stage_results: Dict[str, Dict[str, Any]] = {}
    reason_codes: list[str] = []

    def _abort(reason: str, stage: str = "", error: str = "") -> Dict[str, Any]:
        """Abort the pipeline, write a NEEDS_REVISION verdict, return result."""
        if reason:
            reason_codes.append(reason)
            log.fallback("pipeline", reason)
        if stage:
            log.error(stage, error or reason)
        log.stage_end("pipeline", "failed", time.time() - t0)

        qa_payload = build_verdict(
            {"verdict": "NEEDS_REVISION", "score": 0.0},
            reason_codes=reason_codes,
        )
        write_verdict_file(run_dir, qa_payload)

        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "verdict": "NEEDS_REVISION",
            "score": 0.0,
            "report_path": None,
            "stage_results": stage_results,
            "duration_s": time.time() - t0,
        }

    def _check_caps(stage: str) -> Dict[str, Any] | None:
        """Check hard caps; abort if tripped. Returns abort result dict or None."""
        cap = check_caps(log, t0, cfg)
        if cap:
            return _abort(cap, stage=stage)
        return None

    # ------------------------------------------------------------------
    # Stage 1 — Ingestion
    # ------------------------------------------------------------------
    if cap := _check_caps("ingestion"):
        return cap

    from agents.ingestion_agent import run_ingestion

    log.stage_start("ingestion")
    t_stage = time.time()
    try:
        s1 = run_ingestion(
            str(file_path),
            run_dir=str(run_dir),
            cfg=cfg,
            logger=log,
            use_crew=use_crew,
        )
    except Exception as exc:
        log.stage_end("ingestion", "failed", time.time() - t_stage)
        return _abort(str(exc), stage="ingestion")
    log.stage_end("ingestion", s1.get("status", "failed"), time.time() - t_stage)
    stage_results["ingestion"] = s1

    if s1.get("status") != "passed":
        return _abort(
            f"stage=ingestion status={s1.get('status')}",
            stage="ingestion",
            error=s1.get("error", ""),
        )

    # ------------------------------------------------------------------
    # Stage 2 — Understanding
    # ------------------------------------------------------------------
    if cap := _check_caps("understanding"):
        return cap

    from agents.understanding_agent import run_understanding

    log.stage_start("understanding")
    t_stage = time.time()
    try:
        s2 = run_understanding(
            str(run_dir), cfg=cfg, logger=log, use_crew=use_crew,
        )
    except Exception as exc:
        log.stage_end("understanding", "failed", time.time() - t_stage)
        return _abort(str(exc), stage="understanding")
    log.stage_end("understanding", s2.get("status", "failed"), time.time() - t_stage)
    stage_results["understanding"] = s2

    if s2.get("status") != "passed":
        return _abort(
            f"stage=understanding status={s2.get('status')}",
            stage="understanding",
            error=s2.get("error", ""),
        )

    # ------------------------------------------------------------------
    # Stage 3 — Data Quality (deterministic, no LLM)
    # ------------------------------------------------------------------
    if cap := _check_caps("data_quality"):
        return cap

    from agents.data_quality import run_data_quality

    log.stage_start("data_quality")
    t_stage = time.time()
    try:
        s3 = run_data_quality(str(run_dir), cfg=cfg, logger=log)
    except Exception as exc:
        log.stage_end("data_quality", "failed", time.time() - t_stage)
        return _abort(str(exc), stage="data_quality")
    log.stage_end("data_quality", s3.get("status", "failed"), time.time() - t_stage)
    stage_results["data_quality"] = s3

    dq_gate = check_dq_gate(s3)
    # s3 already applies deterministic repair internally when needed,
    # so dq_gate tells us whether to proceed to cleaning or not.
    # Either way, cleaning is the next step after DQ.

    # ------------------------------------------------------------------
    # Stage 4 — Cleaning + recheck loop
    # ------------------------------------------------------------------
    if cap := _check_caps("cleaning"):
        return cap

    from agents.cleaning_agent import run_cleaning

    log.stage_start("cleaning")
    t_stage = time.time()
    try:
        s4 = run_cleaning(str(run_dir), cfg=cfg, logger=log, use_crew=use_crew)
    except Exception as exc:
        log.stage_end("cleaning", "failed", time.time() - t_stage)
        return _abort(str(exc), stage="cleaning")
    log.stage_end("cleaning", s4.get("status", "failed"), time.time() - t_stage)
    stage_results["cleaning"] = s4

    if s4.get("status") != "passed":
        return _abort(
            f"stage=cleaning status={s4.get('status')}",
            stage="cleaning",
            error=s4.get("error", ""),
        )

    # Cleaning recheck loop
    max_rechecks = cfg.get("limits", {}).get("cleaning_max_rechecks", 3)
    recheck_passed = False

    for attempt in range(1, max_rechecks + 1):
        if cap := _check_caps("data_quality_recheck"):
            return cap

        t_recheck = time.time()
        try:
            s3_re = run_data_quality(
                str(run_dir), cfg=cfg, logger=log, data_source="cleaned",
            )
        except Exception as exc:
            log.stage_end("data_quality_recheck", "failed", time.time() - t_recheck)
            return _abort(str(exc), stage="data_quality_recheck")

        log.stage_end(
            "data_quality_recheck",
            s3_re.get("status", "failed"),
            time.time() - t_recheck,
        )
        stage_results[f"data_quality_recheck_{attempt}"] = s3_re

        if s3_re.get("status") == "passed":
            recheck_passed = True
            break
        # Recheck failed — continue if retries remain
        if not check_cleaning_retry(s3_re, attempt, max_rechecks):
            break

    if not recheck_passed:
        reason = "cleaning_retry_limit_exceeded"
        log.fallback("pipeline", reason)
        log.stage_end("pipeline", "failed", time.time() - t0)
        qa_payload = build_verdict(
            {"verdict": "NEEDS_REVISION", "score": 0.0},
            reason_codes=[reason],
        )
        write_verdict_file(run_dir, qa_payload)
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "verdict": "NEEDS_REVISION",
            "score": 0.0,
            "report_path": None,
            "stage_results": stage_results,
            "duration_s": time.time() - t0,
        }

    # ------------------------------------------------------------------
    # Stage 5 — Analysis
    # ------------------------------------------------------------------
    if cap := _check_caps("analysis"):
        return cap

    from agents.analysis import run_analysis

    log.stage_start("analysis")
    t_stage = time.time()
    try:
        s5 = run_analysis(str(run_dir), cfg=cfg, logger=log, use_crew=use_crew)
    except Exception as exc:
        log.stage_end("analysis", "failed", time.time() - t_stage)
        return _abort(str(exc), stage="analysis")
    log.stage_end("analysis", s5.get("status", "failed"), time.time() - t_stage)
    stage_results["analysis"] = s5

    if s5.get("status") != "passed":
        return _abort(
            f"stage=analysis status={s5.get('status')}",
            stage="analysis",
            error=s5.get("error", ""),
        )

    # ------------------------------------------------------------------
    # Stage 6 — Insights
    # ------------------------------------------------------------------
    if cap := _check_caps("insights"):
        return cap

    from agents.insight_agent import run_insights

    log.stage_start("insights")
    t_stage = time.time()
    try:
        s6 = run_insights(str(run_dir), cfg=cfg, logger=log, use_crew=use_crew)
    except Exception as exc:
        log.stage_end("insights", "failed", time.time() - t_stage)
        return _abort(str(exc), stage="insights")
    log.stage_end("insights", s6.get("status", "failed"), time.time() - t_stage)
    stage_results["insights"] = s6

    if s6.get("status") != "passed":
        return _abort(
            f"stage=insights status={s6.get('status')}",
            stage="insights",
            error=s6.get("error", ""),
        )

    # ------------------------------------------------------------------
    # Stage 7 — Report
    # ------------------------------------------------------------------
    if cap := _check_caps("report"):
        return cap

    from agents.report_agent import run_report

    log.stage_start("report")
    t_stage = time.time()
    try:
        s7 = run_report(
            str(run_dir), cfg=cfg, logger=log, use_crew=use_crew, locale=locale,
        )
    except Exception as exc:
        log.stage_end("report", "failed", time.time() - t_stage)
        return _abort(str(exc), stage="report")
    log.stage_end("report", s7.get("status", "failed"), time.time() - t_stage)
    stage_results["report"] = s7

    if s7.get("status") != "passed":
        return _abort(
            f"stage=report status={s7.get('status')}",
            stage="report",
            error=s7.get("error", ""),
        )

    report_path = s7.get("report_path")

    # ------------------------------------------------------------------
    # Stage 8 — QA
    # ------------------------------------------------------------------
    if cap := _check_caps("qa"):
        return cap

    from agents.qa_agent import run_qa

    log.stage_start("qa")
    t_stage = time.time()
    try:
        s8 = run_qa(str(run_dir), cfg=cfg, logger=log, use_crew=use_crew)
    except Exception as exc:
        log.stage_end("qa", "failed", time.time() - t_stage)
        return _abort(str(exc), stage="qa")
    log.stage_end("qa", s8.get("status", "failed"), time.time() - t_stage)
    stage_results["qa"] = s8

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    verdict_payload = build_verdict(s8, reason_codes=reason_codes)
    write_verdict_file(run_dir, verdict_payload)

    final_verdict = verdict_payload["verdict"]
    final_score = verdict_payload["score"]

    # ------------------------------------------------------------------
    # Master manifest
    # ------------------------------------------------------------------
    manifest = {
        "run_id": run_id,
        "pipeline_version": cfg.get("pipeline_version", "4.3.0"),
        "file": str(file_path),
        "use_crew": use_crew,
        "locale": locale,
        "verdict": final_verdict,
        "score": final_score,
        "reason_codes": reason_codes,
        "stage_results": {
            k: {"status": v.get("status")} for k, v in stage_results.items()
        },
        "report_path": report_path,
        "duration_s": time.time() - t0,
    }
    manifest_path = run_dir / "master_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    log.stage_end("pipeline", "passed", time.time() - t0)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "verdict": final_verdict,
        "score": final_score,
        "report_path": report_path,
        "stage_results": stage_results,
        "duration_s": time.time() - t0,
    }
