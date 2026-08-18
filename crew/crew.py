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
from typing import Any, Callable, Dict

from crew.flows import (
    build_verdict,
    check_caps,
    check_cleaning_retry,
    check_dq_gate,
    write_verdict_file,
)
from shared.logger import RunLogger
from shared.utils import allocate_run_id, init_run_layout, load_config

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def _write_run_comparison(run_dir: Path, source_name: str,
                          current_run_id: str, log: RunLogger) -> None:
    """"vs previous run" callout: compare current KPIs against the most
    recent cached run of the same source file (SQLite index, §8)."""
    from shared.cache import find_previous
    from shared.schemas import SCHEMA_VERSION

    previous_id = find_previous(source_name, exclude_run_id=current_run_id)
    if previous_id is None:
        return
    prev_dir = RUNS_DIR / previous_id
    prev_kpis = prev_dir / "outputs" / "kpis.json"
    curr_kpis = run_dir / "outputs" / "kpis.json"
    if not (prev_kpis.is_file() and curr_kpis.is_file()):
        return
    try:
        prev_payload = json.loads(prev_kpis.read_text(encoding="utf-8"))
        curr_payload = json.loads(curr_kpis.read_text(encoding="utf-8"))
        # Audit L: only compare runs with a matching artifact schema version.
        if (prev_payload.get("schema_version")
                != curr_payload.get("schema_version")
                != SCHEMA_VERSION):
            log.warning("pipeline", "run comparison skipped: schema_version "
                        "mismatch", previous_run_id=previous_id)
            return
        prev_list = prev_payload.get("kpis", [])
        curr_list = curr_payload.get("kpis", [])
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(prev_list, list) or not isinstance(curr_list, list):
        return

    prev_by_id = {k.get("kpi_id"): k for k in prev_list
                  if isinstance(k, dict)}
    rows = []
    for kpi in curr_list:
        if not isinstance(kpi, dict):
            continue
        prev = prev_by_id.get(kpi.get("kpi_id"))
        if prev is None:
            continue
        rows.append({
            "kpi_id": kpi.get("kpi_id"),
            "name": kpi.get("name"),
            "current": kpi.get("value"),
            "previous": prev.get("value"),
            "unit": kpi.get("unit"),
        })
    if not rows:
        return
    out = {
        "previous_run_id": previous_id,
        "compared_kpis": rows,
        "note": "Automatically compared against the most recent cached run "
                "of the same source file.",
    }
    (run_dir / "outputs" / "run_comparison.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("pipeline", "run comparison written",
             previous_run_id=previous_id, kpis=len(rows))


def _human_inputs_summary(run_dir: Path) -> List[Dict[str, Any]]:
    """Human-input audit trail: who answered what, from business_context.json."""
    ctx_path = run_dir / "knowledge" / "business_context.json"
    if not ctx_path.exists():
        return []
    try:
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [{
        "answered_by": ctx.get("answered_by") or None,
        "generic_mode": ctx.get("generic_mode", False),
        "answer_log": ctx.get("answer_log", []),
    }]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    file_path: str | Path,
    use_crew: bool = False,
    locale: str = "en",
    cfg: Dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    answer_provider: Callable[[str], str] | None = None,
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
    output_dir : str or Path, optional
        Explicit run directory instead of allocating one under ``runs/``.
        Must be empty or non-existent; the run layout is created inside it.
    answer_provider : callable, optional
        Replaces the interactive console questions with a programmatic
        provider (return ``""`` to fall back to Generic Analysis Mode).
        Web deployments must pass this — the pipeline never prompts on
        a background worker's stdin.

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

    # §8 result cache — only default-dir, deterministic, APPROVED runs
    cache_on = (
        output_dir is None and not use_crew
        and bool(cfg.get("cache", {}).get("enabled", False))
    )
    if cache_on:
        from shared.cache import cache_key, get_cached, input_hash
        key = cache_key(input_hash(file_path), cfg)
        hit = get_cached(key)
        if hit is not None:
            cached_dir = RUNS_DIR / hit["run_id"]
            cached_manifest = cached_dir / "master_manifest.json"
            if cached_manifest.is_file():
                try:
                    cached = json.loads(
                        cached_manifest.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    cached = None
                if cached and cached.get("verdict") in (
                        "APPROVED", "APPROVED_WITH_WARNINGS"):
                    # Audit L: cached artifact must match the current
                    # schema version, else the hit is invalidated.
                    from shared.schemas import SCHEMA_VERSION
                    profile_path = cached_dir / "metadata" \
                        / "data_profile.json"
                    if profile_path.is_file():
                        try:
                            profile = json.loads(
                                profile_path.read_text(encoding="utf-8"))
                        except (json.JSONDecodeError, OSError):
                            profile = {}
                        if profile.get("schema_version") != SCHEMA_VERSION:
                            hit = None
                    if hit is not None:
                        log = RunLogger(cached_dir, hit["run_id"])
                        log.info("pipeline",
                                 "cache hit — reusing previous run",
                                 reason="cache_hit")
                        return {
                            "run_id": hit["run_id"],
                            "run_dir": str(cached_dir),
                            "verdict": "APPROVED",
                            "score": cached.get("score"),
                            "report_path": cached.get("report_path"),
                            "stage_results": cached.get("stage_results", {}),
                            "duration_s": 0.0,
                            "cached": True,
                            "cache_key": key[:12],
                        }

    # Allocate run directory
    if output_dir is not None:
        run_dir = Path(output_dir)
        run_id = run_dir.name
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
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

    import threading as _threading

    stage_timeout = float(cfg.get("limits", {}).get("stage_timeout_seconds",
                                                    0) or 0)

    def _run_stage(stage: str, fn: Any) -> Any:
        """Run one stage; enforce limits.stage_timeout_seconds in-flight."""
        if stage_timeout <= 0:
            return fn()
        box: Dict[str, Any] = {}

        def worker() -> None:
            try:
                box["v"] = fn()
            except BaseException as exc:  # noqa: BLE001 -- re-raised in main
                box["e"] = exc

        thread = _threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(stage_timeout)
        if thread.is_alive():
            raise TimeoutError(
                f"stage {stage} exceeded stage_timeout_seconds="
                f"{stage_timeout:.0f}")
        if "e" in box:
            raise box["e"]
        return box.get("v")

    # ------------------------------------------------------------------
    # Stage 1 — Ingestion
    # ------------------------------------------------------------------
    if cap := _check_caps("ingestion"):
        return cap

    from agents.ingestion_agent import run_ingestion

    log.stage_start("ingestion")
    t_stage = time.time()
    try:
        s1 = _run_stage("ingestion", lambda: run_ingestion(
            str(file_path),
            run_dir=str(run_dir),
            cfg=cfg,
            logger=log,
            use_crew=use_crew,
            answer_provider=answer_provider,
        ))
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
        s2 = _run_stage("understanding", lambda: run_understanding(
            str(run_dir), cfg=cfg, logger=log, use_crew=use_crew,
        ))
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
        s3 = _run_stage("data_quality",
                        lambda: run_data_quality(str(run_dir), cfg=cfg,
                                                 logger=log))
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
        s4 = _run_stage("cleaning", lambda: run_cleaning(
            str(run_dir), cfg=cfg, logger=log, use_crew=use_crew))
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

    # Cleaning recheck loop — Audit N: a failed recheck RE-RUNS the whole
    # cleaning stage (incremental retry), not just the DQ recheck.
    max_rechecks = cfg.get("limits", {}).get("cleaning_max_rechecks", 3)
    recheck_passed = False

    for attempt in range(1, max_rechecks + 1):
        if cap := _check_caps("data_quality_recheck"):
            return cap

        if attempt > 1:
            # Re-run cleaning with fresh effort before rechecking.
            log.stage_start("cleaning_retry")
            t_stage = time.time()
            try:
                s4 = _run_stage("cleaning_retry", lambda: run_cleaning(
                    str(run_dir), cfg=cfg, logger=log, use_crew=use_crew))
            except Exception as exc:
                log.stage_end("cleaning_retry", "failed",
                              time.time() - t_stage)
                return _abort(str(exc), stage="cleaning_retry")
            log.stage_end("cleaning_retry", s4.get("status", "failed"),
                          time.time() - t_stage)
            stage_results["cleaning"] = s4
            if s4.get("status") != "passed":
                return _abort(
                    f"stage=cleaning_retry status={s4.get('status')} "
                    f"(attempt {attempt})",
                    stage="cleaning_retry",
                    error=s4.get("error", ""),
                )

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
        s5 = _run_stage("analysis", lambda: run_analysis(
            str(run_dir), cfg=cfg, logger=log, use_crew=use_crew))
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
        s6 = _run_stage("insights", lambda: run_insights(
            str(run_dir), cfg=cfg, logger=log, use_crew=use_crew))
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
        s7 = _run_stage("report", lambda: run_report(
            str(run_dir), cfg=cfg, logger=log, use_crew=use_crew, locale=locale,
        ))
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

    # §8: write the vs-previous-run baseline BEFORE rendering is final, so
    # the callout lands inside this run's own report.
    if cache_on:
        from shared.cache import find_previous
        previous_id = find_previous(file_path.name, exclude_run_id=run_id)
        if previous_id is not None:
            _write_run_comparison(run_dir, file_path.name, run_id, log)
            if report_path is not None and Path(report_path).is_file():
                from agents.report_agent import rerender_report
                rerender_report(Path(report_path), cfg=cfg, locale=locale)

    # ------------------------------------------------------------------
    # Stage 8 — QA
    # ------------------------------------------------------------------
    if cap := _check_caps("qa"):
        return cap

    from agents.qa_agent import run_qa

    log.stage_start("qa")
    t_stage = time.time()
    try:
        s8 = _run_stage("qa", lambda: run_qa(
            str(run_dir), cfg=cfg, logger=log, use_crew=use_crew))
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
        "human_inputs": _human_inputs_summary(run_dir),
        "stage_results": {
            k: {"status": v.get("status")} for k, v in stage_results.items()
        },
        "report_path": report_path,
        "duration_s": time.time() - t0,
    }
    manifest_path = run_dir / "master_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    # §8: store successful deterministic runs + build vs-previous comparison
    if cache_on:
        from shared.cache import cache_key, find_previous, input_hash, store
        if final_verdict in ("APPROVED", "APPROVED_WITH_WARNINGS"):
            source_name = file_path.name
            store(key, run_id, source_name)
            _write_run_comparison(run_dir, source_name, run_id, log)

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
