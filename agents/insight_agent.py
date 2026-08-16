"""Stage 6 — Insights & Recommendations (§2.6): CrewAI agent + deterministic core.

Consumes the stage-5 artifacts (kpis.json, statistical_results.json,
evidence_registry.json, chart_metadata.json, business_context.json,
dataset_understanding.json). Python generates every claim through the §2.6
claim taxonomy (DESCRIPTIVE/COMPARATIVE/CORRELATIONAL gated; PREDICTIVE only
when a forecast ran; CAUSAL never), builds the hedged recommendation chain
(Observation -> Finding -> Implication -> recommendation), and validates
everything before saving (evidence refs exist · claim type matches evidence
kinds · recommendations reference surviving insights). Failures are dropped
and logged. The LLM (--crew) may only reword title/description — evidence and
numbers stay Python's call (Python-authoritative).

CLI: python -m agents.insight_agent <run_dir> [--crew]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from analysis.chart_renderer import _fmt
from shared.llm import build_llm
from shared.logger import RunLogger
from shared.schemas import Insight, KpiResult, Recommendation, StatisticalResult
from shared.tools.insights import validate_insights
from shared.utils import init_run_layout, load_config

STAGE = "insights"

_MAX_INSIGHTS = 20
_MAX_RECOMMENDATIONS = 8


# ---------------------------------------------------------------------------
# CrewAI assembly
# ---------------------------------------------------------------------------


def build_insight_agent(cfg: Dict[str, Any]) -> Any:
    from crewai import Agent

    a_cfg = cfg["agents"]["insight"]
    return Agent(
        role=a_cfg["role"],
        goal=a_cfg["goal"],
        backstory=a_cfg["backstory"],
        llm=build_llm(cfg, "insight"),
        verbose=False,
        memory=False,
        allow_delegation=False,
        cache=False,
        max_iter=8,
    )


def build_insight_task(agent: Any, run_dir: str | Path) -> Any:
    from crewai import Task

    insights = _load_insights_digest(Path(run_dir))
    return Task(
        description=(
            "Here is the deterministic, evidence-grounded insight set for "
            "this run (already claim-validated).\n\n"
            f"{json.dumps(insights, ensure_ascii=False, indent=2)}\n\n"
            "Rewrite ONLY the title and description wording of each insight "
            "and recommendation so they read like a business analyst wrote "
            "them. Hard rules:\n"
            "- keep every field except title/description byte-identical "
            "(insight_id, claim_type, confidence, evidence_ids, "
            "required_evidence, related_kpis, recommendation_id, insight_id)\n"
            "- never add, remove or reword numbers, dates or percentages "
            "already in the text\n"
            "- never invent new claims or new evidence\n"
            "Return one JSON object {\"insights\": [...], "
            "\"recommendations\": [...]} and nothing else."
        ),
        expected_output="JSON object with insights and recommendations arrays",
        agent=agent,
    )


def _load_insights_digest(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "outputs" / "insights.json"
    if not path.exists():
        return {"insights": [], "recommendations": [], "warnings": []}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Deterministic generation (§2.6 taxonomy)
# ---------------------------------------------------------------------------


def _generate_insights(run_dir: Path,
                       log: RunLogger) -> Tuple[List[Insight],
                                                List[Recommendation],
                                                List[str]]:
    kpis = _load_kpis(run_dir)
    stats = _load_stats(run_dir)
    insights: List[Insight] = []
    counter = 0

    def _next_id() -> str:
        nonlocal counter
        counter += 1
        return f"INS-{counter:03d}"

    # 1) DESCRIPTIVE — every computed KPI (correlation KPIs are covered by
    # the CORRELATIONAL gate below; None values never ground a claim).
    for kpi in kpis:
        if len(insights) >= _MAX_INSIGHTS:
            break
        if kpi.operation.function == "correlation" or kpi.value is None:
            continue
        evidence_ids = [kpi.evidence_id] if kpi.evidence_id else []
        insights.append(Insight(
            insight_id=_next_id(),
            claim_type="DESCRIPTIVE",
            title=kpi.name,
            description=f"The {kpi.name} is {_fmt(kpi.value)} for this "
                        f"dataset.",
            confidence=_descriptive_confidence(kpi),
            evidence_ids=evidence_ids,
            required_evidence=["aggregate"],
            related_kpis=[kpi.kpi_id],
        ))

    # 2) CORRELATIONAL — only with p-value + CI, and only for meaningful
    # strength; reports strength, never cause.
    for st in stats:
        if (st.category == "correlation" and st.p_value is not None
                and st.statistic is not None
                and abs(st.statistic) >= 0.3 and st.p_value < 0.05):
            if len(insights) >= _MAX_INSIGHTS:
                break
            a, b = _pair_names(st)
            strength = "strong" if abs(st.statistic) >= 0.7 else "moderate"
            ci = ""
            if st.ci_low is not None and st.ci_high is not None:
                ci = (f" (95% CI [{_fmt(st.ci_low)}, "
                      f"{_fmt(st.ci_high)}])")
            insights.append(Insight(
                insight_id=_next_id(),
                claim_type="CORRELATIONAL",
                title=f"Correlation between {a} and {b}",
                description=(f"There is a {strength} association between "
                             f"{a} and {b} (r = {st.statistic:.2f}, "
                             f"p = {st.p_value:.3f}{ci}). This describes "
                             f"association, not cause."),
                confidence="high" if abs(st.statistic) >= 0.7 else "medium",
                evidence_ids=[st.evidence_id] if st.evidence_id else [],
                required_evidence=["correlation"],
            ))

    # 3) COMPARATIVE — group tests only when significant (p < 0.05).
    seen_pairs: set = set()
    for st in stats:
        if st.category == "comparison" and st.p_value is not None \
                and st.p_value < 0.05:
            a, b = _pair_names(st)
            pair = (a, b)
            if pair in seen_pairs:      # chi2 + cramers_v = same association
                continue
            seen_pairs.add(pair)
            if len(insights) >= _MAX_INSIGHTS:
                break
            if st.test_name in ("chi2", "cramers_v"):
                text = (f"{a} and {b} are significantly associated "
                        f"(p = {st.p_value:.3f}).")
            else:
                text = (f"{a} differs significantly across groups of {b} "
                        f"(p = {st.p_value:.3f}).")
            insights.append(Insight(
                insight_id=_next_id(),
                claim_type="COMPARATIVE",
                title=f"{a} differs across {b}",
                description=text,
                confidence="high" if st.p_value < 0.001 else "medium",
                evidence_ids=[st.evidence_id] if st.evidence_id else [],
                required_evidence=["group_comparison"],
            ))
    # (dedupe keeps one test per variable pair)

    # 4) Growth trend — a labeled series claim, gated on enough points.
    for st in stats:
        if st.category == "trend":
            series = (st.extra or {}).get("series") or []
            values = [v for v in (p.get("value") for p in series)
                      if isinstance(v, (int, float))]
            if len(series) < 3 or len(values) < 3:
                continue
            if len(insights) >= _MAX_INSIGHTS:
                break
            avg = sum(values) / len(values)
            a, b = _pair_names(st)
            insights.append(Insight(
                insight_id=_next_id(),
                claim_type="DESCRIPTIVE",
                title=f"{a} trend over {len(series)} periods",
                description=(f"{a} averaged {_fmt(avg)}% growth per "
                             f"{_period_label(st)} period across the last "
                             f"{len(series)} periods. Labeled as a trend "
                             f"series, not a forecast."),
                confidence="medium" if len(series) >= 6 else "low",
                evidence_ids=[st.evidence_id] if st.evidence_id else [],
                required_evidence=["growth_rate"],
            ))

    recommendations = _build_recommendations(insights)
    valid, valid_recs, warnings = validate_insights(
        [i.model_dump() for i in insights],
        [r.model_dump() for r in recommendations], run_dir)
    for warning in warnings:
        log.fallback(STAGE, warning)
    return ([Insight(**v) for v in valid],
            [Recommendation(**r) for r in valid_recs],
            warnings)


def _descriptive_confidence(kpi: KpiResult) -> str:
    if kpi.operation.function in ("sum", "count", "min", "max"):
        return "high"
    return "medium"


def _pair_names(st: StatisticalResult) -> Tuple[str, str]:
    variables = st.variables or []
    a = variables[0] if variables else "the metric"
    b = variables[1] if len(variables) > 1 else "the group"
    return a, b


def _period_label(st: StatisticalResult) -> str:
    return (st.extra or {}).get("period") or "time"


def _build_recommendations(insights: List[Insight]) -> List[Recommendation]:
    """Observation -> Finding -> Implication -> hedged recommendation."""
    implications = {
        "DESCRIPTIVE": ("the current level is the baseline any change will "
                        "be measured against"),
        "COMPARATIVE": ("the gap between groups is where targeted actions "
                        "can move the headline figure"),
        "CORRELATIONAL": ("the association identifies a candidate lever, "
                          "but acting on it as a cause would be premature"),
    }
    hedges = {
        "DESCRIPTIVE": "consider testing a controlled improvement to this "
                       "metric on a small slice before scaling it",
        "COMPARATIVE": "consider investigating what drives the lagging "
                       "group and trialing a targeted action on it",
        "CORRELATIONAL": "consider testing the lever in a controlled "
                         "experiment before committing budget to it",
    }
    recommendations: List[Recommendation] = []
    for insight in insights[: _MAX_RECOMMENDATIONS]:
        recommendations.append(Recommendation(
            recommendation_id=f"REC-{len(recommendations) + 1:03d}",
            insight_id=insight.insight_id,
            title=f"Recommendation on: {insight.title}",
            description=(
                f"Observation — {insight.description}\n"
                f"Finding — the evidence above ("
                f"{', '.join(insight.evidence_ids)}) supports this reading.\n"
                f"Implication — {implications.get(insight.claim_type, '')}.\n"
                f"Recommendation (hedged) — {hedges.get(insight.claim_type, '')}."
            ),
        ))
    return recommendations


# ---------------------------------------------------------------------------
# LLM refinement (--crew only; never touches numbers or evidence)
# ---------------------------------------------------------------------------


def _refine_with_llm(run_dir: Path, cfg: Dict[str, Any],
                     log: RunLogger) -> List[str]:
    from crewai import Crew, Process

    agent = build_insight_agent(cfg)
    task = build_insight_task(agent, run_dir)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential,
                verbose=False, cache=False)
    t0 = time.monotonic()
    result = crew.kickoff(inputs={})
    log.info(STAGE, "crew kickoff finished",
             duration_s=round(time.monotonic() - t0, 3))

    outputs = getattr(result, "tasks_output", None) or []
    raws = [str(getattr(t, "raw", "") or getattr(t, "output", "") or "")
            for t in outputs]
    draft = _parse_json(raws[-1] if raws else "")
    warnings: List[str] = []
    if not isinstance(draft, dict) or "insights" not in draft:
        warnings.append("llm_refinement_missing_fallback_to_deterministic")
        return warnings

    current = _load_insights_digest(run_dir)
    merged = _merge_refinement(draft, current)
    valid, valid_recs, extra = validate_insights(
        merged.get("insights") or [], merged.get("recommendations") or [],
        run_dir)
    warnings.extend(extra)
    if not valid:
        warnings.append("llm_refinement_rejected_all_fallback_to_deterministic")
        return warnings

    log.info(STAGE, "llm refinement accepted",
             insights=len(valid), recommendations=len(valid_recs))
    _save_outputs(run_dir, valid, valid_recs, warnings)
    return warnings


def _merge_refinement(draft: Dict[str, Any],
                      current: Dict[str, Any]) -> Dict[str, Any]:
    """Adopt rewrites for ids that exist today; keep everything else."""
    by_id = {i.get("insight_id"): i for i in draft.get("insights") or []
             if isinstance(i, dict)}
    rec_by_id = {r.get("recommendation_id"): r
                 for r in draft.get("recommendations") or []
                 if isinstance(r, dict)}
    insights = []
    for original in current.get("insights") or []:
        rewrite = by_id.get(original.get("insight_id"), {})
        kept = dict(original)
        kept["title"] = rewrite.get("title", original.get("title"))
        kept["description"] = rewrite.get("description",
                                          original.get("description"))
        insights.append(kept)
    recommendations = []
    for original in current.get("recommendations") or []:
        rewrite = rec_by_id.get(original.get("recommendation_id"), {})
        kept = dict(original)
        kept["title"] = rewrite.get("title", original.get("title"))
        kept["description"] = rewrite.get("description",
                                          original.get("description"))
        recommendations.append(kept)
    return {"insights": insights, "recommendations": recommendations}


# ---------------------------------------------------------------------------
# Human review gate (§2.6, optional — config review_required)
# ---------------------------------------------------------------------------


def _apply_review_gate(run_dir: Path, cfg: Dict[str, Any],
                       log: RunLogger) -> List[str]:
    """approve / edit (evidence refs kept, re-validated) / regenerate.

    Only interactive when a terminal is attached; in automated mode
    (Flow Review / CLI without --review) the gate auto-approves and the
    warning is logged — the pipeline never blocks.
    """
    warnings: List[str] = []
    if not cfg.get("review_required"):
        return warnings
    path = run_dir / "outputs" / "insights.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    preview = json.dumps(current, ensure_ascii=False, indent=2)
    try:
        from shared.core.business_context import BusinessContextGatherer
        timeout_min = float(cfg["limits"].get("human_input_timeout_min", 5.0))
        gatherer = BusinessContextGatherer(timeout_seconds=int(timeout_min * 60))
        answer = gatherer._ask(
            "Review the generated insights. Reply 'approve', "
            "'edit: <new descriptions JSON keeping evidence_ids>', or "
            "'regenerate'.\n\n" + preview, time.monotonic() + timeout_min * 60)
    except Exception:  # noqa: BLE001 -- automated mode must not block
        answer = None
    if answer is None:
        warnings.append("review_gate_auto_approved")
        return warnings
    choice = answer.strip().lower()
    if choice.startswith("approve"):
        log.info(STAGE, "review approved")
    elif choice.startswith("edit:"):
        edited = _parse_json(answer[len("edit:"):])
        if isinstance(edited, dict) and "insights" in edited:
            merged = _merge_refinement(edited, current)
            valid, valid_recs, extra = validate_insights(
                merged.get("insights") or [],
                merged.get("recommendations") or [], run_dir)
            warnings.extend(extra)
            if valid:
                _save_outputs(run_dir, valid, valid_recs, warnings)
                log.info(STAGE, "review edits accepted",
                         insights=len(valid))
    elif choice.startswith("regenerate"):
        warnings.append("review_regenerate_requested")
    else:
        warnings.append("review_gate_auto_approved")
    return warnings


# ---------------------------------------------------------------------------
# Public entry + IO
# ---------------------------------------------------------------------------


def run_insights(run_dir: str | Path,
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
        insights, recommendations, warnings = _generate_insights(run_dir, log)
        _save_outputs(run_dir, insights, recommendations, warnings)
        if use_crew:
            warnings.extend(_refine_with_llm(run_dir, cfg, log))
            insights, recommendations, warnings = _reload_validated(run_dir)
        warnings.extend(_apply_review_gate(run_dir, cfg, log))
        status = "passed"
        summary = _summary(run_dir, insights, recommendations, warnings)
    except Exception as exc:  # noqa: BLE001 -- a failed stage must not crash
        log.error(STAGE, f"insights failed: {exc}")
        status = "failed"
        summary = {"run_id": run_id, "stage": STAGE, "status": status,
                   "error": str(exc)}
    log.stage_end(STAGE, status, time.monotonic() - start)
    summary["log_path"] = str(run_dir / "logs" / "run.jsonl")
    return summary


def _reload_validated(run_dir: Path) -> Tuple[List[Insight],
                                              List[Recommendation],
                                              List[str]]:
    current = _load_insights_digest(run_dir)
    return ([Insight(**i) for i in current.get("insights") or []],
            [Recommendation(**r) for r in current.get("recommendations")
             or []],
            current.get("warnings") or [])


def _save_outputs(run_dir: Path, insights: List[Insight],
                  recommendations: List[Recommendation],
                  warnings: List[str]) -> None:
    outputs = run_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    payload = {
        "insights": [i.model_dump() for i in insights],
        "recommendations": [r.model_dump() for r in recommendations],
        "warnings": warnings,
    }
    (outputs / "insights.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")


def _summary(run_dir: Path, insights: List[Insight],
             recommendations: List[Recommendation],
             warnings: List[str]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "status": "passed",
        "insight_count": len(insights),
        "recommendation_count": len(recommendations),
        "warnings": warnings,
        "insights_path": str(run_dir / "outputs" / "insights.json"),
    }


def _load_kpis(run_dir: Path) -> List[KpiResult]:
    path = run_dir / "outputs" / "kpis.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run Stage 5 first")
    return [KpiResult(**k) for k in
            json.loads(path.read_text(encoding="utf-8")).get("kpis", [])]


def _load_stats(run_dir: Path) -> List[StatisticalResult]:
    path = run_dir / "outputs" / "statistical_results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run Stage 5 first")
    return [StatisticalResult(**s) for s in
            json.loads(path.read_text(encoding="utf-8")).get("results", [])]


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
        prog="insight_agent",
        description="Run Insight Forge stage 6 (insights & recommendations) "
                    "on a Stage-5 run dir.")
    parser.add_argument("run_dir", help="existing run dir (from Stage 5)")
    parser.add_argument("--crew", action="store_true",
                        help="run via the real CrewAI agent (requires API key)")
    parser.add_argument("--review", action="store_true",
                        help="enable the human review gate (interactive)")
    args = parser.parse_args(argv)

    cfg = load_config(require_key=bool(args.crew))
    if args.review:
        cfg["review_required"] = True
    summary = run_insights(args.run_dir, cfg=cfg, use_crew=args.crew)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())