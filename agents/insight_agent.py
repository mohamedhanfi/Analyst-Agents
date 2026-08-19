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

import pandas as pd

from shared.core.semantic_guards import IDENTIFIER_NAME_RE
from shared.formatting import fmt as _fmt, parse_json as _parse_json
from shared.llm import build_llm
from shared.logger import RunLogger
from shared.prompt_guard import data_note
from shared.schemas import (
    DatasetUnderstanding,
    Insight,
    KpiResult,
    Recommendation,
    StatisticalResult,
)
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
            "\"recommendations\": [...]} and nothing else.\n"
            + data_note()
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
    understanding = _load_understanding(run_dir)
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
        # 6.2: no insight may rest on a value-aggregation of an
        # identifier-like column, even if one slipped through upstream.
        if _references_identifier(understanding, kpi.operation.column,
                                  kpi.operation.column_a,
                                  kpi.operation.column_b):
            continue
        evidence_ids = [kpi.evidence_id] if kpi.evidence_id else []
        col_label = _friendly_name(kpi.operation.column or kpi.name)
        description = (f"The {col_label} is {_fmt(kpi.value)} for this "
                       f"dataset.")
        if _is_ordinal_column(understanding, kpi.operation.column):
            # 2.3: a mean satisfaction of 3.4 is an index of ordered
            # categories, not a continuous measurement.
            description += (" This is an ordinal scale, so the value is an "
                            "index of ordered categories rather than a "
                            "continuous measurement.")
        insights.append(Insight(
            insight_id=_next_id(),
            claim_type="DESCRIPTIVE",
            title=kpi.name,
            description=description,
            confidence=_descriptive_confidence(kpi),
            evidence_ids=evidence_ids,
            required_evidence=["aggregate"],
            related_kpis=[kpi.kpi_id],
        ))

    # 2) CORRELATIONAL — only with p-value + CI, and only for meaningful
    # strength; reports strength, never cause. Deduplicate by variable pair
    # so Pearson + Spearman of the same pair don't both appear.
    seen_corr_pairs: set = set()
    for st in stats:
        if (st.category == "correlation" and st.p_value is not None
                and st.statistic is not None
                and abs(st.statistic) >= 0.3 and st.p_value < 0.05):
            if len(insights) >= _MAX_INSIGHTS:
                break
            # 6.2: identifier pairs never ground an insight.
            if _references_identifier(understanding, *st.variables):
                continue
            # 5.3: ordinal pairs — prefer the rank correlation entry. The
            # suite marks it via extra, and the stage-2 ordinal flag is the
            # authoritative backstop if that marking is ever missing.
            if _pair_has_ordinal(understanding, st.variables) \
                    and st.test_name == "pearson":
                continue
            a, b = _pair_names(st)
            pair_key = tuple(sorted([a, b]))
            if pair_key in seen_corr_pairs:
                continue
            seen_corr_pairs.add(pair_key)
            strength = "strong" if abs(st.statistic) >= 0.7 else "moderate"
            ci = ""
            if st.ci_low is not None and st.ci_high is not None:
                ci = (f" (95% CI [{_fmt(st.ci_low)}, "
                      f"{_fmt(st.ci_high)}])")
            insights.append(Insight(
                insight_id=_next_id(),
                claim_type="CORRELATIONAL",
                title=f"Correlation between {_friendly_name(a)} and {_friendly_name(b)}",
                description=(f"There is a {strength} association between "
                             f"{_friendly_name(a)} and {_friendly_name(b)} "
                             f"(r = {st.statistic:.2f}, "
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
            # 6.2: identifier columns never ground group comparisons either.
            if _references_identifier(understanding, *st.variables):
                continue
            if st.test_name in ("chi2", "cramers_v"):
                text = (f"{_friendly_name(a)} and {_friendly_name(b)} are "
                        f"significantly associated (p = {st.p_value:.3f}).")
            else:
                text = (f"{_friendly_name(a)} differs significantly across "
                        f"groups of {_friendly_name(b)} "
                        f"(p = {st.p_value:.3f}).")
            insights.append(Insight(
                insight_id=_next_id(),
                claim_type="COMPARATIVE",
                title=f"{_friendly_name(a)} differs across {_friendly_name(b)}",
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
            if _references_identifier(understanding, *st.variables):
                continue
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

    # 6.1: spread claim types round-robin so the top-N (report headline +
    # recommendations) never restates one finding family in new words.
    insights = _diversify_insights(insights)

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


def _load_understanding(run_dir: Path) -> Optional[DatasetUnderstanding]:
    """Stage-2 roles for the insight-level semantic gates (6.2/2.3)."""
    path = run_dir / "metadata" / "dataset_understanding.json"
    if not path.exists():
        return None
    try:
        return DatasetUnderstanding.model_validate(
            json.loads(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001 -- gates are best-effort
        return None


def _references_identifier(understanding: Optional[DatasetUnderstanding],
                           *columns: str) -> bool:
    """6.2: does a candidate insight rest on an identifier-like column?
    Uses Stage-2 roles plus the shared name pattern — either one firing is
    enough to drop the candidate."""
    if not understanding:
        return False
    idents = {c.name for c in understanding.columns
              if c.role == "identifier"}
    for name in columns:
        if not name:
            continue
        if name in idents:
            return True
        if IDENTIFIER_NAME_RE.search(name):
            return True
    return False


def _is_ordinal_column(understanding: Optional[DatasetUnderstanding],
                       column: str) -> bool:
    """2.3: ordinal scales get explicit interpretation framing."""
    if not understanding or not column:
        return False
    for c in understanding.columns:
        if c.name == column:
            return bool(c.ordinal)
    return False


def _pair_has_ordinal(understanding: Optional[DatasetUnderstanding],
                      variables: List[str]) -> bool:
    """5.3: rank correlation is preferred when either variable is an
    ordinal scale from stage 2."""
    if not understanding:
        return False
    ordinals = {c.name for c in understanding.columns if c.ordinal}
    return any(v in ordinals for v in (variables or []))


def _diversify_insights(insights: List[Insight]) -> List[Insight]:
    """6.1: spread claim types round-robin (stable per type) so the top-N —
    report headline + recommendation chain — never restates one finding
    family in different words. Deterministic by construction."""
    buckets: Dict[str, List[Insight]] = {}
    for insight in insights:
        buckets.setdefault(insight.claim_type, []).append(insight)
    diversified: List[Insight] = []
    while any(buckets.values()):
        for claim_type in list(buckets):
            if buckets[claim_type]:
                diversified.append(buckets[claim_type].pop(0))
    return diversified


def _pair_names(st: StatisticalResult) -> Tuple[str, str]:
    variables = st.variables or []
    a = variables[0] if variables else "the metric"
    b = variables[1] if len(variables) > 1 else "the group"
    return a, b


def _friendly_name(col: str) -> str:
    """Convert column name to human-friendly: 'performance_score' -> 'Performance Score'."""
    return col.replace("_", " ").replace(".", " ").title()


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
    recommendations: List[Recommendation] = []
    for insight in insights[: _MAX_RECOMMENDATIONS]:
        # Build a specific, actionable recommendation based on insight type.
        if insight.claim_type == "DESCRIPTIVE":
            rec_action = (
                "Establish monitoring thresholds around this baseline and "
                "design a controlled pilot to test whether a targeted "
                "intervention can shift the metric before full rollout.")
        elif insight.claim_type == "COMPARATIVE":
            rec_action = (
                "Investigate root causes behind the group gap — operational "
                "differences, resource allocation, or policy variations — "
                "then trial the most promising lever on the lagging segment.")
        elif insight.claim_type == "CORRELATIONAL":
            rec_action = (
                "Treat this association as a hypothesis: run a controlled "
                "experiment or longitudinal analysis to test causality "
                "before committing resources to act on it.")
        else:
            rec_action = "Review this finding and assess next steps."
        recommendations.append(Recommendation(
            recommendation_id=f"REC-{len(recommendations) + 1:03d}",
            insight_id=insight.insight_id,
            title=f"Recommendation on: {insight.title}",
            description=(
                f"Observation — {insight.description}\n"
                f"Finding — the evidence above ("
                f"{', '.join(insight.evidence_ids)}) supports this reading.\n"
                f"Implication — {implications.get(insight.claim_type, '')}.\n"
                f"Recommendation (hedged) — {rec_action}"
            ),
        ))
    return recommendations


# ---------------------------------------------------------------------------
# LLM refinement (--crew only; never touches numbers or evidence)
# ---------------------------------------------------------------------------


def _refine_with_llm(run_dir: Path, cfg: Dict[str, Any],
                     log: RunLogger) -> List[str]:
    from shared.llm import complete_json

    insights = _load_insights_digest(run_dir)
    system = (
        "You rewrite the wording of evidence-grounded business insights. "
        "Numbers, dates, percentages and every non-wording field are "
        "sacred: never change them, never invent new claims or evidence. "
        "Return ONLY one JSON object {\"insights\": [...], "
        "\"recommendations\": [...]}."
    )
    user = (
        "Here is the deterministic, evidence-grounded insight set for this "
        "run (already claim-validated).\n\n"
        f"{json.dumps(insights, ensure_ascii=False, indent=2)}\n\n"
        "Rewrite ONLY the title and description wording of each insight and "
        "recommendation so they read like a business analyst wrote them. "
        "Hard rules:\n"
        "- keep every field except title/description byte-identical "
        "(insight_id, claim_type, confidence, evidence_ids, "
        "required_evidence, related_kpis, recommendation_id, insight_id)\n"
        "- never add, remove or reword numbers, dates or percentages "
        "already in the text\n"
        "- never invent new claims or new evidence\n"
        + data_note()
    )
    draft, warnings = complete_json(cfg, "insight", system, user)
    if draft is None:
        warnings = warnings or ["llm_refinement_missing_fallback_to_deterministic"]
        return warnings
    if not isinstance(draft, dict) or "insights" not in draft:
        return ["llm_refinement_missing_fallback_to_deterministic"]

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
    _save_outputs(run_dir,
                  [Insight(**i) for i in valid],
                  [Recommendation(**r) for r in valid_recs],
                  warnings)
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
# Task B.3 — insight-linked chart kind overrides
# ---------------------------------------------------------------------------


def _find_cleaned_csv(run_dir: Path) -> Optional[Path]:
    cleaned = run_dir / "data" / "processed" / "cleaned_data.csv"
    if cleaned.is_file():
        return cleaned
    extracted = sorted((run_dir / "data" / "extracted").glob("*.csv"))
    return extracted[0] if extracted else None


def _apply_insight_chart_overrides(run_dir: Path, insights: List[Insight],
                                   log: RunLogger) -> None:
    """Task B.3: chart kind follows the insight claim type, not just the
    dtype shape table. Priority override above the deterministic planner;
    affected SVGs are re-rendered so the report never shows a stale shape.
    """
    from analysis.chart_planner import apply_insight_kind_overrides
    from analysis.chart_renderer import render_all
    from shared.schemas import ChartMetadata, DatasetUnderstanding

    meta_path = run_dir / "metadata" / "chart_metadata.json"
    if not meta_path.exists() or not insights:
        return
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        charts = [ChartMetadata(**c) for c in payload.get("charts", [])]
        understanding = DatasetUnderstanding(**json.loads(
            (run_dir / "metadata" / "dataset_understanding.json")
            .read_text(encoding="utf-8")))
        csv_path = _find_cleaned_csv(run_dir)
        if csv_path is None:
            return
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        kpis = _load_kpis(run_dir)
    except Exception as exc:  # noqa: BLE001 -- never fail the insights stage
        log.fallback(STAGE, f"insight chart override skipped: {exc}")
        return
    if not charts:
        return

    updated, applied = apply_insight_kind_overrides(
        charts, insights, df, understanding)
    if not applied:
        return
    render_all(updated, df, kpis, run_dir / "charts")
    payload["charts"] = [c.model_dump() for c in updated]
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    (run_dir / "outputs" / "insight_chart_overrides.json").write_text(
        json.dumps({"applied": applied}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    for override in applied:
        log.info(STAGE, "chart kind overridden by insight",
                 chart_id=override["chart_id"],
                 from_kind=override["from"], to_kind=override["to"],
                 reason=override["reason"])


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
        answer = gatherer.ask(
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
        # Task B.3: link chart kinds to the final insight claim types.
        _apply_insight_chart_overrides(run_dir, insights, log)
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