"""Stage 6 tools — evidence lookup + claim validation (§2.6).

CrewAI @tool wrappers around pure, deterministic functions. The claim
validator is the single gatekeeper for `outputs/insights.json`: evidence_ids
must be non-empty and exist in the registry, the claim type must match the
evidence kinds derived from the stage-5 artifacts, and recommendations may
only reference surviving insights. Failures are dropped and logged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from crewai.tools import tool

# statistical-result category -> evidence kind used for claim matching
_CATEGORY_KIND = {
    "descriptive": "descriptive",
    "correlation": "correlation",
    "comparison": "group_comparison",
    "trend": "growth_rate",
}

# claim type -> kinds of evidence that may ground it (§2.6 taxonomy)
_REQUIRED_KINDS: Dict[str, Set[str]] = {
    "DESCRIPTIVE": {"aggregate", "descriptive", "growth_rate"},
    "COMPARATIVE": {"group_comparison"},
    "CORRELATIONAL": {"correlation"},
    "PREDICTIVE": {"forecast"},
    "CAUSAL": set(),          # never allowed without causal methodology
}

_AGGREGATION_KINDS = ("sum", "mean", "count", "min", "max", "median")


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_evidence_ids(run_dir: str | Path) -> Set[str]:
    """All evidence ids registered in outputs/evidence_registry.json."""
    payload = _load_json(Path(run_dir) / "outputs" / "evidence_registry.json")
    if isinstance(payload, list):
        return {e["evidence_id"] for e in payload if isinstance(e, dict)}
    return set()


def evidence_kinds(run_dir: str | Path) -> Dict[str, Set[str]]:
    """Map every evidence_id -> kinds derived from the stage-5 artifacts.

    Sources: statistical-result category (correlation/comparison/trend/
    descriptive), KPI linkage (aggregate), chart linkage, and the registry
    lineage itself (comparison present -> group_comparison; numeric
    aggregation -> aggregate).
    """
    run_dir = Path(run_dir)
    kinds: Dict[str, Set[str]] = {}

    def add(evidence_id: Any, kind: str) -> None:
        if evidence_id:
            kinds.setdefault(str(evidence_id), set()).add(kind)

    for result in _load_json(run_dir / "outputs"
                             / "statistical_results.json").get("results", []):
        if not isinstance(result, dict):
            continue
        kind = _CATEGORY_KIND.get(result.get("category"))
        if kind:
            add(result.get("evidence_id"), kind)
    for kpi in _load_json(run_dir / "outputs" / "kpis.json").get("kpis", []):
        if isinstance(kpi, dict):
            add(kpi.get("evidence_id"), "aggregate")
    for chart in _load_json(run_dir / "metadata"
                            / "chart_metadata.json").get("charts", []):
        if isinstance(chart, dict):
            add(chart.get("evidence_id"), "chart")
    payload = _load_json(run_dir / "outputs" / "evidence_registry.json")
    entries = payload if isinstance(payload, list) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        src = entry.get("source") or {}
        if src.get("comparison"):
            add(entry.get("evidence_id"), "group_comparison")
        if src.get("aggregation") in _AGGREGATION_KINDS:
            add(entry.get("evidence_id"), "aggregate")
    return kinds


def _insight_failure(insight: Dict[str, Any], ids: Set[str],
                     kinds: Dict[str, Set[str]]) -> str | None:
    """Reason this insight is invalid, or None when it may be kept."""
    claim_type = insight.get("claim_type")
    if claim_type not in _REQUIRED_KINDS:
        return f"unknown claim_type {claim_type!r}"
    if not _REQUIRED_KINDS[claim_type]:
        return "claim type CAUSAL is never allowed without causal methodology"
    evidence_ids = [e for e in (insight.get("evidence_ids") or []) if e]
    if not evidence_ids:
        return "evidence_ids is empty"
    missing = [e for e in evidence_ids if e not in ids]
    if missing:
        return f"evidence refs not in registry: {missing}"
    available: Set[str] = set()
    for e in evidence_ids:
        available |= kinds.get(e, set())
    if not (available & _REQUIRED_KINDS[claim_type]):
        return ("claim type does not match evidence kinds "
                f"(has {sorted(available) or ['none']})")
    if insight.get("confidence") not in ("high", "medium", "low"):
        return f"bad confidence {insight.get('confidence')!r}"
    if not str(insight.get("title") or "").strip() \
            or not str(insight.get("description") or "").strip():
        return "title/description empty"
    return None


def validate_insights(insights: List[Dict[str, Any]],
                      recommendations: List[Dict[str, Any]],
                      run_dir: str | Path
                      ) -> Tuple[List[Dict[str, Any]],
                                 List[Dict[str, Any]],
                                 List[str]]:
    """Claim validation before saving (§2.6).

    Any failure -> the item is removed and logged as a warning. Returns the
    surviving (insights, recommendations, warnings).
    """
    ids = load_evidence_ids(run_dir)
    kinds = evidence_kinds(run_dir)
    warnings: List[str] = []

    valid: List[Dict[str, Any]] = []
    for insight in insights:
        why = _insight_failure(insight, ids, kinds)
        if why:
            warnings.append(f"insight {insight.get('insight_id', '?')} "
                            f"rejected: {why}")
            continue
        valid.append(insight)
    surviving = {i["insight_id"] for i in valid}

    valid_recs: List[Dict[str, Any]] = []
    for rec in recommendations:
        ref = rec.get("insight_id")
        if not ref or ref not in surviving:
            warnings.append(
                f"recommendation {rec.get('recommendation_id', '?')} "
                f"rejected: references missing insight {ref!r}")
            continue
        valid_recs.append(rec)
    return valid, valid_recs, warnings


@tool("evidence_lookup_tool")
def evidence_lookup_tool(evidence_ids: str, run_dir: str) -> str:
    """Look up evidence entries by id (comma-separated) in the run's
    evidence_registry.json. Returns the matching entries with full lineage."""
    run_dir = Path(run_dir)
    wanted = {e.strip() for e in evidence_ids.split(",") if e.strip()}
    payload = _load_json(run_dir / "outputs" / "evidence_registry.json")
    entries = payload if isinstance(payload, list) else []
    found = [e for e in entries if isinstance(e, dict)
             and e.get("evidence_id") in wanted]
    return json.dumps(found, ensure_ascii=False, indent=2)


@tool("claim_validator_tool")
def claim_validator_tool(insights_json: str, run_dir: str) -> str:
    """Validate a draft insights payload ({"insights": [...],
    "recommendations": [...]}) against the §2.6 claim taxonomy. Returns
    {"insights": [...], "recommendations": [...], "warnings": [...]} with all
    invalid items removed and a reason logged per removal."""
    try:
        draft = json.loads(insights_json)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"insights": [], "recommendations": [],
                           "warnings": ["invalid insights JSON"]},
                          ensure_ascii=False, indent=2)
    if not isinstance(draft, dict):
        return json.dumps({"insights": [], "recommendations": [],
                           "warnings": ["invalid insights payload"]},
                          ensure_ascii=False, indent=2)
    valid, valid_recs, warnings = validate_insights(
        draft.get("insights") or [], draft.get("recommendations") or [],
        run_dir)
    return json.dumps({"insights": valid, "recommendations": valid_recs,
                       "warnings": warnings},
                      ensure_ascii=False, indent=2)