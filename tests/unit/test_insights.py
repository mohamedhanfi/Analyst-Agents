"""Stage 6 tests — claim taxonomy gating + validation + agent run.

The claim taxonomy (§2.6) is the contract: weak/non-significant statistics
never ground CORRELATIONAL/COMPARATIVE claims, trend claims need enough
points, CAUSAL is always rejected, every evidence_id must exist in the
registry, and recommendations may only reference surviving insights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agents.insight_agent import (STAGE, _generate_insights, run_insights)
from shared.logger import RunLogger
from shared.tools.insights import (claim_validator_tool, evidence_kinds,
                                   evidence_lookup_tool, validate_insights)

# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------


def _kpi(kpi_id: str, name: str, function: str, column: str, value: Any,
         evidence_id: str) -> Dict[str, Any]:
    return {
        "kpi_id": kpi_id, "name": name,
        "operation": {
            "function": function, "column": column, "column_a": None,
            "column_b": None, "method": None, "over_column": None,
            "period": None, "basis": None, "as_percent": None,
            "group_by": None, "filter": None, "numerator": None,
            "denominator": None,
        },
        "value": value, "evidence_id": evidence_id, "computed_by": "pandas",
    }


def _stat(test_id: str, category: str, test_name: str, variables: List[str],
          statistic: float | None, p_value: float | None,
          evidence_id: str, **extra: Any) -> Dict[str, Any]:
    return {
        "test_id": test_id, "category": category, "test_name": test_name,
        "variables": variables, "statistic": statistic, "p_value": p_value,
        "ci_low": None, "ci_high": None, "effect_size": None, "n": 100,
        "evidence_id": evidence_id, "extra": extra,
    }


def _make_run(tmp_path: Path, kpis: List[Dict[str, Any]],
              stats: List[Dict[str, Any]],
              registry: List[Dict[str, Any]]) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "outputs" / "kpis.json").write_text(
        json.dumps({"kpis": kpis}), encoding="utf-8")
    (run_dir / "outputs" / "statistical_results.json").write_text(
        json.dumps({"results": stats}), encoding="utf-8")
    (run_dir / "outputs" / "evidence_registry.json").write_text(
        json.dumps(registry), encoding="utf-8")
    (run_dir / "metadata" / "chart_metadata.json").write_text(
        json.dumps({"charts": [], "charts_truncated": False}),
        encoding="utf-8")
    return run_dir


def _registry(ids: List[str]) -> List[Dict[str, Any]]:
    return [{"evidence_id": eid,
             "source": {"aggregation": "sum", "result": 1.0}}
            for eid in ids]


_BASE_KPIS = [
    _kpi("KPI-001", "Total revenue", "sum", "revenue", 89618.13, "EV-001"),
    _kpi("KPI-002", "Average revenue", "mean", "revenue", 457.24, "EV-002"),
]

_WEAK_CORR = _stat("ST-CORR-001", "correlation", "pearson",
                   ["revenue", "quantity"], -0.014, 0.85, "EV-009")
_STRONG_CORR = _stat("ST-CORR-002", "correlation", "pearson",
                     ["revenue", "quantity"], 0.85, 0.001, "EV-010",
                     ci_low=0.7, ci_high=0.92)
_WEAK_ANOVA = _stat("ST-CMP-001", "comparison", "anova",
                    ["revenue", "category"], 1.4, 0.24, "EV-013")
_SIG_CHI2 = _stat("ST-CMP-002", "comparison", "chi2",
                  ["product", "category"], 396.0, 1e-79, "EV-021")
_SIG_CRAMERS = _stat("ST-CMP-003", "comparison", "cramers_v",
                     ["product", "category"], 0.99, 1e-79, "EV-022")
_TREND_2PT = _stat("ST-TREND-001", "trend", "growth_WoW",
                   ["revenue", "date"], None, None, "EV-030",
                   period="WoW", series=[{"period": "2024-W01", "value": 0.2},
                                         {"period": "2024-W02", "value": 0.3}])
_TREND_4PT = _stat("ST-TREND-002", "trend", "growth_MoM",
                   ["revenue", "date"], None, None, "EV-031",
                   period="MoM",
                   series=[{"period": "2024-01", "value": 0.2},
                           {"period": "2024-02", "value": 0.3},
                           {"period": "2024-03", "value": 0.25},
                           {"period": "2024-04", "value": 0.35}])


def _generate(tmp_path: Path, kpis: List[Dict[str, Any]],
              stats: List[Dict[str, Any]], registry: List[Dict[str, Any]]):
    run_dir = _make_run(tmp_path, kpis, stats, registry)
    insights, recommendations, warnings = _generate_insights(
        run_dir, RunLogger(run_dir, "run"))
    return insights, recommendations, warnings


def _with_understanding(tmp_path: Path,
                        kpis: List[Dict[str, Any]],
                        stats: List[Dict[str, Any]],
                        registry: List[Dict[str, Any]],
                        columns: List[Dict[str, Any]]) -> Path:
    """Semantic-sweep builder: run dir that also carries stage-2 roles."""
    run_dir = _make_run(tmp_path, kpis, stats, registry)
    (run_dir / "metadata" / "dataset_understanding.json").write_text(
        json.dumps({"detected_domain": "generic", "domain_confidence": 0.0,
                    "columns": columns}), encoding="utf-8")
    return run_dir


def _understanding_column(name: str, role: str, ordinal: bool = False,
                          identifier_score: float = 0.0) -> Dict[str, Any]:
    return {"name": name, "role": role, "dtype": "float64", "nunique": 10,
            "nullable": False, "override_source": "rules",
            "override_reason": "test", "identifier_score": identifier_score,
            "ordinal": ordinal}


# ---------------------------------------------------------------------------
# semantic sweep: 6.1 diversity, 6.2 identifier gate, 2.3 ordinal framing,
# 5.3 spearman preference, 6.3 confidence heuristic golden validation
# ---------------------------------------------------------------------------


def test_insights_diversify_claim_types_round_robin(tmp_path):
    """6.1: the top-N (report + recommendations) never restates one
    finding family — first insight of every claim type, then repeats."""
    insights, _, _ = _generate(
        tmp_path,
        [_kpi("KPI-001", "Total revenue", "sum", "revenue", 100.0, "EV-001"),
         _kpi("KPI-002", "Total quantity", "sum", "quantity", 50.0, "EV-002"),
         _kpi("KPI-003", "Total cost", "sum", "cost", 20.0, "EV-003"),
         _kpi("KPI-004", "Average revenue", "mean", "revenue", 10.0,
              "EV-004")],
        [_STRONG_CORR],
        _registry(["EV-001", "EV-002", "EV-003", "EV-004", "EV-010"]))
    order = [i.claim_type for i in insights]
    assert order[0] == "DESCRIPTIVE"          # first of its family
    assert order[1] == "CORRELATIONAL"        # second family first
    assert order[2:] == ["DESCRIPTIVE", "DESCRIPTIVE", "DESCRIPTIVE"]


def test_insights_never_generated_on_identifier_columns(tmp_path):
    """6.2: even if an identifier KPI slips through upstream, no insight
    rests on it."""
    kpis = [_kpi("KPI-001", "Total phone_number", "sum", "phone_number",
                 89618.13, "EV-001"),
            _kpi("KPI-002", "Total revenue", "sum", "revenue", 89618.13,
                 "EV-002")]
    run_dir = _with_understanding(
        tmp_path, kpis, [], _registry(["EV-001", "EV-002"]),
        [_understanding_column("phone_number", "identifier",
                               identifier_score=0.95),
         _understanding_column("revenue", "measure")])
    insights, _, _ = _generate_insights(run_dir, RunLogger(run_dir, "run"))
    titles = {i.title for i in insights}
    assert "Total revenue" in titles
    assert "Total phone_number" not in titles


def test_insights_correlation_prefers_spearman_for_ordinal(tmp_path):
    """5.3: ordinal pairs — the rank correlation entry is the insight."""
    pearson = _stat("ST-CORR-001", "correlation", "pearson",
                    ["rating", "revenue"], 0.72, 0.001, "EV-010")
    spearman = _stat("ST-CORR-002", "correlation", "spearman",
                     ["rating", "revenue"], 0.81, 0.0005, "EV-011",
                     recommended_method="spearman")
    run_dir = _with_understanding(
        tmp_path, _BASE_KPIS, [pearson, spearman],
        _registry(["EV-001", "EV-002", "EV-010", "EV-011"]),
        [_understanding_column("rating", "measure", ordinal=True),
         _understanding_column("revenue", "measure")])
    insights, _, _ = _generate_insights(run_dir, RunLogger(run_dir, "run"))
    corr = [i for i in insights if i.claim_type == "CORRELATIONAL"]
    assert len(corr) == 1
    assert "r = 0.81" in corr[0].description      # spearman value used
    assert corr[0].evidence_ids == ["EV-011"]


def test_insights_ordinal_scale_framing(tmp_path):
    """2.3: a mean rating of 3.4 must read as an ordered-category index,
    not a continuous measurement."""
    kpis = [_kpi("KPI-001", "Average rating", "mean", "rating", 3.4,
                 "EV-001")]
    run_dir = _with_understanding(
        tmp_path, kpis, [], _registry(["EV-001"]),
        [_understanding_column("rating", "measure", ordinal=True)])
    insights, _, _ = _generate_insights(run_dir, RunLogger(run_dir, "run"))
    desc = insights[0].description
    assert "ordinal scale" in desc
    assert "ordered categories" in desc


def test_descriptive_confidence_heuristic_golden(tmp_path):
    """6.3: the sum/count/min/max = high vs mean/median = medium heuristic
    validated against golden expectations."""
    kpis = [
        _kpi("KPI-001", "Total revenue", "sum", "revenue", 100.0, "EV-001"),
        _kpi("KPI-002", "revenue count", "count", "revenue", 10, "EV-002"),
        _kpi("KPI-003", "Min revenue", "min", "revenue", 5.0, "EV-003"),
        _kpi("KPI-004", "Max revenue", "max", "revenue", 20.0, "EV-004"),
        _kpi("KPI-005", "Average revenue", "mean", "revenue", 10.0,
             "EV-005"),
        _kpi("KPI-006", "Median revenue", "median", "revenue", 9.0,
             "EV-006"),
    ]
    insights, _, _ = _generate(
        tmp_path, kpis, [],
        _registry([f"EV-00{i}" for i in range(1, 7)]))
    by_title = {i.title: i for i in insights}
    assert by_title["Total revenue"].confidence == "high"
    assert by_title["revenue count"].confidence == "high"
    assert by_title["Min revenue"].confidence == "high"
    assert by_title["Max revenue"].confidence == "high"
    assert by_title["Average revenue"].confidence == "medium"
    assert by_title["Median revenue"].confidence == "medium"


# ---------------------------------------------------------------------------
# claim taxonomy gating
# ---------------------------------------------------------------------------


def test_descriptive_from_kpis(tmp_path):
    insights, _, _ = _generate(tmp_path, _BASE_KPIS, [],
                               _registry(["EV-001", "EV-002"]))
    descriptive = [i for i in insights if i.claim_type == "DESCRIPTIVE"]
    assert len(descriptive) == 2
    assert {i.title for i in descriptive} == {"Total revenue",
                                              "Average revenue"}
    assert all(i.evidence_ids for i in descriptive)
    by_title = {i.title: i for i in descriptive}
    assert by_title["Total revenue"].confidence == "high"
    assert by_title["Average revenue"].confidence == "medium"


def test_correlation_gated_when_weak_or_nonsignificant(tmp_path):
    insights, _, _ = _generate(tmp_path, _BASE_KPIS, [_WEAK_CORR],
                               _registry(["EV-001", "EV-002", "EV-009"]))
    assert not [i for i in insights if i.claim_type == "CORRELATIONAL"]


def test_correlation_accepted_when_significant_and_strong(tmp_path):
    insights, _, _ = _generate(tmp_path, _BASE_KPIS, [_STRONG_CORR],
                               _registry(["EV-001", "EV-002", "EV-010"]))
    corr = [i for i in insights if i.claim_type == "CORRELATIONAL"]
    assert len(corr) == 1
    assert corr[0].confidence == "high"
    assert "not cause" in corr[0].description


def test_comparative_gated_when_weak(tmp_path):
    insights, _, _ = _generate(tmp_path, _BASE_KPIS, [_WEAK_ANOVA],
                               _registry(["EV-001", "EV-002", "EV-013"]))
    assert not [i for i in insights if i.claim_type == "COMPARATIVE"]


def test_comparative_accepted_when_significant_and_deduped(tmp_path):
    insights, _, _ = _generate(
        tmp_path, _BASE_KPIS, [_SIG_CHI2, _SIG_CRAMERS],
        _registry(["EV-001", "EV-002", "EV-021", "EV-022"]))
    comp = [i for i in insights if i.claim_type == "COMPARATIVE"]
    assert len(comp) == 1          # chi2 + cramers_v = one association
    assert comp[0].confidence == "high"
    assert comp[0].evidence_ids == ["EV-021"]


def test_trend_gated_without_enough_points(tmp_path):
    insights, _, _ = _generate(tmp_path, _BASE_KPIS, [_TREND_2PT],
                               _registry(["EV-001", "EV-002", "EV-030"]))
    assert "revenue trend over 2 periods" not in {i.title for i in insights}


def test_trend_accepted_with_enough_points(tmp_path):
    insights, _, _ = _generate(tmp_path, _BASE_KPIS, [_TREND_4PT],
                               _registry(["EV-001", "EV-002", "EV-031"]))
    trend = [i for i in insights if "trend over 4 periods" in i.title]
    assert len(trend) == 1
    assert trend[0].required_evidence == ["growth_rate"]
    assert "not a forecast" in trend[0].description


def test_recommendation_chain_is_hedged(tmp_path):
    _, recommendations, _ = _generate(
        tmp_path, _BASE_KPIS, [_STRONG_CORR],
        _registry(["EV-001", "EV-002", "EV-010"]))
    assert recommendations
    text = recommendations[0].description
    for marker in ("Observation", "Finding", "Implication",
                   "Recommendation (hedged)"):
        assert marker in text, marker


# ---------------------------------------------------------------------------
# claim validation
# ---------------------------------------------------------------------------


def test_causal_never_allowed(tmp_path):
    run_dir = _make_run(tmp_path, _BASE_KPIS, [],
                        _registry(["EV-001", "EV-002"]))
    draft = [{
        "insight_id": "INS-900", "claim_type": "CAUSAL",
        "title": "discounts caused growth",
        "description": "discounts caused growth", "confidence": "high",
        "evidence_ids": ["EV-001"], "required_evidence": [],
        "related_kpis": ["KPI-001"],
    }]
    valid, _, warnings = validate_insights(draft, [], run_dir)
    assert valid == []
    assert any("CAUSAL" in w for w in warnings)


def test_unknown_claim_type_removed(tmp_path):
    run_dir = _make_run(tmp_path, _BASE_KPIS, [],
                        _registry(["EV-001", "EV-002"]))
    draft = [{
        "insight_id": "INS-901", "claim_type": "MAGICAL",
        "title": "x", "description": "y", "confidence": "high",
        "evidence_ids": ["EV-001"], "required_evidence": [],
    }]
    valid, _, warnings = validate_insights(draft, [], run_dir)
    assert valid == []
    assert any("unknown claim_type" in w for w in warnings)


def test_missing_evidence_reference_removed(tmp_path):
    run_dir = _make_run(tmp_path, _BASE_KPIS, [],
                        _registry(["EV-001"]))
    draft = [{
        "insight_id": "INS-902", "claim_type": "DESCRIPTIVE",
        "title": "x", "description": "y", "confidence": "high",
        "evidence_ids": ["EV-999"], "required_evidence": ["aggregate"],
    }]
    valid, _, warnings = validate_insights(draft, [], run_dir)
    assert valid == []
    assert any("not in registry" in w for w in warnings)


def test_empty_evidence_removed(tmp_path):
    run_dir = _make_run(tmp_path, _BASE_KPIS, [],
                        _registry(["EV-001"]))
    draft = [{
        "insight_id": "INS-903", "claim_type": "DESCRIPTIVE",
        "title": "x", "description": "y", "confidence": "high",
        "evidence_ids": [], "required_evidence": ["aggregate"],
    }]
    valid, _, warnings = validate_insights(draft, [], run_dir)
    assert valid == []
    assert any("evidence_ids is empty" in w for w in warnings)


def test_claim_type_mismatch_removed(tmp_path):
    """DESCRIPTIVE grounded only on correlation evidence must be dropped."""
    registry = [{"evidence_id": "EV-010",
                 "source": {"aggregation": "pearson", "result": 0.85}}]
    run_dir = _make_run(tmp_path, [], [_STRONG_CORR], registry)
    draft = [{
        "insight_id": "INS-904", "claim_type": "DESCRIPTIVE",
        "title": "x", "description": "y", "confidence": "high",
        "evidence_ids": ["EV-010"], "required_evidence": ["aggregate"],
    }]
    valid, _, warnings = validate_insights(draft, [], run_dir)
    assert valid == []
    assert any("does not match evidence kinds" in w for w in warnings)


def test_recommendation_refs_surviving_insights_only(tmp_path):
    run_dir = _make_run(tmp_path, _BASE_KPIS, [],
                        _registry(["EV-001", "EV-002"]))
    insight = {
        "insight_id": "INS-001", "claim_type": "DESCRIPTIVE",
        "title": "Total revenue", "description": "the Total revenue is 89,618",
        "confidence": "high", "evidence_ids": ["EV-001"],
        "required_evidence": ["aggregate"],
    }
    recs = [
        {"recommendation_id": "REC-001", "insight_id": "INS-001",
         "title": "r", "description": "d"},
        {"recommendation_id": "REC-002", "insight_id": "INS-MISSING",
         "title": "r", "description": "d"},
        {"recommendation_id": "REC-003", "insight_id": "",
         "title": "r", "description": "d"},
    ]
    valid_ins, valid_recs, warnings = validate_insights(
        [insight], recs, run_dir)
    assert [r["recommendation_id"] for r in valid_recs] == ["REC-001"]
    assert any("missing insight" in w for w in warnings)


def test_evidence_kinds_mapping(tmp_path):
    run_dir = _make_run(
        tmp_path, _BASE_KPIS,
        [_STRONG_CORR, _SIG_CHI2, _TREND_4PT],
        [{"evidence_id": "EV-060",
          "source": {"comparison": "Q4 vs Q3", "result": 27.4}},
         {"evidence_id": "EV-061", "source": {"aggregation": "sum"}}])
    kinds = evidence_kinds(run_dir)
    assert "correlation" in kinds["EV-010"]
    assert "group_comparison" in kinds["EV-021"]
    assert "growth_rate" in kinds["EV-031"]
    assert "aggregate" in kinds["EV-001"] and "aggregate" in kinds["EV-061"]
    assert "group_comparison" in kinds["EV-060"]


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def test_evidence_lookup_tool(tmp_path):
    run_dir = _make_run(tmp_path, [], [],
                        [{"evidence_id": "EV-001",
                          "source": {"aggregation": "sum", "result": 5.0}}])
    raw = evidence_lookup_tool.run("EV-001,EV-999", str(run_dir))
    found = json.loads(raw)
    assert [e["evidence_id"] for e in found] == ["EV-001"]


def test_claim_validator_tool_roundtrip(tmp_path):
    run_dir = _make_run(tmp_path, _BASE_KPIS, [],
                        _registry(["EV-001", "EV-002"]))
    draft = json.dumps({
        "insights": [{
            "insight_id": "INS-001", "claim_type": "DESCRIPTIVE",
            "title": "Total revenue",
            "description": "the Total revenue is 89,618",
            "confidence": "high", "evidence_ids": ["EV-001"],
            "required_evidence": ["aggregate"],
        }],
        "recommendations": [],
    })
    raw = claim_validator_tool.run(draft, str(run_dir))
    result = json.loads(raw)
    assert [i["insight_id"] for i in result["insights"]] == ["INS-001"]
    assert result["warnings"] == []


def test_claim_validator_tool_bad_json(tmp_path):
    run_dir = _make_run(tmp_path, [], [], [])
    result = json.loads(claim_validator_tool.run("not json", str(run_dir)))
    assert result["insights"] == []
    assert any("invalid" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# agent run
# ---------------------------------------------------------------------------


def test_run_insights_passes_on_synthetic_run(tmp_path):
    run_dir = _make_run(tmp_path, _BASE_KPIS, [_WEAK_CORR, _SIG_CHI2],
                        _registry(["EV-001", "EV-002", "EV-009", "EV-021"]))
    summary = run_insights(run_dir, cfg={"limits": {}})
    assert summary["stage"] == STAGE
    assert summary["status"] == "passed"
    assert summary["insight_count"] == 3      # 2 descriptive + 1 comparative
    assert summary["recommendation_count"] == 3
    path = run_dir / "outputs" / "insights.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["insights"]) == 3
    assert payload["warnings"] == []


def test_run_insights_crew_refinement_saves_validated_models(
        tmp_path, monkeypatch):
    """Regression: LLM refinement passed DICTS into _save_outputs which
    calls .model_dump() -> 'dict' object has no attribute 'model_dump'
    crashed the insights stage in crew mode after accepting a draft."""
    run_dir = _make_run(tmp_path, _BASE_KPIS, [_WEAK_CORR, _SIG_CHI2],
                        _registry(["EV-001", "EV-002", "EV-009", "EV-021"]))
    insights, recommendations, _ = _generate_insights(
        run_dir, RunLogger(run_dir, "run"))
    draft = {
        "insights": [{**i.model_dump(),
                      "title": "Rewritten title", "description": "Rewritten."}
                     for i in insights],
        "recommendations": [
            {**r.model_dump(), "title": "Rewritten rec"}
            for r in recommendations],
    }

    def fake_complete_json(cfg, agent_name, system, user,
                           schema=None, validator=None):
        return draft, []

    monkeypatch.setattr("shared.llm.complete_json", fake_complete_json)
    summary = run_insights(run_dir, cfg={"limits": {}},
                           use_crew=True)
    assert summary["status"] == "passed"
    payload = json.loads(
        (run_dir / "outputs" / "insights.json").read_text(encoding="utf-8"))
    assert payload["insights"][0]["title"] == "Rewritten title"