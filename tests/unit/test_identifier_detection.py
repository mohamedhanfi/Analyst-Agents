"""Tests for task A — identifier detection heuristics.

Covers: detect_identifier_like scoring (A.1), the golden phone_number
fixture, the LLM ambiguous-band fallback with reason logging (A.2), and
the KPI sanity gate (A.3) that blocks mean/sum/avg on ID-like names.
"""
from __future__ import annotations

import pandas as pd

from analysis.dsl_executor import (execute_plan,
                                   is_blocked_identifier_aggregation)
from analysis.evidence import EvidenceRegistry
from shared.core.understanding import (ColumnProfiler, apply_role_overrides,
                                       assemble_understanding,
                                       default_plan,
                                       detect_identifier_like)
from shared.schemas import DataProfile, DslOperation, KpiCandidate
from shared.logger import RunLogger


def _profile_for(df: pd.DataFrame, name: str = "sales.csv") -> DataProfile:
    return DataProfile(
        file_name=name, file_hash="sha256:abc",
        row_count=len(df), column_count=len(df.columns),
        columns=list(df.columns),
        column_types={c: str(df[c].dtype) for c in df.columns},
        nunique={c: int(df[c].nunique()) for c in df.columns},
        missing_values={c: int(df[c].isna().sum()) for c in df.columns},
        sample=df.head(20).to_dict("records"),
        validation_status="passed",
    )


PHONE_DF = pd.DataFrame({
    "phone_number": [201000000001 + i for i in range(120)],
    "revenue": [100.5 + i * 1.7 for i in range(120)],
    "city": ["Cairo" if i % 2 else "Giza" for i in range(120)],
})


# ---------------------------------------------------------------------------
# A.1 — detect_identifier_like scoring
# ---------------------------------------------------------------------------


def test_phone_number_scores_high():
    signal = detect_identifier_like("phone_number", PHONE_DF["phone_number"])
    assert signal.score >= 0.7
    assert any("name" in s for s in signal.signals)


def test_revenue_scores_low():
    signal = detect_identifier_like("revenue", PHONE_DF["revenue"])
    assert signal.score < 0.7
    assert signal.signals == [] or all("name" not in s for s in signal.signals)


def test_zip_with_leading_zeros_scores_high():
    zips = pd.Series([f"0{i:04d}" for i in range(100)])  # e.g. 000123
    signal = detect_identifier_like("zip_code", zips)
    assert signal.score >= 0.7
    assert any("leading zeros" in s for s in signal.signals)


def test_ambiguous_band_column_scores_mid():
    """A unique constant-length integral column with a neutral name lands
    in the ambiguous band (0.3–0.7) — eligible for the LLM fallback."""
    values = pd.Series([100000 + i for i in range(100)])
    signal = detect_identifier_like("memberno", values)
    assert 0.3 <= signal.score < 0.7


# ---------------------------------------------------------------------------
# A.1 golden fixture — phone_number is never a measure
# ---------------------------------------------------------------------------


def test_golden_phone_number_identified_as_identifier():
    profile = _profile_for(PHONE_DF)
    facts = ColumnProfiler().profile_columns(profile, df=PHONE_DF)
    by_name = {f.name: f for f in facts}
    assert by_name["phone_number"].suggested_role == "identifier"
    assert by_name["revenue"].suggested_role == "measure"


def test_golden_phone_number_not_in_default_plan():
    plan = default_plan(_profile_for(PHONE_DF))
    names = [k.name for k in plan.candidate_kpis]
    assert "Total phone_number" not in names
    assert "Average phone_number" not in names
    assert any("revenue" in n.lower() for n in names)


def test_golden_phone_number_no_kpi_executed():
    profile = _profile_for(PHONE_DF)
    plan = default_plan(profile)
    registry = EvidenceRegistry(run_id="test")
    results = execute_plan(PHONE_DF, plan, registry)
    for result in results:
        column = str(result.operation.column)
        function = str(result.operation.function)
        assert not (column == "phone_number"
                    and function in ("sum", "mean", "avg"))


def test_identifier_signal_logged_in_understanding():
    profile = _profile_for(PHONE_DF)
    facts = ColumnProfiler().profile_columns(profile, df=PHONE_DF)
    understanding = assemble_understanding(
        profile=profile, facts=facts, role_overrides={},
        domain=("sales", 0.7, []), context=None, limitations=[])
    phone = next(c for c in understanding.columns
                 if c.name == "phone_number")
    assert phone.role == "identifier"
    assert phone.identifier_score >= 0.7
    assert "identifier heuristic" in phone.override_reason


# ---------------------------------------------------------------------------
# A.2 — LLM fallback for the ambiguous band
# ---------------------------------------------------------------------------


def test_ambiguous_llm_fallback_reclassifies(tmp_path, monkeypatch):
    from agents.understanding_agent import (_apply_identifier_reasons,
                                            _resolve_ambiguous_identifiers)

    df = pd.DataFrame({
        "memberno": [100000 + i for i in range(100)],
        "revenue": [10.0 + i for i in range(100)],
    })
    profile = _profile_for(df)
    facts = ColumnProfiler().profile_columns(profile, df=df)
    ambiguous = [f for f in facts if f.name == "memberno"]
    assert 0.3 <= ambiguous[0].identifier_score < 0.7
    assert ambiguous[0].suggested_role == "measure"

    def fake_complete_json(cfg, agent_name, system, user, **kwargs):
        return ({"memberno": {"role": "identifier", "confidence": 0.9,
                              "reason": "constant-length unique code"}}, [])

    monkeypatch.setattr("agents.understanding_agent.complete_json",
                        fake_complete_json)
    cfg = {"understanding": {"identifier_confidence_threshold": 0.7,
                             "identifier_llm_fallback": True}}
    log = RunLogger(tmp_path, "t")
    overrides, reasons = _resolve_ambiguous_identifiers(
        facts, profile, cfg, use_crew=True, log=log)
    assert overrides == {"memberno": "identifier"}
    assert "constant-length unique code" in reasons["memberno"]

    understanding = assemble_understanding(
        profile=profile, facts=facts, role_overrides=overrides,
        domain=("generic", 0.0, []), context=None, limitations=[])
    _apply_identifier_reasons(understanding, reasons)
    col = next(c for c in understanding.columns if c.name == "memberno")
    assert col.role == "identifier"
    assert col.override_source == "llm"
    assert "identifier fallback" in col.override_reason
    assert "constant-length unique code" in col.override_reason


def test_llm_fallback_gated_by_config(tmp_path, monkeypatch):
    from agents.understanding_agent import _resolve_ambiguous_identifiers

    df = pd.DataFrame({"memberno": [100000 + i for i in range(100)]})
    profile = _profile_for(df)
    facts = ColumnProfiler().profile_columns(profile, df=df)

    def boom(*args, **kwargs):
        raise AssertionError("complete_json must not be called")

    monkeypatch.setattr("agents.understanding_agent.complete_json", boom)
    cfg = {"understanding": {"identifier_llm_fallback": False}}
    overrides, reasons = _resolve_ambiguous_identifiers(
        facts, profile, cfg, use_crew=True, log=RunLogger(tmp_path, "t"))
    assert overrides == {} and reasons == {}
    # deterministic mode never calls the LLM either
    overrides, reasons = _resolve_ambiguous_identifiers(
        facts, profile, cfg, use_crew=False, log=RunLogger(tmp_path, "t"))
    assert overrides == {} and reasons == {}


# ---------------------------------------------------------------------------
# A.3 — KPI sanity gate (defense-in-depth)
# ---------------------------------------------------------------------------


def test_sanity_gate_blocks_mean_sum_avg_on_identifier_names():
    assert is_blocked_identifier_aggregation(
        DslOperation(function="sum", column="phone_number"))
    assert is_blocked_identifier_aggregation(
        DslOperation(function="mean", column="zip_code"))
    assert not is_blocked_identifier_aggregation(
        DslOperation(function="sum", column="revenue"))
    assert not is_blocked_identifier_aggregation(
        DslOperation(function="count", column="phone_number"))


def test_sanity_gate_blocks_even_when_role_is_measure():
    """A.3 works regardless of the upstream role assignment: a synthetic
    test force-labels phone_number as a measure — the gate still rejects."""
    df = pd.DataFrame({"phone_number": [20100000001, 20100000002],
                       "revenue": [10.0, 20.0]})
    registry = EvidenceRegistry(run_id="test")
    plan = [
        KpiCandidate(kpi_id="K1", name="Total phone",
                     operation=DslOperation(function="sum",
                                            column="phone_number")),
        KpiCandidate(kpi_id="K2", name="Total revenue",
                     operation=DslOperation(function="sum", column="revenue")),
    ]
    results = execute_plan(df, {"candidate_kpis": plan}, registry)
    assert len(results) == 1
    assert results[0].kpi_id == "K2"
