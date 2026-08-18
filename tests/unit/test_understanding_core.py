"""Unit tests for stage-2 core helpers (§2.2): role overrides, domain
heuristic, default plan, understanding assembly."""
from __future__ import annotations

from shared.core.understanding import (
    ColumnFacts,
    ColumnProfiler,
    assemble_understanding,
    default_plan,
    detect_domain_heuristic,
    apply_role_overrides,
)
from shared.dsl_validator import validate_plan
from shared.schemas import BusinessContext, DataProfile


def make_profile() -> DataProfile:
    return DataProfile(
        file_name="sales.csv", file_hash="sha256:abc",
        row_count=100, column_count=2,
        columns=["zip_code", "city"],
        column_types={"zip_code": "float64", "city": "object"},
        nunique={"zip_code": 80, "city": 12},
        sample=[], validation_status="passed",
    )


def facts_for(profile):
    return ColumnProfiler().profile_columns(profile)


# ---------------------------------------------------------------------------
# apply_role_overrides
# ---------------------------------------------------------------------------


def test_override_into_alternate_accepted():
    facts = [ColumnFacts(name="rating", dtype="float64", nunique=5,
                         nullable=False, suggested_role="measure",
                         alternate_roles=["categorical"])]
    cols = apply_role_overrides(facts, {"rating": "categorical"})
    assert cols[0].role == "categorical"


def test_override_to_identifier_accepted():
    facts = facts_for(make_profile())
    cols = apply_role_overrides(facts, {"zip_code": "identifier"})
    assert cols[0].role == "identifier"


def test_invalid_override_keeps_rule_role():
    facts = facts_for(make_profile())
    cols = apply_role_overrides(facts, {"city": "measure"})
    assert cols[1].role == "dimension"


def test_override_reason_logged_accepted():
    facts = [ColumnFacts(name="rating", dtype="float64", nunique=5,
                         nullable=False, suggested_role="measure",
                         alternate_roles=["categorical"])]
    cols = apply_role_overrides(facts, {"rating": "categorical"})
    assert cols[0].override_source == "llm"
    assert "accepted" in cols[0].override_reason


def test_override_reason_logged_rejected():
    facts = [ColumnFacts(name="city", dtype="object", nunique=12,
                         nullable=False, suggested_role="categorical",
                         alternate_roles=[])]
    cols = apply_role_overrides(facts, {"city": "measure"})
    assert cols[0].role == "categorical"
    assert cols[0].override_source == "llm"
    assert "rejected" in cols[0].override_reason


def test_no_override_reason_rules():
    facts = [ColumnFacts(name="city", dtype="object", nunique=12,
                         nullable=False, suggested_role="categorical",
                         alternate_roles=[])]
    cols = apply_role_overrides(facts, {})
    assert cols[0].override_source == "rules"
    assert "no override" in cols[0].override_reason


def test_unknown_column_ignored():
    facts = facts_for(make_profile())
    cols = apply_role_overrides(facts, {"ghost": "identifier"})
    # zip_code is an identifier via the semantic heuristic (task A.1) —
    # the unknown override is ignored and its rule-based role is kept.
    assert [c.role for c in cols] == ["identifier", "dimension"]


# ---------------------------------------------------------------------------
# detect_domain_heuristic
# ---------------------------------------------------------------------------


def _ctx(**overrides):
    base = dict(file_name="x.csv", goal_summary="", business_questions=[],
                answers={}, context_confidence=0.0, generic_mode=False)
    base.update(overrides)
    return BusinessContext(**base)


def test_generic_mode_returns_generic():
    assert detect_domain_heuristic(_ctx(generic_mode=True)) == ("generic", 0.0)


def test_sales_keyword_detected():
    domain, confidence = detect_domain_heuristic(
        _ctx(answers={"domain": "sales and revenue tracking"}))
    assert domain == "sales"
    assert 0.0 < confidence <= 0.9


def test_no_match_returns_generic():
    domain, confidence = detect_domain_heuristic(
        _ctx(answers={"domain": "quantum cucumbers"}))
    assert domain == "generic"
    assert confidence == 0.0


# ---------------------------------------------------------------------------
# default_plan
# ---------------------------------------------------------------------------


def _wide_profile() -> DataProfile:
    return DataProfile(
        file_name="sales.csv", file_hash="sha256:abc",
        row_count=100, column_count=5,
        columns=["order_id", "date", "product", "revenue", "quantity"],
        column_types={"order_id": "int64", "date": "datetime64[ns]",
                      "product": "object", "revenue": "float64",
                      "quantity": "int64"},
        nunique={"order_id": 100, "date": 60, "product": 5,
                 "revenue": 87, "quantity": 30},
        sample=[], validation_status="passed",
    )


def test_default_plan_covers_all_functions():
    plan = default_plan(_wide_profile())
    assert validate_plan(plan) == []
    names = [k.name for k in plan.candidate_kpis]
    assert "Total revenue" in names
    assert "Average revenue" in names
    assert "order_id count" in names
    assert "revenue YoY growth" in names
    assert "Correlation revenue x quantity" in names
    assert plan.statistical_tests == ["descriptive", "correlation", "trend",
                                      "anova"]
    assert plan.has_temporal_data is True


def test_default_plan_minimal_profile():
    p = DataProfile(
        file_name="m.csv", file_hash="h", row_count=10, column_count=1,
        columns=["note"], column_types={"note": "object"},
        nunique={"note": 8}, sample=[], validation_status="passed")
    plan = default_plan(p)
    assert validate_plan(plan) == []
    assert plan.candidate_kpis == []
    assert plan.statistical_tests == ["descriptive"]


# ---------------------------------------------------------------------------
# assemble_understanding
# ---------------------------------------------------------------------------


def test_assemble_groups_columns_by_role():
    profile = _wide_profile()
    facts = ColumnProfiler().profile_columns(profile)
    understanding = assemble_understanding(
        profile=profile, facts=facts, role_overrides={},
        domain=("sales", 0.7, ["Order", "Product"]),
        context=_ctx(), limitations=["kpi x dropped"])
    assert understanding.detected_domain == "sales"
    assert understanding.domain_confidence == 0.7
    assert understanding.entities == ["Order", "Product"]
    assert understanding.identifiers == ["order_id"]
    assert understanding.temporal_columns == ["date"]
    assert understanding.measures == ["revenue", "quantity"]
    assert understanding.dimensions == ["product"]
    assert understanding.has_temporal_data is True
    assert understanding.limitations == ["kpi x dropped"]
    assert len(understanding.columns) == 5
