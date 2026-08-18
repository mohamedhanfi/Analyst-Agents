"""Semantic sweep tests — §0 shared gates + the golden semantic-traps fixture
(one dataset covering 2.2/2.3/2.4/3.1/3.2/4.1/4.2/5.2/5.3/6.1/6.2/8.1 at once).

Golden fixture: tests/fixtures/semantic_traps.csv
  customer_id   -> identifier (all-unique id name)
  phone_number  -> identifier (name + leading zeros)
  gender_code   -> encoded categorical (0/1)  [2.2]
  rating        -> ordinal scale 1-5          [2.3]
  revenue       -> measure                    [controls]
  price_tag     -> mixed units ("100 USD", "EGP 500", "50 EUR", "200") [2.4]
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.dsl_executor import execute_plan, is_blocked_identifier_aggregation
from analysis.evidence import EvidenceRegistry
from analysis.generic.correlation import run_correlation
from analysis.qa_recompute import QaCheck, check_semantic_relevance
from analysis.qa_verdict import decide_verdict
from analysis.report_builder import _kpi_relevance, render_kpis
from shared.core.cleaning import (
    build_strategy,
    execute_strategy,
    normalize_strategy,
)
from shared.core.data_quality import check_invalid_values
from shared.core.semantic_guards import (
    IDENTIFIER_NAME_RE,
    NEGATIVE_ALLOWED_RE,
    aggregation_is_meaningful,
    is_code_like,
    is_identifier_like,
    is_mixed_unit,
    is_ordinal_like,
)
from shared.core.understanding import (
    ColumnProfiler,
    apply_role_overrides,
    default_plan,
)
from shared.schemas import (
    AnalysisPlan,
    BusinessContext,
    DataProfile,
    DataQualityReport,
    DatasetUnderstanding,
    DslOperation,
)
from shared.tools.cleaning import iqr_outlier_tool

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


# ---------------------------------------------------------------------------
# §0 — shared gates
# ---------------------------------------------------------------------------


def test_is_code_like_flags_encoded_categoricals():
    assert is_code_like("gender_code", [0, 1, 0, 1, 0])
    assert is_code_like("status", [1, 2, 3, 2, 1])
    assert not is_code_like("revenue", [100.5, 200.0, 150.0])
    assert not is_code_like("rating", [1, 2, 3, 4, 5])      # ordinal 2.3
    assert not is_code_like("quantity", list(range(1, 31)))  # >10 levels
    assert not is_code_like("price_tag", ["100 USD", "EGP 500"])


def test_is_ordinal_like_flags_scales():
    assert is_ordinal_like("rating", [1, 2, 3, 4, 5])
    assert is_ordinal_like("satisfaction", list(range(1, 11)))
    assert not is_ordinal_like("revenue", [100.5, 200.0])
    assert not is_ordinal_like("gender_code", [0, 1, 0, 1])  # 2 levels


def test_is_mixed_unit_detects_unit_strings():
    assert is_mixed_unit(["$100", "200", "EGP 500"])
    assert is_mixed_unit(["100 kg", "50", "12"])
    assert is_mixed_unit(["100 km", "200 km"])  # unit-tagged values -> flag
    assert not is_mixed_unit(["100", "200.5", "12"])
    assert not is_mixed_unit(["SKU-100", "order100", "ABC"])
    assert not is_mixed_unit(["abc", "def"])


def test_aggregation_is_meaningful_single_gate():
    ok, _ = aggregation_is_meaningful("phone_number", None, "sum")
    assert not ok
    ok, _ = aggregation_is_meaningful("customer_id", None, "correlation")
    assert not ok
    ok, reason = aggregation_is_meaningful("phone_number", None, "count")
    assert ok and not reason
    ok, _ = aggregation_is_meaningful("gender_code", [0, 1, 0, 1], "mean")
    assert not ok                     # code-like categorical
    ok, _ = aggregation_is_meaningful("rating", [1, 2, 3, 4, 5], "mean")
    assert ok                         # ordinal stays meanable (2.3)
    ok, _ = aggregation_is_meaningful("revenue", None, "sum")
    assert ok


def test_negative_allowed_re():
    assert NEGATIVE_ALLOWED_RE.search("temperature_celsius")
    assert NEGATIVE_ALLOWED_RE.search("account_balance")
    assert NEGATIVE_ALLOWED_RE.search("growth_rate")
    assert not NEGATIVE_ALLOWED_RE.search("revenue")
    assert not NEGATIVE_ALLOWED_RE.search("quantity")


def test_is_identifier_like_name_gate_matches_phone():
    signal = is_identifier_like("phone_number")
    assert signal.score >= 0.75
    assert IDENTIFIER_NAME_RE.search("zip_code")
    assert not IDENTIFIER_NAME_RE.search("revenue")


# ---------------------------------------------------------------------------
# Golden fixture helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def traps():
    df = pd.read_csv(FIXTURES / "semantic_traps.csv", encoding="utf-8-sig")
    profile = DataProfile(
        file_name="semantic_traps.csv", file_hash="sha256:traps",
        row_count=len(df), column_count=len(df.columns),
        columns=list(df.columns),
        column_types={c: str(df[c].dtype) for c in df.columns},
        missing_values={}, nunique={c: int(df[c].nunique())
                                   for c in df.columns},
        sample=[], validation_status="passed")
    return df, profile


@pytest.fixture(scope="module")
def traps_facts(traps):
    df, profile = traps
    return ColumnProfiler().profile_columns(profile, df=df)


@pytest.fixture(scope="module")
def traps_understanding(traps_facts):
    cols = apply_role_overrides(traps_facts, {})
    return DatasetUnderstanding(
        detected_domain="generic", domain_confidence=0.0, entities=[],
        temporal_columns=[], dimensions=[], measures=[],
        identifiers=[], columns=cols, has_temporal_data=False)


# ---------------------------------------------------------------------------
# Stage 2 — 2.2 encoded categoricals, 2.3 ordinal
# ---------------------------------------------------------------------------


def test_stage2_roles_on_golden_traps(traps_facts):
    by_name = {f.name: f for f in traps_facts}
    assert by_name["phone_number"].suggested_role == "identifier"
    assert by_name["customer_id"].suggested_role == "identifier"
    assert by_name["gender_code"].suggested_role == "categorical"  # 2.2
    assert by_name["rating"].suggested_role == "measure"           # 2.3
    assert by_name["rating"].ordinal is True
    assert by_name["revenue"].suggested_role == "measure"
    assert by_name["revenue"].code_like is False


def test_stage2_ordinal_survives_into_understanding(traps_understanding):
    rating = next(c for c in traps_understanding.columns
                  if c.name == "rating")
    assert rating.ordinal is True
    assert rating.role == "measure"


def test_default_plan_never_proposes_trap_aggregations(traps):
    df, profile = traps
    plan = default_plan(profile, df=df)
    functions = {(kpi.operation.function, kpi.operation.column)
                 for kpi in plan.candidate_kpis}
    assert not any(col == "phone_number" for _, col in functions)
    assert not any(col == "gender_code" for _, col in functions)
    assert ("sum", "revenue") in functions
    assert any(func == "count" and col in ("customer_id", "phone_number")
               for func, col in functions)   # identifier count is allowed


# ---------------------------------------------------------------------------
# 5.2 / 5.3 — correlation gates
# ---------------------------------------------------------------------------


def test_dsl_correlation_on_identifier_fails_cleanly(traps):
    df, _ = traps
    plan = AnalysisPlan(candidate_kpis=[
        __import__("shared.schemas", fromlist=["KpiCandidate"]).KpiCandidate(
            kpi_id="KPI-BAD", name="Correlation customer_id x revenue",
            operation=DslOperation(function="correlation", column_a="customer_id",
                                   column_b="revenue", method="pearson"))])
    registry = EvidenceRegistry(file_hash="t", sheet=None,
                                transformations=[])
    results = execute_plan(df, plan, registry)
    assert results[0].value is None           # gate 5.2 -> failed op


def test_suite_correlation_skips_identifier_pairs(traps, traps_understanding):
    df, _ = traps
    registry = EvidenceRegistry(file_hash="t", sheet=None,
                                transformations=[])
    results = run_correlation(df, traps_understanding, registry)
    variables = {tuple(r.variables) for r in results}
    assert not any("phone_number" in v or "customer_id" in v
                   for v in variables)


def test_suite_correlation_recommends_spearman_for_ordinal(traps,
                                                           traps_understanding):
    df, _ = traps
    registry = EvidenceRegistry(file_hash="t", sheet=None,
                                transformations=[])
    results = run_correlation(df, traps_understanding, registry)
    rating_pairs = [r for r in results if "rating" in r.variables]
    assert rating_pairs
    spearman = [r for r in rating_pairs if r.test_name == "spearman"]
    pearson = [r for r in rating_pairs if r.test_name == "pearson"]
    assert spearman and pearson
    assert all((r.extra or {}).get("recommended_method") == "spearman"
               for r in spearman)                     # 5.3
    assert all((r.extra or {}).get("recommended_method") is None
               for r in pearson)


def test_is_blocked_identifier_aggregation_uses_shared_gate():
    assert is_blocked_identifier_aggregation(
        DslOperation(function="sum", column="phone_number"))
    assert not is_blocked_identifier_aggregation(
        DslOperation(function="count", column="phone_number"))


# ---------------------------------------------------------------------------
# Stage 3 — 3.2 negatives by semantics, 2.4 mixed units
# ---------------------------------------------------------------------------


def test_dq_negative_measure_allowed_for_temperature():
    df = pd.DataFrame({"temperature_celsius": [-5.0, 12.0, -2.5, 4.0],
                       "revenue": [100.0, -50.0, 30.0, 40.0]})
    from shared.schemas import ColumnUnderstanding
    understanding = DatasetUnderstanding(
        detected_domain="generic", domain_confidence=0.0, columns=[
            ColumnUnderstanding(name="temperature_celsius", role="measure",
                                dtype="float64", nunique=4, nullable=False),
            ColumnUnderstanding(name="revenue", role="measure",
                                dtype="float64", nunique=4, nullable=False),
        ])
    issues = check_invalid_values(understanding, df)
    flagged = {(i.column, i.detail) for i in issues}
    assert ("revenue", "negative") in flagged          # still flagged
    assert not any(col == "temperature_celsius" and detail == "negative"
                   for col, detail in flagged)        # 3.2 allowed


def test_dq_mixed_units_warning(traps, traps_understanding):
    df, _ = traps
    issues = check_invalid_values(traps_understanding, df)
    assert any(i.column == "price_tag" and i.detail == "mixed_units"
               for i in issues)


# ---------------------------------------------------------------------------
# Stage 4 — 3.1 outlier role gate, 4.1 imputation by role, 4.2 drop_negative
# ---------------------------------------------------------------------------


def test_cleaning_drop_negative_skipped_for_semantic_negatives():
    from shared.schemas import ColumnUnderstanding
    understanding = DatasetUnderstanding(
        detected_domain="generic", domain_confidence=0.0, columns=[
            ColumnUnderstanding(name="temperature_celsius", role="measure",
                                dtype="float64", nunique=4, nullable=False),
            ColumnUnderstanding(name="revenue", role="measure",
                                dtype="float64", nunique=4, nullable=False),
        ])
    report = DataQualityReport(status="needs_repair",
                               invalid={"temperature_celsius": ["negative"]},
                               missingness={})
    strategy = build_strategy(understanding, report)
    by_name = {c["column"]: c for c in strategy["columns"]}
    assert by_name["temperature_celsius"]["action"] != "drop_negative"  # 4.2
    report2 = DataQualityReport(status="needs_repair",
                                invalid={"revenue": ["negative"]},
                                missingness={})
    strategy2 = build_strategy(understanding, report2)
    by_name2 = {c["column"]: c for c in strategy2["columns"]}
    assert by_name2["revenue"]["action"] == "drop_negative"


def test_cleaning_outlier_proposal_rejected_for_non_measures(
        traps_understanding):
    raw = {"columns": [], "deduplicate": False,
           "outliers": {"gender_code": "flag", "revenue": "flag"}}
    strategy, errors = normalize_strategy(raw, traps_understanding)
    assert "revenue" in strategy["outliers"]
    assert "gender_code" not in strategy["outliers"]      # 3.1
    assert any("not a measure" in e for e in errors)


def test_iqr_tool_rejects_identifier_column(traps, traps_understanding):
    df, _ = traps
    csv_path = FIXTURES / "semantic_traps.csv"
    raw = iqr_outlier_tool.run(str(csv_path),
                               traps_understanding.model_dump_json(),
                               "customer_id", "flag")
    result = json.loads(raw)
    assert "error" in result                              # 3.1


def test_cleaning_never_median_fills_identifier(traps_understanding):
    """4.1: identifier columns with missingness always drop rows/columns —
    mean/median imputation never applies (role table is authoritative)."""
    df = pd.DataFrame({"customer_id": ["C1", "C2", None, "C4", "C5"]})
    report = DataQualityReport(
        status="passed", missingness={
            "by_column": {"customer_id": {"missing": 1, "rate": 0.2,
                                          "assessment": "MCAR"}}})
    strategy = build_strategy(traps_understanding, report)
    by_name = {c["column"]: c for c in strategy["columns"]}
    assert by_name["customer_id"]["action"] == "drop_row"


def test_iqr_outlier_execution_guarded_in_execute_strategy(
        traps, traps_understanding):
    df, _ = traps
    df = df.copy()
    before = len(df)
    out_df, log = execute_strategy(
        df, {"columns": [], "deduplicate": False,
             "outliers": {"customer_id": "drop"}}, traps_understanding)
    assert len(out_df) == before                         # never drops on IDs
    ops = [str(op.get("op", "")) for op in log]
    assert not any(op in ("iqr_outlier_flag", "iqr_outlier_drop")
                   for op in ops)                        # 3.1 executor gate
    assert any("skipped_not_measure" in str(op.get("detail", ""))
               for op in log)


# ---------------------------------------------------------------------------
# Stage 8 — 8.1 QA hard-fail
# ---------------------------------------------------------------------------


def _traps_run(tmp_path: Path, kpis, insights=None, understanding=None):
    run_dir = tmp_path / "run"
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "outputs" / "kpis.json").write_text(
        json.dumps({"kpis": kpis}), encoding="utf-8")
    (run_dir / "outputs" / "insights.json").write_text(
        json.dumps({"insights": insights or [], "recommendations": []}),
        encoding="utf-8")
    (run_dir / "metadata" / "dataset_understanding.json").write_text(
        json.dumps(understanding or {}), encoding="utf-8")
    return run_dir


def test_qa_semantic_relevance_hard_fail(tmp_path, traps_understanding):
    kpis = [
        {"kpi_id": "KPI-1", "name": "Mean phone_number",
         "operation": {"function": "mean", "column": "phone_number"},
         "value": 5.0},
        {"kpi_id": "KPI-2", "name": "phone_number count",
         "operation": {"function": "count", "column": "phone_number"},
         "value": 32},
        {"kpi_id": "KPI-3", "name": "Total revenue",
         "operation": {"function": "sum", "column": "revenue"},
         "value": 9000.0},
    ]
    run_dir = _traps_run(tmp_path, kpis,
                         understanding=traps_understanding.model_dump())
    checks = check_semantic_relevance(run_dir)
    assert any(c.check == "semantic_identifier_reference"
               and "phone_number" in c.message
               and c.severity == "critical" for c in checks)
    assert not any("KPI-2" in c.message for c in checks)   # count is fine
    assert not any("KPI-3" in c.message for c in checks)
    assert decide_verdict(checks) == "NEEDS_REVISION"      # hard fail


def test_qa_semantic_relevance_via_insight(tmp_path, traps_understanding):
    kpis = [{"kpi_id": "KPI-1", "name": "Average customer_id",
             "operation": {"function": "mean", "column": "customer_id"},
             "value": 500.0}]
    insights = [{"insight_id": "INS-1", "claim_type": "DESCRIPTIVE",
                 "title": "x", "description": "y", "confidence": "high",
                 "evidence_ids": ["E1"], "required_evidence": ["aggregate"],
                 "related_kpis": ["KPI-1"]}]
    run_dir = _traps_run(tmp_path, kpis, insights=insights,
                         understanding=traps_understanding.model_dump())
    checks = check_semantic_relevance(run_dir)
    assert any("INS-1" in c.message and c.severity == "critical"
               for c in checks)


def test_qa_clean_golden_run_passes_semantic_check(tmp_path,
                                                   traps_understanding):
    """The golden dataset end-to-end: legitimate KPIs never trip 8.1."""
    kpis = [
        {"kpi_id": "KPI-1", "name": "Total revenue",
         "operation": {"function": "sum", "column": "revenue"},
         "value": 9000.0},
        {"kpi_id": "KPI-2", "name": "phone_number count",
         "operation": {"function": "count", "column": "phone_number"},
         "value": 32},
    ]
    run_dir = _traps_run(tmp_path, kpis,
                         understanding=traps_understanding.model_dump())
    assert check_semantic_relevance(run_dir) == []


# ---------------------------------------------------------------------------
# 5.5 — report KPI ranking by business relevance
# ---------------------------------------------------------------------------


def test_report_kpi_ranking_prefers_primary_measure():
    kpis = [
        {"kpi_id": "A", "name": "Average quality_metric",
         "operation": {"function": "mean", "column": "quality_metric"},
         "value": 99999.0},
        {"kpi_id": "B", "name": "Total revenue",
         "operation": {"function": "sum", "column": "revenue"},
         "value": 1200.0},
    ]
    assert _kpi_relevance(kpis[1]) > _kpi_relevance(kpis[0])
    html = render_kpis(kpis)
    assert html.index("Total revenue") < html.index("quality_metric")


# ---------------------------------------------------------------------------
# 6.1/6.2 — insight diversity + insight-level identifier gate
# ---------------------------------------------------------------------------


def test_insight_diversify_round_robin():
    from agents.insight_agent import _diversify_insights
    from shared.schemas import Insight

    def make(iid, claim_type):
        return Insight(insight_id=iid, claim_type=claim_type, title=iid,
                       description="d", confidence="high")

    insights = [make("D1", "DESCRIPTIVE"), make("D2", "DESCRIPTIVE"),
                make("C1", "CORRELATIONAL"), make("T1", "DESCRIPTIVE")]
    order = [i.insight_id for i in _diversify_insights(insights)]
    assert order[0] == "D1" and order[1] == "C1"        # one per family
    assert order[2] == "D2" and order[3] == "T1"        # then repeats
    assert _diversify_insights([]) == []