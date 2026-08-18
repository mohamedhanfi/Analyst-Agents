"""Agent tests — plan viability and LLM JSON structure.

Validates that analysis plans conform to the DSL whitelist, have
required fields, and don't contain forbidden claim types.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shared.core.profiler import DataProfiler
from shared.core.understanding import default_plan
from shared.dsl_validator import WHITELIST, validate_plan
from shared.schemas import AnalysisPlan

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _make_plan() -> AnalysisPlan:
    """Build a real analysis plan from the sales_small fixture."""
    csv_path = FIXTURES / "sales_small.csv"
    df = pd.read_csv(csv_path)
    profiler = DataProfiler()
    profile = profiler.profile(df, file_name="sales_small.csv",
                               file_hash="sha256:test")
    return default_plan(profile)


class TestPlanOperationsWhitelist:
    """All KPI operations must be in the DSL whitelist."""

    def test_all_ops_in_whitelist(self) -> None:
        plan = _make_plan()
        for kpi in plan.candidate_kpis:
            op = kpi.operation
            assert op.function in WHITELIST, (
                f"KPI {kpi.kpi_id} uses non-whitelisted function: {op.function}"
            )

    def test_whitelist_is_nonempty(self) -> None:
        assert len(WHITELIST) > 0


class TestPlanRequiredFields:
    """Every KPI must have kpi_id, name, and operation."""

    def test_kpi_has_id(self) -> None:
        plan = _make_plan()
        for kpi in plan.candidate_kpis:
            assert kpi.kpi_id, f"KPI missing kpi_id"

    def test_kpi_has_name(self) -> None:
        plan = _make_plan()
        for kpi in plan.candidate_kpis:
            assert kpi.name, f"KPI {kpi.kpi_id} missing name"

    def test_kpi_has_operation(self) -> None:
        plan = _make_plan()
        for kpi in plan.candidate_kpis:
            assert kpi.operation is not None, f"KPI {kpi.kpi_id} missing operation"

    def test_plan_has_kpis(self) -> None:
        plan = _make_plan()
        assert len(plan.candidate_kpis) > 0


class TestPlanNoCausalClaims:
    """Plan must not contain CAUSAL claim types."""

    def test_no_causal_in_plan(self) -> None:
        plan = _make_plan()
        errors = validate_plan(plan)
        assert len(errors) == 0, f"Plan validation errors: {errors}"


class TestPlanGroupByValid:
    """group_by references must be valid columns or None."""

    def test_group_by_none_or_list(self) -> None:
        plan = _make_plan()
        for kpi in plan.candidate_kpis:
            gb = kpi.operation.group_by
            assert gb is None or isinstance(gb, list), (
                f"KPI {kpi.kpi_id} has invalid group_by: {gb}"
            )


class TestPlanFilterSyntax:
    """Filter expressions must be parseable (not raw garbage)."""

    def test_filter_none_or_dict(self) -> None:
        plan = _make_plan()
        for kpi in plan.candidate_kpis:
            filt = kpi.operation.filter
            assert filt is None or isinstance(filt, dict), (
                f"KPI {kpi.kpi_id} has invalid filter: {filt}"
            )


class TestPlanStatisticalTestsValid:
    """Requested statistical tests must be in the allowed set."""

    VALID_TESTS = {
        "pearson", "spearman", "kendall", "ttest_ind", "ttest_rel",
        "anova", "kruskal", "mannwhitneyu", "chi2", "normality",
        "descriptive", "comparison", "correlation", "distribution", "trend",
    }

    def test_stat_tests_in_allowed_set(self) -> None:
        plan = _make_plan()
        for test in plan.statistical_tests:
            assert test in self.VALID_TESTS or test.startswith("st_"), (
                f"Invalid statistical test: {test}"
            )
