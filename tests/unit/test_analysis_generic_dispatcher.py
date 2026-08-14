"""Unit tests for analysis/generic/__init__ — the statistical suite dispatcher."""
from __future__ import annotations

from analysis.evidence import EvidenceRegistry
from analysis.generic import run_statistical_suite
from tests.unit.conftest import GROUPS, SALES


def _registry():
    return EvidenceRegistry(run_id="test")


def test_default_suite_runs_three_categories(make_understanding):
    results = run_statistical_suite(SALES, make_understanding(SALES),
                                    _registry())
    categories = {r.category for r in results}
    assert categories == {"descriptive", "correlation", "trend"}


def test_explicit_tests_respected(make_understanding):
    results = run_statistical_suite(SALES, make_understanding(SALES),
                                    _registry(),
                                    tests=["descriptive"])
    assert {r.category for r in results} == {"descriptive"}


def test_anova_maps_to_comparison(make_understanding):
    results = run_statistical_suite(GROUPS, make_understanding(GROUPS),
                                    _registry(), tests=["anova"])
    assert {r.category for r in results} == {"comparison"}
    assert "t_test" in {r.test_name for r in results}


def test_unknown_category_ignored(make_understanding):
    results = run_statistical_suite(SALES, make_understanding(SALES),
                                    _registry(),
                                    tests=["bogus", "descriptive"])
    assert {r.category for r in results} == {"descriptive"}


def test_empty_tests_list_falls_back_to_default(make_understanding):
    results = run_statistical_suite(SALES, make_understanding(SALES),
                                    _registry(), tests=[])
    categories = {r.category for r in results}
    assert categories == {"descriptive", "correlation", "trend"}


def test_suite_ids_unique_across_categories(make_understanding):
    results = run_statistical_suite(SALES, make_understanding(SALES),
                                    _registry(),
                                    tests=["descriptive", "correlation"])
    ids = [r.test_id for r in results]
    assert len(ids) == len(set(ids))


def test_suite_mints_evidence(make_understanding):
    registry = _registry()
    results = run_statistical_suite(SALES, make_understanding(SALES), registry)
    assert len(registry) == len(results)
