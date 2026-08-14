"""Unit tests for analysis/generic/correlation — §2.5 correlation suite."""
from __future__ import annotations

import pandas as pd
import pytest
from scipy import stats

from analysis.evidence import EvidenceRegistry
from analysis.generic.correlation import run_correlation
from tests.unit.conftest import SALES


def _registry():
    return EvidenceRegistry(run_id="test")


def test_correlation_pair_order(make_understanding):
    results = run_correlation(SALES, make_understanding(SALES), _registry())
    assert [r.variables for r in results] == [["revenue", "quantity"],
                                              ["revenue", "quantity"]]
    assert [r.test_name for r in results] == ["pearson", "spearman"]


def test_correlation_matches_scipy(make_understanding):
    results = run_correlation(SALES, make_understanding(SALES), _registry())
    pearson = results[0]
    spearman = results[1]
    r, p = stats.pearsonr(SALES["revenue"], SALES["quantity"])
    rho, sp = stats.spearmanr(SALES["revenue"], SALES["quantity"])
    assert pearson.statistic == pytest.approx(r)
    assert pearson.p_value == pytest.approx(p)
    assert pearson.n == len(SALES)
    assert pearson.effect_size == pytest.approx(r ** 2)
    assert spearman.statistic == pytest.approx(rho)
    assert spearman.p_value == pytest.approx(sp)


def test_correlation_ci_bounds_fisher(make_understanding):
    results = run_correlation(SALES, make_understanding(SALES), _registry())
    pearson = results[0]
    assert pearson.ci_low is not None and pearson.ci_high is not None
    assert pearson.ci_low < pearson.statistic < pearson.ci_high


def test_correlation_mints_two_evidences(make_understanding):
    registry = _registry()
    results = run_correlation(SALES, make_understanding(SALES), registry)
    assert len(results) == 2
    assert len(registry) == 2
    for result in results:
        assert registry.get(result.evidence_id) is not None


def test_correlation_skips_constant_column(make_understanding):
    frame = SALES.copy()
    frame["revenue"] = 5.0
    results = run_correlation(frame, make_understanding(frame), _registry())
    assert results == []


def test_correlation_skips_insufficient_pairs(make_understanding):
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    understanding = make_understanding(frame, measures=("a", "b"),
                                       temporal=(), dimensions=())
    results = run_correlation(frame, understanding, _registry())
    assert results == []


def test_correlation_test_ids_sequential(make_understanding):
    results = run_correlation(SALES, make_understanding(SALES), _registry())
    assert [r.test_id for r in results] == ["ST-CORR-001", "ST-CORR-002"]
