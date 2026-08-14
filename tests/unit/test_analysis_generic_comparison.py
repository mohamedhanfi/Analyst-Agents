"""Unit tests for analysis/generic/comparison — §2.5 group comparisons."""
from __future__ import annotations

import pandas as pd
import pytest
from scipy import stats

from analysis.evidence import EvidenceRegistry
from analysis.generic.comparison import run_comparison
from tests.unit.conftest import GROUPS


def _registry():
    return EvidenceRegistry(run_id="test")


def _group_understanding(make_understanding, frame):
    return make_understanding(frame, measures=("revenue",),
                              temporal=(), dimensions=("category",))


def test_two_group_t_test_matches_scipy(make_understanding):
    frame = GROUPS
    results = run_comparison(frame, _group_understanding(make_understanding,
                                                         frame), _registry())
    names = [r.test_name for r in results]
    assert "t_test" in names and "mann_whitney" in names
    t_result = results[names.index("t_test")]
    a = frame.loc[frame["category"] == "X", "revenue"]
    b = frame.loc[frame["category"] == "Y", "revenue"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    assert t_result.statistic == pytest.approx(t)
    assert t_result.p_value == pytest.approx(p)
    assert t_result.n == len(frame)
    assert t_result.extra["test"] == "welch"


def test_two_group_cohens_d_sign(make_understanding):
    frame = GROUPS
    results = run_comparison(frame, _group_understanding(make_understanding,
                                                         frame), _registry())
    t_result = next(r for r in results if r.test_name == "t_test")
    assert t_result.effect_size is not None
    assert t_result.effect_size < 0  # Y (31.5) > X (11.5)


def test_two_group_mann_whitney_matches_scipy(make_understanding):
    frame = GROUPS
    results = run_comparison(frame, _group_understanding(make_understanding,
                                                         frame), _registry())
    u_result = next(r for r in results if r.test_name == "mann_whitney")
    a = frame.loc[frame["category"] == "X", "revenue"]
    b = frame.loc[frame["category"] == "Y", "revenue"]
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    assert u_result.statistic == pytest.approx(u)
    assert u_result.p_value == pytest.approx(p)


def _three_group_frame():
    return pd.DataFrame({
        "category": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
        "revenue": [10, 12, 11, 13, 12, 11,
                    20, 22, 21, 23, 22, 21,
                    30, 32, 31, 33, 32, 31],
    })


def test_three_group_anova_matches_scipy(make_understanding):
    frame = _three_group_frame()
    results = run_comparison(frame, _group_understanding(make_understanding,
                                                         frame), _registry())
    names = [r.test_name for r in results]
    assert "anova" in names and "kruskal_wallis" in names
    anova = results[names.index("anova")]
    groups = [g for _, g in frame.groupby("category")["revenue"]]
    f, p = stats.f_oneway(*groups)
    assert anova.statistic == pytest.approx(f)
    assert anova.p_value == pytest.approx(p)
    assert anova.n == len(frame)
    assert set(anova.extra["groups"]) == {"A", "B", "C"}


def test_kruskal_matches_scipy(make_understanding):
    frame = _three_group_frame()
    results = run_comparison(frame, _group_understanding(make_understanding,
                                                         frame), _registry())
    kruskal = next(r for r in results if r.test_name == "kruskal_wallis")
    groups = [g for _, g in frame.groupby("category")["revenue"]]
    h, p = stats.kruskal(*groups)
    assert kruskal.statistic == pytest.approx(h)
    assert kruskal.p_value == pytest.approx(p)


def test_anova_posthoc_present(make_understanding):
    frame = _three_group_frame()
    results = run_comparison(frame, _group_understanding(make_understanding,
                                                         frame), _registry())
    kruskal = next(r for r in results if r.test_name == "kruskal_wallis")
    posthoc = kruskal.extra["posthoc_bonferroni"]
    assert set(posthoc) == {"A vs B", "A vs C", "B vs C"}
    assert all(0 <= v <= 1 for v in posthoc.values())


def _categorical_frame(perfect: bool):
    segments = (["H", "H", "L", "L", "H", "L"]
                if perfect else ["H", "L", "H", "L", "H", "L"])
    return pd.DataFrame({
        "region": ["N", "N", "S", "S", "N", "S"] * 5,
        "segment": segments * 5,
    })


def test_chi2_and_cramers_v(make_understanding):
    frame = _categorical_frame(perfect=False)
    understanding = make_understanding(frame, measures=(), temporal=(),
                                       dimensions=("region", "segment"))
    results = run_comparison(frame, understanding, _registry())
    names = [r.test_name for r in results]
    assert "chi2" in names and "cramers_v" in names
    chi2_result = results[names.index("chi2")]
    observed = pd.crosstab(frame["region"], frame["segment"])
    chi2, p, dof, _ = stats.chi2_contingency(observed, correction=False)
    assert chi2_result.statistic == pytest.approx(chi2)
    assert chi2_result.p_value == pytest.approx(p)
    assert chi2_result.extra["dof"] == dof
    cramers = results[names.index("cramers_v")]
    assert 0 <= cramers.statistic <= 1
    assert cramers.n == len(frame)


def test_perfect_association_cramers_v_equals_one(make_understanding):
    frame = _categorical_frame(perfect=True)
    understanding = make_understanding(frame, measures=(), temporal=(),
                                       dimensions=("region", "segment"))
    results = run_comparison(frame, understanding, _registry())
    cramers = next(r for r in results if r.test_name == "cramers_v")
    assert cramers.statistic == pytest.approx(1.0)


def test_single_group_skipped(make_understanding):
    frame = pd.DataFrame({"category": ["X"] * 6, "revenue": [1, 2, 3, 4, 5, 6]})
    understanding = _group_understanding(make_understanding, frame)
    assert run_comparison(frame, understanding, _registry()) == []


def test_mints_evidence(make_understanding):
    registry = _registry()
    frame = GROUPS
    results = run_comparison(frame, _group_understanding(make_understanding,
                                                         frame), registry)
    assert len(registry) == len(results)
    for result in results:
        assert registry.get(result.evidence_id) is not None
