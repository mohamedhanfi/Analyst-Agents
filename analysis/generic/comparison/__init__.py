"""Comparison suite (§2.5) — 2-group (t-test, Mann-Whitney U) and 3+ groups
(ANOVA, Kruskal-Wallis + post-hoc), plus categorical chi-square / Cramér's V.

Every comparison is deterministic, scipy-backed and evidence-minted.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from analysis.evidence import EvidenceRegistry
from analysis.generic._helpers import (
    categorical_columns,
    measure_columns,
)
from shared.schemas import DatasetUnderstanding, StatisticalResult

MAX_POSTHOC_GROUPS = 8


def _groups(df: pd.DataFrame, measure: str,
            dimension: str) -> List[pd.Series]:
    out = []
    for value in sorted(df[dimension].dropna().unique()):
        member = pd.to_numeric(
            df.loc[df[dimension] == value, measure], errors="coerce")
        member = member.dropna()
        if len(member) >= 2:
            member.name = str(value)
            out.append(member)
    return out


def _cohens_d(a: pd.Series, b: pd.Series) -> float | None:
    na, nb = len(a), len(b)
    if na + nb < 3:
        return None
    pooled = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2))
    if pooled == 0:
        return None
    return float((a.mean() - b.mean()) / pooled)


def _eta_squared(groups: List[pd.Series]) -> float | None:
    all_values = np.concatenate([g.values for g in groups])
    grand = float(np.mean(all_values))
    ss_between = sum(float(len(g)) * (float(g.mean()) - grand) ** 2
                     for g in groups)
    ss_total = float(np.sum((all_values - grand) ** 2))
    if ss_total == 0:
        return None
    return float(ss_between / ss_total)


def _next_index(index: int, results: List[StatisticalResult]) -> int:
    return index + sum(1 for r in results if r.category == "comparison")


def _group_results(df: pd.DataFrame, registry: EvidenceRegistry,
                   measure: str, dimension: str, groups: List[pd.Series],
                   index: int) -> Tuple[List[StatisticalResult], int]:
    results: List[StatisticalResult] = []
    n_groups = len(groups)
    if n_groups == 2:
        a, b = groups
        t, p = stats.ttest_ind(a, b, equal_var=False)
        index += 1
        eid = registry.add_value(float(t), aggregation="comparison",
                                 comparison="t_test")
        results.append(StatisticalResult(
            test_id=f"ST-CMP-{index:03d}", category="comparison",
            test_name="t_test", variables=[measure, dimension],
            statistic=float(t), p_value=float(p),
            effect_size=_cohens_d(a, b),
            n=int(len(a) + len(b)), evidence_id=eid,
            extra={"test": "welch"}))
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        index += 1
        eid = registry.add_value(float(u), aggregation="comparison",
                                 comparison="mann_whitney")
        results.append(StatisticalResult(
            test_id=f"ST-CMP-{index:03d}", category="comparison",
            test_name="mann_whitney", variables=[measure, dimension],
            statistic=float(u), p_value=float(p), n=int(len(a) + len(b)),
            evidence_id=eid, extra={}))
        return results, index

    f, p = stats.f_oneway(*groups)
    index += 1
    eid = registry.add_value(float(f), aggregation="comparison",
                             comparison="anova")
    results.append(StatisticalResult(
        test_id=f"ST-CMP-{index:03d}", category="comparison",
        test_name="anova", variables=[measure, dimension],
        statistic=float(f), p_value=float(p),
        effect_size=_eta_squared(groups),
        n=int(sum(len(g) for g in groups)), evidence_id=eid,
        extra={"groups": [str(g.name) for g in groups]}))
    h, p = stats.kruskal(*groups)
    index += 1
    eid = registry.add_value(float(h), aggregation="comparison",
                             comparison="kruskal")
    results.append(StatisticalResult(
        test_id=f"ST-CMP-{index:03d}", category="comparison",
        test_name="kruskal_wallis", variables=[measure, dimension],
        statistic=float(h), p_value=float(p),
        n=int(sum(len(g) for g in groups)), evidence_id=eid,
        extra={"groups": [str(g.name) for g in groups]}))

    if n_groups <= MAX_POSTHOC_GROUPS:
        posthoc = {}
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                _, pp = stats.mannwhitneyu(groups[i], groups[j],
                                           alternative="two-sided")
                pair = f"{groups[i].name} vs {groups[j].name}"
                posthoc[pair] = round(float(pp), 6)
        results[-1].extra["posthoc_bonferroni"] = posthoc
    return results, index


def _categorical_results(df: pd.DataFrame, registry: EvidenceRegistry,
                         col_a: str, col_b: str,
                         index: int) -> Tuple[List[StatisticalResult], int]:
    observed = pd.crosstab(df[col_a], df[col_b])
    if observed.shape[0] < 2 or observed.shape[1] < 2:
        return [], index
    chi2, p, dof, _ = stats.chi2_contingency(observed, correction=False)
    n = int(observed.values.sum())
    cramers_denom = n * (min(observed.shape) - 1)
    v = float(np.sqrt(chi2 / cramers_denom)) if cramers_denom else 0.0
    index += 1
    eid = registry.add_value(float(chi2), aggregation="comparison",
                             comparison="chi2")
    results = [StatisticalResult(
        test_id=f"ST-CMP-{index:03d}", category="comparison",
        test_name="chi2", variables=[col_a, col_b],
        statistic=float(chi2), p_value=float(p), effect_size=v,
        n=n, evidence_id=eid,
        extra={"dof": int(dof)})]
    index += 1
    eid = registry.add_value(v, aggregation="comparison",
                             comparison="cramers_v")
    results.append(StatisticalResult(
        test_id=f"ST-CMP-{index:03d}", category="comparison",
        test_name="cramers_v", variables=[col_a, col_b],
        statistic=v, p_value=float(p), n=n, evidence_id=eid,
        extra={"chi2": float(chi2)}))
    return results, index


def run_comparison(df: pd.DataFrame,
                   understanding: DatasetUnderstanding,
                   registry: EvidenceRegistry,
                   index: int = 0) -> List[StatisticalResult]:
    results: List[StatisticalResult] = []
    measures = measure_columns(understanding, df)
    dimensions = categorical_columns(understanding, df)

    for dimension in dimensions:
        for measure in measures:
            groups = _groups(df, measure, dimension)
            if len(groups) < 2:
                continue
            index = _next_index(index, results)
            found, index = _group_results(df, registry, measure, dimension,
                                          groups, index)
            results.extend(found)

    for i, col_a in enumerate(dimensions):
        for col_b in dimensions[i + 1:]:
            index = _next_index(index, results)
            found, index = _categorical_results(df, registry, col_a, col_b,
                                                index)
            results.extend(found)

    return results
