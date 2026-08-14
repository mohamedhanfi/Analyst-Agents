"""Correlation suite (§2.5) — Pearson + p-value + CI + effect size + n; Spearman.

Every pair of measure columns, both methods. CI via Fisher z-transform; effect
size is r^2 (variance explained).
"""
from __future__ import annotations

from typing import List

import pandas as pd
from scipy import stats

from analysis.evidence import EvidenceRegistry
from analysis.generic._helpers import (
    fisher_ci,
    measure_columns,
    mean_variance_variant,
    numeric_series,
)
from shared.schemas import DatasetUnderstanding, StatisticalResult


def _pearson(a: pd.Series, b: pd.Series):
    r, p = stats.pearsonr(a, b)
    return float(r), float(p)


def _spearman(a: pd.Series, b: pd.Series):
    rho, p = stats.spearmanr(a, b)
    return float(rho), float(p)


def run_correlation(df: pd.DataFrame,
                    understanding: DatasetUnderstanding,
                    registry: EvidenceRegistry,
                    index: int = 0) -> List[StatisticalResult]:
    results: List[StatisticalResult] = []
    measures = measure_columns(understanding, df)
    for i, col_a in enumerate(measures):
        for col_b in measures[i + 1:]:
            a = numeric_series(df, col_a)
            b = numeric_series(df, col_b)
            valid = a.index.intersection(b.index)
            a, b = a[valid], b[valid]
            n = len(a)
            if n < 3 or not mean_variance_variant(a) \
                    or not mean_variance_variant(b):
                continue
            for method, (statistic, p_value), name in (
                    ("pearson", _pearson(a, b), "pearson"),
                    ("spearman", _spearman(a, b), "spearman")):
                index += 1
                ci_low, ci_high = fisher_ci(statistic, n)
                effect = statistic ** 2
                eid = registry.add_value(statistic, aggregation="correlation",
                                         comparison=name)
                results.append(StatisticalResult(
                    test_id=f"ST-CORR-{index:03d}",
                    category="correlation",
                    test_name=name,
                    variables=[col_a, col_b],
                    statistic=statistic,
                    p_value=p_value,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    effect_size=effect,
                    n=n,
                    evidence_id=eid,
                    extra={"method": method},
                ))
    return results
