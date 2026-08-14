"""Statistical suite dispatcher (§2.5).

Maps the whitelisted test categories from AnalysisPlan.statistical_tests to the
generic module runners. ``anova`` maps to the comparison module (t-test /
Mann-Whitney / ANOVA / Kruskal-Wallis / chi-square / Cramér's V). The empty
plan defaults to descriptive + correlation + trend.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from analysis.evidence import EvidenceRegistry
from analysis.generic.comparison import run_comparison
from analysis.generic.correlation import run_correlation
from analysis.generic.descriptive import run_descriptive
from analysis.generic.distribution import run_distribution
from analysis.generic.trend import run_trend
from shared.schemas import DatasetUnderstanding, StatisticalResult

DEFAULT_TESTS = ("descriptive", "correlation", "trend")

RUNNERS: Dict[str, callable] = {
    "descriptive": run_descriptive,
    "correlation": run_correlation,
    "distribution": run_distribution,
    "trend": run_trend,
    "anova": run_comparison,
}


def run_statistical_suite(
    df: pd.DataFrame,
    understanding: DatasetUnderstanding,
    registry: EvidenceRegistry,
    tests: List[str] | None = None,
) -> List[StatisticalResult]:
    """Execute the requested statistical tests against ``df``.

    ``tests`` comes from AnalysisPlan.statistical_tests; unknown categories are
    ignored, so the module set stays a pure whitelist.
    """
    requested = tests or list(DEFAULT_TESTS)
    results: List[StatisticalResult] = []
    index = 0
    for category in requested:
        runner = RUNNERS.get(category)
        if runner is None:
            continue
        found = runner(df, understanding, registry, index=index)
        results.extend(found)
        index += len(found)
    return results
