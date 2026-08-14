"""Distribution suite (§2.5) — histograms with Freedman–Diaconis bins.

Pure numpy bin computation (drawing happens in stage 5b). Degenerate data
(IQR == 0) falls back to a sqrt bin count.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd

from analysis.evidence import EvidenceRegistry
from analysis.generic._helpers import measure_columns, numeric_series
from shared.schemas import DatasetUnderstanding, StatisticalResult

MIN_BINS = 2
MAX_BINS = 50


def freedman_diaconis_bins(series: pd.Series) -> tuple[int, float]:
    """Return (bin_count, bin_width) via the Freedman–Diaconis rule."""
    n = len(series)
    iqr = float(series.quantile(0.75)) - float(series.quantile(0.25))
    span = float(series.max()) - float(series.min())
    if n < 2 or span <= 0:
        return 1, 0.0
    if iqr <= 0:
        bin_count = max(1, int(math.sqrt(n)))
        return int(np.clip(bin_count, MIN_BINS, MAX_BINS)), span / bin_count
    width = 2.0 * iqr / (n ** (1.0 / 3.0))
    if width <= 0:
        return MIN_BINS, span / MIN_BINS
    bin_count = int(np.ceil(span / width))
    bin_count = int(np.clip(bin_count, MIN_BINS, MAX_BINS))
    return bin_count, span / bin_count


def run_distribution(df: pd.DataFrame,
                     understanding: DatasetUnderstanding,
                     registry: EvidenceRegistry,
                     index: int = 0) -> List[StatisticalResult]:
    results: List[StatisticalResult] = []
    for column in measure_columns(understanding, df):
        series = numeric_series(df, column)
        if series.empty:
            continue
        bin_count, bin_width = freedman_diaconis_bins(series)
        edges = np.linspace(float(series.min()), float(series.max()),
                            bin_count + 1)
        counts = np.histogram(series, bins=edges)[0]
        index += 1
        eid = registry.add_value(bin_count, aggregation="distribution",
                                 comparison="freedman_diaconis")
        results.append(StatisticalResult(
            test_id=f"ST-DIST-{index:03d}",
            category="distribution",
            test_name="histogram",
            variables=[column],
            statistic=None,
            n=int(len(series)),
            evidence_id=eid,
            extra={
                "bin_count": bin_count,
                "bin_width": float(bin_width),
                "bin_edges": [float(e) for e in edges],
                "counts": [int(c) for c in counts],
                "min": float(series.min()), "max": float(series.max()),
            },
        ))
    return results
