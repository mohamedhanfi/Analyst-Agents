"""Descriptive statistics (§2.5) — mean/median/std/quantiles/IQR/skew/kurtosis.

One StatisticalResult per measure column, every value evidence-minted.
"""
from __future__ import annotations

from typing import List

import pandas as pd

from analysis.evidence import EvidenceRegistry
from analysis.generic._helpers import (
    measure_columns,
    numeric_series,
)
from shared.schemas import DatasetUnderstanding, StatisticalResult


def run_descriptive(df: pd.DataFrame,
                    understanding: DatasetUnderstanding,
                    registry: EvidenceRegistry,
                    index: int = 0) -> List[StatisticalResult]:
    results: List[StatisticalResult] = []
    for column in measure_columns(understanding, df):
        series = numeric_series(df, column)
        if series.empty:
            continue
        mean = float(series.mean())
        median = float(series.median())
        std = float(series.std())
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        index += 1
        eid = registry.add_value(mean, aggregation="descriptive",
                                 comparison="summary")
        results.append(StatisticalResult(
            test_id=f"ST-DESC-{index:03d}",
            category="descriptive",
            test_name="descriptive_summary",
            variables=[column],
            statistic=mean,
            n=int(len(series)),
            evidence_id=eid,
            extra={
                "mean": mean, "median": median, "std": std,
                "q1": q1, "q3": q3, "iqr": q3 - q1,
                "skew": float(series.skew()),
                "kurtosis": float(series.kurt()),
                "min": float(series.min()), "max": float(series.max()),
                "count": int(len(series)),
            },
        ))
    return results
