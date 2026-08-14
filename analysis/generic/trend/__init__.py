"""Trend suite (§2.5) — YoY/MoM/WoW growth series, rolling mean, seasonality.

Growth series reuse the DSL executor (same aggregation semantics). Rolling
mean + seasonality work on the monthly totals of each measure x temporal pair.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from analysis.dsl_executor import execute_operation
from analysis.evidence import EvidenceRegistry
from analysis.generic._helpers import (
    measure_columns,
    numeric_series,
    temporal_columns,
)
from shared.schemas import DslOperation, DatasetUnderstanding, StatisticalResult

ROLLING_WINDOW = 3


def _monthly_totals(df: pd.DataFrame, measure: str,
                    temporal: str) -> pd.DataFrame:
    dates = pd.to_datetime(df[temporal], errors="coerce")
    frame = df[dates.notna()].copy()
    frame["_period"] = dates[dates.notna()].dt.to_period("M")
    values = pd.to_numeric(frame[measure], errors="coerce")
    totals = frame.groupby("_period")[measure].apply(
        lambda s: float(pd.to_numeric(s, errors="coerce").sum()))
    totals = totals.sort_index()
    return totals


def _growth_series(df: pd.DataFrame, measure: str, temporal: str,
                   period: str) -> List[Dict[str, Any]]:
    op = DslOperation(function="growth", column=measure,
                      over_column=temporal, period=period)
    try:
        result = execute_operation(df, op)
    except Exception:  # noqa: BLE001 -- trend is best-effort
        return []
    return result.growth_series


def _seasonality(totals: pd.Series) -> Dict[str, Any]:
    if len(totals) < 6:
        return {"strength": None, "verdict": "insufficient_data"}
    monthly = [totals.index[i].month for i in range(len(totals))]
    values = totals.values.astype(float)
    total_var = float(np.var(values))
    if total_var <= 0:
        return {"strength": 0.0, "verdict": "not_seasonal"}
    means = {m: np.mean(values[[j for j in range(len(values))
                                if monthly[j] == m]])
             for m in sorted(set(monthly))}
    seasonal_var = float(np.var(list(means.values())))
    strength = seasonal_var / total_var if total_var else 0.0
    return {"strength": float(strength),
            "verdict": "seasonal" if strength > 0.3 else "not_seasonal"}


def run_trend(df: pd.DataFrame,
              understanding: DatasetUnderstanding,
              registry: EvidenceRegistry,
              index: int = 0) -> List[StatisticalResult]:
    results: List[StatisticalResult] = []
    temporals = temporal_columns(understanding, df)
    for measure in measure_columns(understanding, df):
        for temporal in temporals:
            if numeric_series(df, measure).empty:
                continue
            for period in ("MoM", "YoY", "WoW"):
                series = _growth_series(df, measure, temporal, period)
                if not series:
                    continue
                index += 1
                eid = registry.add_value(len(series),
                                         aggregation="trend",
                                         comparison=f"growth_{period}")
                results.append(StatisticalResult(
                    test_id=f"ST-TREND-{index:03d}",
                    category="trend",
                    test_name=f"growth_{period}",
                    variables=[measure, temporal],
                    statistic=None,
                    n=len(series),
                    evidence_id=eid,
                    extra={"period": period, "series": series},
                ))

            totals = _monthly_totals(df, measure, temporal)
            if len(totals) < ROLLING_WINDOW:
                continue
            index += 1
            rolling = totals.rolling(ROLLING_WINDOW, min_periods=1).mean()
            eid = registry.add_value(len(totals), aggregation="trend",
                                     comparison="rolling_mean")
            results.append(StatisticalResult(
                test_id=f"ST-TREND-{index:03d}",
                category="trend",
                test_name="rolling_mean",
                variables=[measure, temporal],
                statistic=None,
                n=int(len(totals)),
                evidence_id=eid,
                extra={"window": ROLLING_WINDOW,
                       "series": [{"period": str(p), "value": float(v)}
                                  for p, v in rolling.items()]},
            ))

            index += 1
            seasonality = _seasonality(totals)
            eid = registry.add_value(seasonality["strength"],
                                     aggregation="trend",
                                     comparison="seasonality")
            results.append(StatisticalResult(
                test_id=f"ST-TREND-{index:03d}",
                category="trend",
                test_name="seasonality",
                variables=[measure, temporal],
                statistic=seasonality["strength"],
                n=int(len(totals)),
                evidence_id=eid,
                extra={"series": [{"period": str(p), "value": float(v)}
                                  for p, v in totals.items()],
                       "verdict": seasonality["verdict"]},
            ))
    return results
