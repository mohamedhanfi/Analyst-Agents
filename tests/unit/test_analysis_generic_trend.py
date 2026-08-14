"""Unit tests for analysis/generic/trend — §2.5 YoY/MoM/WoW + rolling + seasonality."""
from __future__ import annotations

import pandas as pd
import pytest

from analysis.evidence import EvidenceRegistry
from analysis.generic.trend import run_trend
from tests.unit.conftest import SALES


def _registry():
    return EvidenceRegistry(run_id="test")


def test_trend_produces_growth_per_period(make_understanding):
    results = run_trend(SALES, make_understanding(SALES), _registry())
    growth = [r for r in results if r.test_name.startswith("growth")]
    assert {r.test_name for r in growth} == {"growth_MoM", "growth_YoY",
                                             "growth_WoW"}
    for result in growth:
        assert result.category == "trend"
        assert result.variables[1] == "date"
        assert result.variables[0] in ("revenue", "quantity")
        assert result.n > 0


def test_trend_mom_matches_dsl(make_understanding):
    from analysis.dsl_executor import execute_operation
    from shared.schemas import DslOperation
    results = run_trend(SALES, make_understanding(SALES), _registry())
    mom = next(r for r in results if r.test_name == "growth_MoM")
    op = DslOperation(function="growth", column="revenue",
                      over_column="date", period="MoM")
    reference = execute_operation(SALES, op).growth_series
    assert mom.extra["series"] == reference


def test_trend_rolling_mean_matches_pandas(make_understanding):
    results = run_trend(SALES, make_understanding(SALES), _registry())
    rolling = next(r for r in results if r.test_name == "rolling_mean")
    frame = SALES.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    frame["_period"] = dates.dt.to_period("M")
    totals = (frame.groupby("_period")["revenue"]
              .apply(lambda s: float(pd.to_numeric(s, errors="coerce").sum())))
    totals = totals.sort_index().rolling(3, min_periods=1).mean()
    assert rolling.extra["window"] == 3
    assert [s["value"] for s in rolling.extra["series"]] == \
        pytest.approx([float(v) for v in totals.values])


def test_trend_seasonality_present(make_understanding):
    results = run_trend(SALES, make_understanding(SALES), _registry())
    seasonal = next(r for r in results if r.test_name == "seasonality")
    assert seasonal.statistic is not None
    assert seasonal.extra["verdict"] in ("seasonal", "not_seasonal")
    assert seasonal.n == 6  # six distinct months in SALES


def test_trend_skips_short_series(make_understanding):
    short = pd.DataFrame({
        "date": ["2024-01-15", "2024-02-15"],
        "revenue": [100, 200],
    })
    results = run_trend(short, make_understanding(short, measures=("revenue",)),
                        _registry())
    growth = [r for r in results if r.test_name.startswith("growth")]
    assert all(r.test_name not in ("rolling_mean", "seasonality")
               for r in results)
    assert growth  # MoM has one row


def test_trend_mints_evidence_per_result(make_understanding):
    registry = _registry()
    results = run_trend(SALES, make_understanding(SALES), registry)
    assert len(registry) == len(results)
    for result in results:
        assert registry.get(result.evidence_id) is not None
