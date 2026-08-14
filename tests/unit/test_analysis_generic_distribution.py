"""Unit tests for analysis/generic/distribution — §2.5 histograms."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.evidence import EvidenceRegistry
from analysis.generic.distribution import (
    freedman_diaconis_bins,
    run_distribution,
)
from tests.unit.conftest import SALES


def _registry():
    return EvidenceRegistry(run_id="test")


def test_fd_bins_matches_math():
    series = SALES["revenue"]
    n = len(series)
    iqr = float(series.quantile(0.75)) - float(series.quantile(0.25))
    width = 2.0 * iqr / (n ** (1.0 / 3.0))
    expected = int(np.ceil((series.max() - series.min()) / width))
    expected = int(np.clip(expected, 2, 50))
    count, returned_width = freedman_diaconis_bins(series)
    assert count == expected
    assert returned_width == pytest.approx((series.max() - series.min())
                                           / count)


def test_fd_bins_degenerate_iqr_uses_sqrt_fallback():
    series = pd.Series([1.0] * 19 + [100.0])
    count, width = freedman_diaconis_bins(series)
    assert count == int(np.clip(int(np.sqrt(len(series))), 2, 50))
    assert width > 0


def test_fd_bins_constant_series_single_bin():
    series = pd.Series([5.0] * 20)
    count, width = freedman_diaconis_bins(series)
    assert count == 1
    assert width == 0.0


def test_fd_bins_single_value():
    series = pd.Series([1.0])
    count, _ = freedman_diaconis_bins(series)
    assert count == 1


def test_distribution_histogram_counts_sum_to_n(make_understanding):
    result = run_distribution(
        SALES, make_understanding(SALES), _registry())[0]
    assert result.test_name == "histogram"
    assert result.n == len(SALES)
    assert sum(result.extra["counts"]) == len(SALES)
    assert result.extra["bin_count"] == len(result.extra["counts"])
    assert result.extra["bin_edges"][0] == pytest.approx(SALES["revenue"].min())
    assert result.extra["bin_edges"][-1] == pytest.approx(SALES["revenue"].max())


def test_distribution_mints_evidence(make_understanding):
    registry = _registry()
    results = run_distribution(SALES, make_understanding(SALES), registry)
    assert len(results) == 2
    assert len(registry) == 2
    for result in results:
        assert registry.get(result.evidence_id) is not None


def test_distribution_test_ids(make_understanding):
    results = run_distribution(SALES, make_understanding(SALES), _registry())
    assert [r.test_id for r in results] == ["ST-DIST-001", "ST-DIST-002"]
