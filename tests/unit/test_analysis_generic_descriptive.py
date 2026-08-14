"""Unit tests for analysis/generic/descriptive — §2.5 descriptive summary."""
from __future__ import annotations

import pandas as pd
import pytest

from analysis.evidence import EvidenceRegistry
from analysis.generic.descriptive import run_descriptive
from tests.unit.conftest import SALES


def _registry():
    return EvidenceRegistry(run_id="test")


def test_descriptive_one_result_per_measure(make_understanding):
    results = run_descriptive(SALES, make_understanding(SALES), _registry())
    assert [r.category for r in results] == ["descriptive"] * 2
    assert [r.variables[0] for r in results] == ["revenue", "quantity"]


def test_descriptive_matches_pandas(make_understanding):
    result = run_descriptive(
        SALES, make_understanding(SALES), _registry())[0]
    extra = result.extra
    series = SALES["revenue"]
    assert result.statistic == pytest.approx(series.mean())
    assert extra["mean"] == pytest.approx(series.mean())
    assert extra["median"] == pytest.approx(series.median())
    assert extra["std"] == pytest.approx(series.std())
    assert extra["q1"] == pytest.approx(series.quantile(0.25))
    assert extra["q3"] == pytest.approx(series.quantile(0.75))
    assert extra["iqr"] == pytest.approx(
        series.quantile(0.75) - series.quantile(0.25))
    assert extra["skew"] == pytest.approx(series.skew())
    assert extra["kurtosis"] == pytest.approx(series.kurt())
    assert extra["min"] == pytest.approx(series.min())
    assert extra["max"] == pytest.approx(series.max())
    assert result.n == len(SALES)


def test_descriptive_mints_evidence(make_understanding):
    registry = _registry()
    results = run_descriptive(SALES, make_understanding(SALES), registry)
    assert len(results) == 2
    for result in results:
        assert registry.get(result.evidence_id) is not None
    assert len(registry) == 2


def test_descriptive_test_ids_sequential(make_understanding):
    results = run_descriptive(SALES, make_understanding(SALES), _registry())
    assert [r.test_id for r in results] == ["ST-DESC-001", "ST-DESC-002"]


def test_descriptive_respects_index_offset(make_understanding):
    results = run_descriptive(SALES, make_understanding(SALES), _registry(),
                              index=10)
    assert [r.test_id for r in results] == ["ST-DESC-011", "ST-DESC-012"]


def test_descriptive_skips_empty_measure(make_understanding):
    frame = SALES.copy()
    frame["revenue"] = None
    results = run_descriptive(frame, make_understanding(frame), _registry())
    assert [r.variables[0] for r in results] == ["quantity"]


def test_descriptive_ignores_non_measure_columns(make_understanding):
    frame = SALES.copy()
    frame["notes"] = ["a", "b", "c", "d", "e", "f"]
    understanding = make_understanding(frame, measures=("revenue",))
    results = run_descriptive(frame, understanding, _registry())
    assert [r.variables[0] for r in results] == ["revenue"]
