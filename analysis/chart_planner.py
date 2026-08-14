"""Data-shape chart planner (§2.5) — deterministic, no fixed menu.

Inspects each KPI candidate + the numeric/ordinal columns and picks a chart
kind from the ordered rule table. The LLM only decides *which facts deserve a
chart* and may re-rank; the shape is always Python's call. Output is
ChartMetadata entries (one per chart) — the actual drawing happens in stage 5b.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import pandas as pd

from analysis.evidence import EvidenceRegistry
from analysis.generic._helpers import measure_columns
from shared.schemas import (AnalysisPlan, ChartMetadata, DatasetUnderstanding,
                            KpiCandidate)

THIN_THRESHOLD = 10          # rows below this => downgrade + reliability low_n
RULE = {                     # rule-table reference (spec §2.5)
    "rule_1": "rule_1: single dimension, <= 2 values swap -> bar/donut",
    "rule_2": "rule_2: ordered axis (dates/months) with >= 3 points -> line",
    "rule_3": "rule_3: single dimension, 3-12 values -> vertical bar",
    "rule_4": "rule_4: single dimension, 13-50 values -> horizontal bar",
    "rule_5": "rule_5: single dimension, > 50 values -> barh top-15 + '$rest'",
    "rule_6": "rule_6: numeric distribution / skew ask -> histogram",
    "rule_7": "rule_7: 2 numeric measures, asked together -> scatter + trend",
    "rule_8": "rule_8: >= 3 numeric measures -> ranked correlation heatmap",
    "rule_9": "rule_9: share / '% of whole', parts ~ 100% -> doughnut",
}

_SHARE_RE = re.compile(r"share|%\b|\bof\s+whole|\bpart\b", re.IGNORECASE)
_AGGREGATES = {"sum", "mean", "median", "count", "nunique", "min", "max",
               "std", "ratio"}
_TOP_N_ROLLUP = 15


def _thin(df: pd.DataFrame, threshold: int) -> bool:
    return len(df) < threshold


def _register(registry: EvidenceRegistry, chart: ChartMetadata) -> None:
    registry.add(chart.evidence_id, {
        "file_hash": registry.file_hash,
        "sheet": registry.sheet,
        "transformations": registry.transformations,
        "aggregation": "chart",
        "comparison": chart.kind,
        "result": chart.title or chart.reason,
    })


def _downgrade_low_n(chart: ChartMetadata, why: str) -> ChartMetadata:
    """Downgrade a too-thin chart to a simple bar + reliability stamp."""
    chart.kind = "bar"
    chart.reliability = "low_n"
    chart.reason = f"low_n ({why}): {chart.reason}"
    return chart


def _dedupe(candidates: List[ChartMetadata]) -> List[ChartMetadata]:
    seen = set()
    unique: List[ChartMetadata] = []
    for chart in candidates:
        key = (chart.kind, tuple(sorted(chart.columns)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(chart)
    return unique


# ---------------------------------------------------------------------------
# Rule selection
# ---------------------------------------------------------------------------

def _plan_kpi(df: pd.DataFrame, kpi: KpiCandidate,
              registry: EvidenceRegistry, index: int,
              threshold: int) -> Optional[ChartMetadata]:
    op = kpi.operation
    function = op.function
    evidence_id = registry.mint()

    if function == "growth":
        chart = _rule_2_growth(df, kpi, evidence_id, index, threshold)
    elif function == "correlation":
        chart = _rule_7_scatter(df, kpi, op.column_a or op.column,
                                op.column_b, evidence_id, index, threshold)
    elif function in _AGGREGATES and op.group_by:
        chart = _rule_dimension(df, kpi, op.group_by[0], evidence_id, index,
                                threshold)
    else:
        return None                     # headline number, no visual shape

    if chart is not None:
        _register(registry, chart)
    return chart


def _rule_2_growth(df: pd.DataFrame, kpi: KpiCandidate, evidence_id: str,
                   index: int, threshold: int) -> Optional[ChartMetadata]:
    op = kpi.operation
    over = op.over_column
    if not over or over not in df.columns:
        return None
    values = pd.to_datetime(df[over], errors="coerce").dropna()
    if values.empty:
        return None
    points = values.nunique()
    chart = ChartMetadata(
        chart_id=f"CH-{index:03d}", kind="line",
        reason=RULE["rule_2"],
        columns=[over, op.column or ""],
        title=f"{kpi.name} ({op.period or 'MoM'} growth)",
        evidence_id=evidence_id,
    )
    if points < 3:
        return _downgrade_low_n(chart, f"{points} time points < 3")
    if _thin(df, threshold):
        return _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
    return chart


def _rule_7_scatter(df: pd.DataFrame, kpi: KpiCandidate, col_a: str,
                    col_b: str, evidence_id: str, index: int,
                    threshold: int) -> Optional[ChartMetadata]:
    if not col_a or not col_b or col_a not in df.columns \
            or col_b not in df.columns:
        return None
    chart = ChartMetadata(
        chart_id=f"CH-{index:03d}", kind="scatter",
        reason=RULE["rule_7"],
        columns=[col_a, col_b],
        title=f"{kpi.name} ({col_a} x {col_b})",
        evidence_id=evidence_id,
    )
    if _thin(df, threshold):
        return _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
    return chart


def _rule_dimension(df: pd.DataFrame, kpi: KpiCandidate, dimension: str,
                    evidence_id: str, index: int,
                    threshold: int) -> Optional[ChartMetadata]:
    if dimension not in df.columns:
        return None
    values = df[dimension].dropna().nunique()
    is_share = bool(_SHARE_RE.search(kpi.name or ""))

    if values <= 2 and is_share:
        kind, reason = "doughnut", RULE["rule_9"]
    elif values <= 2:
        kind, reason = "bar", RULE["rule_1"]
    elif values <= 12:
        kind, reason = "bar", RULE["rule_3"]
    elif values <= 50:
        kind, reason = "barh", RULE["rule_4"]
    else:
        kind, reason = "barh", f"{RULE['rule_5']} (top-{_TOP_N_ROLLUP} + $rest)"
    chart = ChartMetadata(
        chart_id=f"CH-{index:03d}", kind=kind, reason=reason,
        columns=[dimension, kpi.operation.column or ""],
        title=kpi.name or dimension,
        evidence_id=evidence_id,
    )
    if _thin(df, threshold):
        return _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
    return chart


# ---------------------------------------------------------------------------
# Shape-driven extras (numeric/ordinal columns beyond the KPI list)
# ---------------------------------------------------------------------------

def _plan_histograms(df: pd.DataFrame, registry: EvidenceRegistry, index: int,
                     measures: List[str],
                     threshold: int) -> List[ChartMetadata]:
    charts: List[ChartMetadata] = []
    for measure in measures:
        series = pd.to_numeric(df[measure], errors="coerce").dropna()
        if len(series) < 3:
            continue
        index += 1
        chart = ChartMetadata(
            chart_id=f"CH-{index:03d}", kind="histogram",
            reason=RULE["rule_6"], columns=[measure],
            title=f"{measure} distribution",
            evidence_id=registry.mint(),
        )
        if _thin(df, threshold):
            chart = _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
        _register(registry, chart)
        charts.append(chart)
    return charts


def _plan_measure_relation(df: pd.DataFrame, registry: EvidenceRegistry,
                           index: int, measures: List[str],
                           existing: List[ChartMetadata],
                           threshold: int) -> List[ChartMetadata]:
    """Rule 7 (scatter) for exactly 2 measures, rule 8 (heatmap) for >= 3."""
    if len(measures) < 2:
        return []
    if len(measures) == 2:
        if any(c.kind == "scatter" and set(c.columns) == set(measures)
               for c in existing):
            return []
        index += 1
        chart = ChartMetadata(
            chart_id=f"CH-{index:03d}", kind="scatter",
            reason=RULE["rule_7"], columns=list(measures),
            title=f"{measures[0]} x {measures[1]}",
            evidence_id=registry.mint(),
        )
        if _thin(df, threshold):
            chart = _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
        _register(registry, chart)
        return [chart]
    index += 1
    chart = ChartMetadata(
        chart_id=f"CH-{index:03d}", kind="heatmap",
        reason=RULE["rule_8"], columns=measures,
        title="Ranked correlation heatmap",
        evidence_id=registry.mint(),
    )
    if _thin(df, threshold):
        chart = _downgrade_low_n(chart, f"{len(df)} rows < {threshold}")
    _register(registry, chart)
    return [chart]


# ---------------------------------------------------------------------------
# Ranking + truncation (spec §1: max_chart_count, drop lowest-ranked)
# ---------------------------------------------------------------------------

_STRENGTH = {
    "line": 100, "scatter": 90, "heatmap": 85, "doughnut": 85,
    "bar": 80, "barh": 80, "histogram": 60,
}


def rank_candidates(candidates: List[ChartMetadata]) -> List[ChartMetadata]:
    """Rank chart candidates by evidence strength (stable on chart_id)."""
    return sorted(candidates, key=lambda c: (-_STRENGTH.get(c.kind, 0),
                                             c.chart_id))


def truncate(candidates: List[ChartMetadata],
             max_chart_count: int) -> Tuple[List[ChartMetadata], bool]:
    """Drop the lowest-ranked charts beyond max_chart_count."""
    if max_chart_count <= 0:
        return [], True
    kept = candidates[:max_chart_count]
    truncated = len(candidates) > max_chart_count
    return kept, truncated


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def plan_charts(df: pd.DataFrame, plan: AnalysisPlan,
                understanding: DatasetUnderstanding,
                registry: EvidenceRegistry,
                max_chart_count: int = 20,
                thin_threshold: int = THIN_THRESHOLD,
                ) -> Tuple[List[ChartMetadata], bool]:
    """Plan all candidate charts from the KPI list + numeric/ordinal columns.

    Returns (chart_metadata list, charts_truncated). Order after ranking is the
    final draw order; excess candidates are dropped lowest-ranked first.
    """
    measures = measure_columns(understanding, df)
    candidates: List[ChartMetadata] = []
    index = 0

    for kpi in plan.candidate_kpis:
        chart = _plan_kpi(df, kpi, registry, index + 1, thin_threshold)
        if chart is not None:
            candidates.append(chart)
            index += 1

    charts = _plan_histograms(df, registry, index, measures, thin_threshold)
    candidates.extend(charts)
    index += len(charts)

    candidates.extend(_plan_measure_relation(
        df, registry, index, measures, candidates, thin_threshold))
    candidates = _dedupe(candidates)
    ranked = rank_candidates(candidates)
    kept, truncated = truncate(ranked, max_chart_count)
    return kept, truncated
