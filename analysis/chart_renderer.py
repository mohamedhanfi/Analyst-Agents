"""Stage 5b — hand-rolled SVG chart renderers (§2.5), zero dependencies.

Deterministic SVG generation, one pure function per kind over the 12-kind
whitelist. Every output is XML-escaped (charts are embedded in the HTML
report later), uses the Okabe-Ito color-blind-safe palette, labels values on
bars/points (never color-only), and carries a caption (title + reliability +
evidence_id) as <title>/<desc> for alt text.

Golden rule: aggregate on ALL rows — the renderer never samples.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from analysis.dsl_executor import (grouped_growth_values, grouped_values,
                                   growth_series)
from shared.schemas import ChartMetadata, DslOperation, KpiResult

WIDTH, HEIGHT = 800, 450
MARGIN = {"top": 40, "right": 30, "bottom": 80, "left": 70}

# Okabe-Ito — color-blind safe (never red-green as the only encoding)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#CC79A7",
           "#F0E442", "#D55E00", "#000000"]

_FILL = "none"
_FONT = "12px system-ui, sans-serif"


# ---------------------------------------------------------------------------
# Text + numbers
# ---------------------------------------------------------------------------


def escape_xml(text: Any) -> str:
    """Escape text for safe embedding inside SVG/HTML."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;")
            .replace("'", "&apos;"))


def _fmt(value: float) -> str:
    """Short, deterministic number formatting (no trailing float noise)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, int):
        return f"{value:,}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if value == 0:
        return "0"
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) or math.isinf(result) else result


def _caption(chart: ChartMetadata) -> str:
    parts = [chart.title or "chart", f"kind={chart.kind}",
             f"evidence={chart.evidence_id or 'none'}"]
    if chart.reliability:
        parts.append(f"reliability={chart.reliability}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# SVG scaffolding
# ---------------------------------------------------------------------------


def _svg(chart: ChartMetadata, inner: str, width: int = WIDTH,
         height: int = HEIGHT) -> str:
    title = escape_xml(chart.title or f"chart {chart.chart_id}")
    desc = escape_xml(_caption(chart))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-labelledby="t-{chart.chart_id} d-{chart.chart_id}">'
        f'<title id="t-{chart.chart_id}">{title}</title>'
        f'<desc id="d-{chart.chart_id}">{desc}</desc>'
        f"<rect width='{width}' height='{height}' fill='#ffffff'/>"
        f"{inner}</svg>")


def _axes(width: int, height: int) -> Tuple[float, float, float, float]:
    left = MARGIN["left"]
    top = MARGIN["top"]
    plot_w = width - left - MARGIN["right"]
    plot_h = height - top - MARGIN["bottom"]
    return left, top, plot_w, plot_h


def _grid_and_axis(left: float, top: float, plot_w: float, plot_h: float,
                   y_min: float, y_max: float, x_labels: Sequence[str] | None,
                   y_ticks: int = 5) -> str:
    """Light horizontal gridlines + value axis labels (deterministic)."""
    parts: List[str] = []
    if y_max > y_min:
        for i in range(y_ticks):
            ratio = i / (y_ticks - 1)
            y = top + plot_h * ratio
            value = y_max - (y_max - y_min) * ratio
            parts.append(
                f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' "
                f"y2='{y:.1f}' stroke='#e6e6e6' stroke-width='1'/>"
                f"<text x='{left - 8:.1f}' y='{y + 4:.1f}' "
                f"text-anchor='end' font-size='{_FONT[0:2]}px' "
                f"font-family='{_FONT[5:]}' fill='#555555'>{_fmt(value)}</text>")
    if x_labels:
        step = max(1, math.ceil(len(x_labels) / 12))
        for i, label in enumerate(x_labels):
            if i % step != 0:
                continue
            x = left + plot_w * (i / max(1, len(x_labels) - 1))
            parts.append(
                f"<text x='{x:.1f}' y='{top + plot_h + 20}' text-anchor='middle' "
                f"font-size='11px' font-family='{_FONT[5:]}' "
                f"fill='#555555'>{escape_xml(label)}</text>")
    return "".join(parts)


def _color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


# ---------------------------------------------------------------------------
# Data preparation (deterministic, all rows)
# ---------------------------------------------------------------------------


def _find_kpi(chart: ChartMetadata, kpis: List[KpiResult]) -> KpiResult | None:
    """Match KPI results expanded from the chart's candidate (base kpi_id)."""
    if chart.kpi_id:
        for kpi in kpis:
            if kpi.kpi_id == chart.kpi_id:
                return kpi
        for kpi in kpis:
            if kpi.kpi_id.startswith(chart.kpi_id + "-"):
                return kpi
    for kpi in kpis:
        if kpi.name == chart.title:
            return kpi
    return None


def _operation(chart: ChartMetadata, kpis: List[KpiResult]
               ) -> DslOperation | None:
    kpi = _find_kpi(chart, kpis)
    return kpi.operation if kpi is not None else None


def _aggregate_pairs(df: pd.DataFrame, op: DslOperation,
                     chart: ChartMetadata) -> Tuple[List[str], List[float]]:
    """Deterministic (labels, values) — recomputed from ALL rows, matching
    the DSL executor's own expansion (sorted, group-labeled)."""
    if op.function == "growth":
        pairs = grouped_growth_values(df, op) if op.group_by else []
        labels = [str(k) for k, _ in pairs]
        values = [v for v in (_to_float(v) for _, v in pairs) if v is not None]
        if labels:
            return labels, values
        rows = growth_series(df, op)
        labels = [str(r.get("period") or "") for r in rows]
        values = [v for v in (_to_float(r.get("value")) for r in rows)
                  if v is not None]
        return (labels, values) if len(labels) == len(values) else ([], [])
    if op.group_by:
        pairs = grouped_values(df, op)
        labels = [str(k) for k, _ in pairs]
        values = [v for v in (_to_float(v) for _, v in pairs) if v is not None]
        if len(labels) == len(values):
            return labels, values
    over = op.over_column
    column = op.column
    if over and column and over in df.columns and column in df.columns:
        keys = pd.to_datetime(df[over], errors="coerce")
        pairs: Dict[Any, float] = {}
        for key, raw in zip(keys, df[column]):
            if pd.isna(key):
                continue
            number = _to_float(raw)
            if number is None:
                continue
            pairs.setdefault(key, 0.0)
            pairs[key] += number
        ordered = sorted(pairs)
        return ([str(k.date()) for k in ordered],
                [pairs[k] for k in ordered])
    return [], []


def _raw_numeric(df: pd.DataFrame, column: str) -> List[float]:
    if column not in df.columns:
        return []
    return [v for v in (_to_float(x) for x in df[column]) if v is not None]


# ---------------------------------------------------------------------------
# Bar / barh / lollipop / line / area
# ---------------------------------------------------------------------------


def _render_bar(chart: ChartMetadata, df: pd.DataFrame, kpi: KpiResult,
                labels: List[str], values: List[float]) -> str:
    if not labels:
        return _svg(chart, "<text x='50%' y='50%' text-anchor='middle' "
                           "fill='#999'>no data</text>")
    left, top, plot_w, plot_h = _axes(WIDTH, HEIGHT)
    v_min, v_max = 0.0, max(values)
    span = (v_max - v_min) or 1.0
    n = len(labels)
    slot = plot_w / n
    bar_w = max(2.0, slot * 0.6)
    parts = [_grid_and_axis(left, top, plot_w, plot_h, v_min, v_max, None)]
    for i, (label, value) in enumerate(zip(labels, values)):
        h = plot_h * (value - v_min) / span
        x = left + slot * i + (slot - bar_w) / 2
        y = top + plot_h - h
        parts.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' "
            f"fill='{_color(i)}'/>"
            f"<text x='{x + bar_w / 2:.1f}' y='{y - 5:.1f}' text-anchor='middle' "
            f"font-size='11px' font-family='{_FONT[5:]}' fill='#222222'>"
            f"{_fmt(value)}</text>"
            f"<text x='{x + bar_w / 2:.1f}' y='{top + plot_h + 20}' "
            f"text-anchor='middle' font-size='11px' "
            f"font-family='{_FONT[5:]}' fill='#555555'>"
            f"{escape_xml(label)}</text>")
    return _svg(chart, "".join(parts))


def _render_barh(chart: ChartMetadata, df: pd.DataFrame, kpi: KpiResult,
                 labels: List[str], values: List[float],
                 lollipop: bool = False) -> str:
    if not labels:
        return _svg(chart, "<text x='50%' y='50%' text-anchor='middle' "
                           "fill='#999'>no data</text>")
    left, top, plot_w, plot_h = _axes(WIDTH, HEIGHT)
    v_min, v_max = 0.0, max(values)
    span = (v_max - v_min) or 1.0
    n = len(labels)
    slot = plot_h / n
    bar_h = max(2.0, slot * 0.6)
    parts = []
    for i, (label, value) in enumerate(zip(labels, values)):
        w = plot_w * (value - v_min) / span
        y = top + slot * i + (slot - bar_h) / 2
        x = left
        if lollipop:
            parts.append(
                f"<line x1='{left}' y1='{y + bar_h / 2:.1f}' "
                f"x2='{left + w:.1f}' y2='{y + bar_h / 2:.1f}' "
                f"stroke='{_color(i)}' stroke-width='2'/>"
                f"<circle cx='{left + w:.1f}' cy='{y + bar_h / 2:.1f}' "
                f"r='4' fill='{_color(i)}'/>")
        else:
            parts.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' "
                f"height='{bar_h:.1f}' fill='{_color(i)}'/>")
        parts.append(
            f"<text x='{left + w + 6:.1f}' y='{y + bar_h / 2 + 4:.1f}' "
            f"font-size='11px' font-family='{_FONT[5:]}' fill='#222222'>"
            f"{_fmt(value)}</text>"
            f"<text x='{left - 8:.1f}' y='{y + bar_h / 2 + 4:.1f}' "
            f"text-anchor='end' font-size='11px' "
            f"font-family='{_FONT[5:]}' fill='#555555'>"
            f"{escape_xml(label)}</text>")
    return _svg(chart, "".join(parts))


def _render_line(chart: ChartMetadata, df: pd.DataFrame, kpi: KpiResult,
                 labels: List[str], values: List[float],
                 filled: bool = False) -> str:
    if len(labels) < 2:
        return _svg(chart, "<text x='50%' y='50%' text-anchor='middle' "
                           "fill='#999'>no data</text>")
    left, top, plot_w, plot_h = _axes(WIDTH, HEIGHT)
    v_min, v_max = min(values), max(values)
    span = (v_max - v_min) or 1.0
    pad = span * 0.1
    v_min, v_max = v_min - pad, v_max + pad
    span = v_max - v_min or 1.0
    points: List[str] = []
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + plot_w * (i / (len(labels) - 1))
        y = top + plot_h * (1 - (value - v_min) / span)
        points.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(points)
    color = _color(0)
    markers = "".join(
        f"<circle cx='{p.split(',')[0]}' cy='{p.split(',')[1]}' r='3' "
        f"fill='{color}'/>" for p in points)
    labels_svg = ""
    if len(labels) <= 12:
        step = max(1, len(labels) // 12)
        labels_svg = "".join(
            f"<text x='{points[i].split(',')[0]}' "
            f"y='{top + plot_h + 20}' text-anchor='middle' font-size='11px' "
            f"font-family='{_FONT[5:]}' fill='#555555'>"
            f"{escape_xml(label)}</text>"
            for i, label in enumerate(labels) if i % step == 0)
    inner = _grid_and_axis(left, top, plot_w, plot_h, v_min, v_max,
                           [str(l) for l in labels])
    if filled:
        base = f"{left:.1f},{top + plot_h:.1f}"
        inner += (f"<polygon points='{base} {poly} "
                  f"{points[-1].split(',')[0]},{top + plot_h:.1f}' "
                  f"fill='{color}' fill-opacity='0.25'/>")
    inner += (f"<polyline points='{poly}' fill='{_FILL}' "
              f"stroke='{color}' stroke-width='2.5'/>" + markers + labels_svg)
    return _svg(chart, inner)


# ---------------------------------------------------------------------------
# Doughnut / pie
# ---------------------------------------------------------------------------


def _render_pie(chart: ChartMetadata, df: pd.DataFrame, kpi: KpiResult,
                labels: List[str], values: List[float],
                donut: bool = False) -> str:
    if not labels or sum(values) <= 0:
        return _svg(chart, "<text x='50%' y='50%' text-anchor='middle' "
                           "fill='#999'>no data</text>")
    cx, cy, r = 300.0, 225.0, 150.0
    total = sum(values)
    angle = -90.0
    parts: List[str] = []
    for i, (label, value) in enumerate(zip(labels, values)):
        sweep = 360.0 * value / total
        large = 1 if sweep > 180 else 0
        a1, a2 = angle, angle + sweep
        rad1, rad2 = math.radians(a1), math.radians(a2)
        x1 = cx + r * math.cos(rad1)
        y1 = cy + r * math.sin(rad1)
        x2 = cx + r * math.cos(rad2)
        y2 = cy + r * math.sin(rad2)
        if donut:
            inner_r = r * 0.62
            ix1 = cx + inner_r * math.cos(rad1)
            iy1 = cy + inner_r * math.sin(rad1)
            ix2 = cx + inner_r * math.cos(rad2)
            iy2 = cy + inner_r * math.sin(rad2)
            path = (f"M {x1:.1f} {y1:.1f} A {r:.1f} {r:.1f} 0 {large} 1 "
                    f"{x2:.1f} {y2:.1f} L {ix2:.1f} {iy2:.1f} "
                    f"A {inner_r:.1f} {inner_r:.1f} 0 {large} 0 "
                    f"{ix1:.1f} {iy1:.1f} Z")
        else:
            path = (f"M {cx:.1f} {cy:.1f} L {x1:.1f} {y1:.1f} "
                    f"A {r:.1f} {r:.1f} 0 {large} 1 {x2:.1f} {y2:.1f} Z")
        parts.append(f"<path d='{path}' fill='{_color(i)}' stroke='#ffffff' "
                     f"stroke-width='1.5'/>")
        mid = math.radians(a1 + sweep / 2)
        lx = cx + r * 0.78 * math.cos(mid)
        ly = cy + r * 0.78 * math.sin(mid)
        pct = 100.0 * value / total
        parts.append(
            f"<text x='{lx:.1f}' y='{ly + 4:.1f}' text-anchor='middle' "
            f"font-size='11px' font-family='{_FONT[5:]}' fill='#ffffff'>"
            f"{_fmt(pct)}%</text>")
    legend = ""
    for i, (label, value) in enumerate(zip(labels, values)):
        ly = 90 + i * 22
        legend += (
            f"<rect x='520' y='{ly - 10}' width='14' height='14' "
            f"fill='{_color(i)}'/>"
            f"<text x='542' y='{ly}' font-size='12px' "
            f"font-family='{_FONT[5:]}' fill='#222222'>"
            f"{escape_xml(label)}: {_fmt(value)}</text>")
    return _svg(chart, "".join(parts) + legend)


# ---------------------------------------------------------------------------
# Histogram / boxplot
# ---------------------------------------------------------------------------


def _fd_bins(values: List[float]) -> int:
    series = pd.Series(values)
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    n = len(series)
    if n == 0 or pd.isna(iqr) or iqr <= 0:
        return 1
    h = 2.0 * iqr / (n ** (1.0 / 3.0))
    if h <= 0:
        return 1
    return max(1, min(40, int(math.ceil((series.max() - series.min()) / h))))


def _render_histogram(chart: ChartMetadata, df: pd.DataFrame, kpi: KpiResult,
                      values: List[float]) -> str:
    if not values:
        return _svg(chart, "<text x='50%' y='50%' text-anchor='middle' "
                           "fill='#999'>no data</text>")
    bins = _fd_bins(values)
    hist, edges = _hist(values, bins)
    left, top, plot_w, plot_h = _axes(WIDTH, HEIGHT)
    v_max = max(hist) or 1
    slot = plot_w / len(edges)
    parts = [_grid_and_axis(left, top, plot_w, plot_h, 0, v_max, None)]
    for i, count in enumerate(hist):
        h = plot_h * count / v_max
        x = left + slot * i + 1
        w = max(1.0, slot - 2)
        parts.append(
            f"<rect x='{x:.1f}' y='{top + plot_h - h:.1f}' width='{w:.1f}' "
            f"height='{h:.1f}' fill='{_color(0)}' fill-opacity='0.75'/>")
    tick_count = min(7, bins + 1)
    ticks = [round(i * bins / (tick_count - 1)) for i in range(tick_count)] \
        if tick_count > 1 else [0]
    for i in ticks:
        parts.append(
            f"<text x='{left + slot * i:.1f}' y='{top + plot_h + 36}' "
            f"text-anchor='middle' font-size='11px' "
            f"font-family='{_FONT[5:]}' fill='#555555'>"
            f"{_fmt(edges[i])}</text>")
    parts.append(
        f"<text x='{left}' y='{top + plot_h + 20}' font-size='11px' "
        f"font-family='{_FONT[5:]}' fill='#555555'>"
        f"{escape_xml(chart.columns[0])} (n={len(values)}, bins={bins})</text>")
    return _svg(chart, "".join(parts))


def _hist(values: List[float], bins: int) -> Tuple[List[int], List[float]]:
    series = pd.Series(values)
    lo, hi = series.min(), series.max()
    if hi == lo:
        return [len(values)], [lo, hi + 1]
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    edges[-1] += 1e-9  # include the max in the last bin
    counts = [0] * bins
    for value in values:
        for i in range(bins):
            if edges[i] <= value < edges[i + 1]:
                counts[i] += 1
                break
    return counts, edges


def _render_boxplot(chart: ChartMetadata, df: pd.DataFrame, kpi: KpiResult,
                    values: List[float]) -> str:
    if len(values) < 4:
        return _svg(chart, "<text x='50%' y='50%' text-anchor='middle' "
                           "fill='#999'>too few values</text>")
    series = pd.Series(values)
    q1, q2, q3 = series.quantile(0.25), series.quantile(0.5), \
        series.quantile(0.75)
    iqr = q3 - q1
    whisker_lo = q1 - 1.5 * iqr
    whisker_hi = q3 + 1.5 * iqr
    lo, hi = series.min(), series.max()
    lower = max(lo, whisker_lo)
    upper = min(hi, whisker_hi)
    outliers = [v for v in values if v < lower or v > upper]
    v_min = min(lower, q1, q2, q3, upper)
    v_max = max(lower, q1, q2, q3, upper)
    if outliers:
        v_min = min(v_min, min(outliers))
        v_max = max(v_max, max(outliers))
    span = (v_max - v_min) or 1.0
    cx = 400.0
    box_w = 120.0
    y = lambda value: 60 + 320 * (1 - (value - v_min) / span)
    color = _color(0)
    parts = [
        f"<line x1='{cx - box_w / 2}' y1='{y(lower):.1f}' "
        f"x2='{cx + box_w / 2}' y2='{y(lower):.1f}' stroke='#555555' "
        f"stroke-width='2'/>",
        f"<line x1='{cx}' y1='{y(lower):.1f}' x2='{cx}' y2='{y(q1):.1f}' "
        f"stroke='#555555' stroke-width='2'/>",
        f"<rect x='{cx - box_w / 2:.1f}' y='{y(q3):.1f}' width='{box_w:.1f}' "
        f"height='{y(q1) - y(q3):.1f}' fill='{color}' fill-opacity='0.35' "
        f"stroke='{color}' stroke-width='2'/>",
        f"<line x1='{cx - box_w / 2}' y1='{y(q2):.1f}' x2='{cx + box_w / 2}' "
        f"y2='{y(q2):.1f}' stroke='{color}' stroke-width='2.5'/>",
        f"<line x1='{cx}' y1='{y(q3):.1f}' x2='{cx}' y2='{y(upper):.1f}' "
        f"stroke='#555555' stroke-width='2'/>",
        f"<line x1='{cx - box_w / 2}' y1='{y(upper):.1f}' "
        f"x2='{cx + box_w / 2}' y2='{y(upper):.1f}' stroke='#555555' "
        f"stroke-width='2'/>",
    ]
    for value in outliers:
        parts.append(f"<circle cx='{cx}' cy='{y(value):.1f}' r='3.5' "
                     f"fill='{_color(1)}'/>")
    for name, value in (("Q1", q1), ("median", q2), ("Q3", q3)):
        parts.append(
            f"<text x='{cx + box_w / 2 + 10}' y='{y(value) + 4:.1f}' "
            f"font-size='11px' font-family='{_FONT[5:]}' fill='#555555'>"
            f"{name}: {_fmt(float(value))}</text>")
    parts.append(
        f"<text x='{cx}' y='30' text-anchor='middle' font-size='12px' "
        f"font-family='{_FONT[5:]}' fill='#222222'>"
        f"{escape_xml(chart.columns[0])} (n={len(values)})</text>")
    return _svg(chart, "".join(parts))


# ---------------------------------------------------------------------------
# Scatter / heatmap
# ---------------------------------------------------------------------------


def _render_scatter(chart: ChartMetadata, df: pd.DataFrame, kpi: KpiResult,
                    values: List[float]) -> str:
    cols = [c for c in chart.columns if c in df.columns][:2]
    if len(cols) != 2:
        return _svg(chart, "<text x='50%' y='50%' text-anchor='middle' "
                           "fill='#999'>no data</text>")
    pairs = [(x, y) for x, y in
             ((_to_float(a), _to_float(b))
              for a, b in zip(df[cols[0]], df[cols[1]]))
             if x is not None and y is not None]
    if len(pairs) < 2:
        return _svg(chart, "<text x='50%' y='50%' text-anchor='middle' "
                           "fill='#999'>no data</text>")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    left, top, plot_w, plot_h = _axes(WIDTH, HEIGHT)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0
    color = _color(0)
    shown = min(len(xs), 1500)
    idx = [round(i * (len(xs) - 1) / (shown - 1)) for i in range(shown)] \
        if shown > 1 else [0]
    dots = []
    for i in idx:
        x, y = xs[i], ys[i]
        px = left + plot_w * (x - x_min) / x_span
        py = top + plot_h * (1 - (y - y_min) / y_span)
        dots.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='3' "
                    f"fill='{color}' fill-opacity='0.6'/>")
    trend = ""
    if len(xs) >= 3 and x_span > 0:
        slope, intercept = _linreg(xs, ys)
        x0, x1 = x_min, x_max
        y0 = intercept + slope * x0
        y1 = intercept + slope * x1
        px0 = left + plot_w * (x0 - x_min) / x_span
        py0 = top + plot_h * (1 - (y0 - y_min) / y_span)
        px1 = left + plot_w * (x1 - x_min) / x_span
        py1 = top + plot_h * (1 - (y1 - y_min) / y_span)
        trend = (f"<line x1='{px0:.1f}' y1='{py0:.1f}' x2='{px1:.1f}' "
                 f"y2='{py1:.1f}' stroke='{_color(1)}' stroke-width='2' "
                 f"stroke-dasharray='4 3'/>")
    parts = [
        _grid_and_axis(left, top, plot_w, plot_h, y_min, y_max, None),
        f"<text x='{left}' y='{top + plot_h + 20}' font-size='11px' "
        f"font-family='{_FONT[5:]}' fill='#555555'>"
        f"{escape_xml(cols[0])} → x · {escape_xml(cols[1])} → y</text>",
        trend,
        "".join(dots),
    ]
    return _svg(chart, "".join(parts))


def _linreg(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = num / den if den else 0.0
    return slope, mean_y - slope * mean_x


def _render_heatmap(chart: ChartMetadata, df: pd.DataFrame, kpi: KpiResult,
                    values: List[float]) -> str:
    cols = [c for c in chart.columns if c in df.columns]
    if len(cols) < 2:
        return _svg(chart, "<text x='50%' y='50%' text-anchor='middle' "
                           "fill='#999'>no data</text>")
    corr = df[cols].corr().round(3)
    n = len(cols)
    cell = 40.0
    x0, y0 = 120.0, 60.0
    parts: List[str] = []
    for i, row in enumerate(cols):
        for j, col in enumerate(cols):
            value = float(corr.iloc[i, j])
            if pd.isna(value):
                value = 0.0
            alpha = 0.15 + 0.85 * abs(value)
            color = "#0072B2" if value >= 0 else "#D55E00"
            parts.append(
                f"<rect x='{x0 + j * cell:.1f}' y='{y0 + i * cell:.1f}' "
                f"width='{cell - 2:.1f}' height='{cell - 2:.1f}' "
                f"fill='{color}' fill-opacity='{alpha:.2f}'/>"
                f"<text x='{x0 + j * cell + cell / 2:.1f}' "
                f"y='{y0 + i * cell + cell / 2 + 4:.1f}' text-anchor='middle' "
                f"font-size='11px' font-family='{_FONT[5:]}' fill='#ffffff'>"
                f"{_fmt(value)}</text>"
                f"<text x='{x0 + j * cell + cell / 2:.1f}' "
                f"y='{y0 + i * cell - 8:.1f}' text-anchor='middle' "
                f"font-size='10px' font-family='{_FONT[5:]}' fill='#555555'>"
                f"{escape_xml(col)}</text>")
        parts.append(
            f"<text x='{x0 - 8:.1f}' y='{y0 + i * cell + cell / 2 + 4:.1f}' "
            f"text-anchor='end' font-size='10px' font-family='{_FONT[5:]}' "
            f"fill='#555555'>{escape_xml(row)}</text>")
    return _svg(chart, "".join(parts))


# ---------------------------------------------------------------------------
# Stacked bar
# ---------------------------------------------------------------------------


def _render_stacked(chart: ChartMetadata, df: pd.DataFrame, kpi: KpiResult,
                    labels: List[str], values: List[float]) -> str:
    dimension = chart.columns[0] if chart.columns else None
    measures = [c for c in chart.columns[1:] if c in df.columns]
    if not dimension or dimension not in df.columns or len(measures) < 2:
        return _svg(chart, "<text x='50%' y='50%' text-anchor='middle' "
                           "fill='#999'>no data</text>")
    grouped = df.groupby(dimension, dropna=False)
    rows = sorted(grouped, key=lambda pair: str(pair[0]))
    labels_out: List[str] = []
    stacks: List[List[float]] = []
    for key, group in rows:
        labels_out.append(str(key))
        stacks.append([_to_float(group[col].sum()) or 0.0
                       for col in measures])
    if not labels_out:
        return _svg(chart, "<text x='50%' y='50%' text-anchor='middle' "
                           "fill='#999'>no data</text>")
    totals = [sum(s) for s in stacks]
    v_max = max(totals) or 1.0
    left, top, plot_w, plot_h = _axes(WIDTH, HEIGHT)
    n = len(labels_out)
    slot = plot_w / n
    bar_w = max(2.0, slot * 0.6)
    parts = [_grid_and_axis(left, top, plot_w, plot_h, 0, v_max, None)]
    for i, (label, stack) in enumerate(zip(labels_out, stacks)):
        x = left + slot * i + (slot - bar_w) / 2
        y_cursor = top + plot_h
        for j, value in enumerate(stack):
            h = plot_h * value / v_max
            parts.append(
                f"<rect x='{x:.1f}' y='{y_cursor - h:.1f}' width='{bar_w:.1f}' "
                f"height='{h:.1f}' fill='{_color(j)}'/>")
            y_cursor -= h
        parts.append(
            f"<text x='{x + bar_w / 2:.1f}' y='{y_cursor - 5:.1f}' "
            f"text-anchor='middle' font-size='11px' font-family='{_FONT[5:]}' "
            f"fill='#222222'>{_fmt(totals[i])}</text>"
            f"<text x='{x + bar_w / 2:.1f}' y='{top + plot_h + 20}' "
            f"text-anchor='middle' font-size='11px' "
            f"font-family='{_FONT[5:]}' fill='#555555'>"
            f"{escape_xml(label)}</text>")
    legend = ""
    for j, measure in enumerate(measures):
        ly = 30 + j * 20
        legend += (
            f"<rect x='{left}' y='{ly - 10}' width='14' height='14' "
            f"fill='{_color(j)}'/>"
            f"<text x='{left + 20}' y='{ly}' font-size='12px' "
            f"font-family='{_FONT[5:]}' fill='#222222'>"
            f"{escape_xml(measure)}</text>")
    return _svg(chart, "".join(parts) + legend)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_RENDERERS = {
    "bar": lambda chart, df, kpi, data: _render_bar(
        chart, df, kpi, data["labels"], data["values"]),
    "barh": lambda chart, df, kpi, data: _render_barh(
        chart, df, kpi, data["labels"], data["values"]),
    "lollipop": lambda chart, df, kpi, data: _render_barh(
        chart, df, kpi, data["labels"], data["values"], lollipop=True),
    "line": lambda chart, df, kpi, data: _render_line(
        chart, df, kpi, data["labels"], data["values"]),
    "area": lambda chart, df, kpi, data: _render_line(
        chart, df, kpi, data["labels"], data["values"], filled=True),
    "doughnut": lambda chart, df, kpi, data: _render_pie(
        chart, df, kpi, data["labels"], data["values"], donut=True),
    "pie": lambda chart, df, kpi, data: _render_pie(
        chart, df, kpi, data["labels"], data["values"]),
    "histogram": lambda chart, df, kpi, data: _render_histogram(
        chart, df, kpi, data["series"]),
    "boxplot": lambda chart, df, kpi, data: _render_boxplot(
        chart, df, kpi, data["series"]),
    "scatter": lambda chart, df, kpi, data: _render_scatter(
        chart, df, kpi, data["series"]),
    "heatmap": lambda chart, df, kpi, data: _render_heatmap(
        chart, df, kpi, data["series"]),
    "stacked_bar": lambda chart, df, kpi, data: _render_stacked(
        chart, df, kpi, data["labels"], data["values"]),
}


def _prepare(chart: ChartMetadata, df: pd.DataFrame,
             kpis: List[KpiResult]) -> Dict[str, Any]:
    op = _operation(chart, kpis)
    data: Dict[str, Any] = {"labels": [], "values": [], "series": []}
    if chart.kind in ("histogram", "boxplot", "scatter", "heatmap"):
        columns = [c for c in chart.columns if c in df.columns]
        if chart.kind == "heatmap":
            data["series"] = columns
        else:
            column = columns[0] if columns else (
                op.column if op is not None else None)
            data["series"] = _raw_numeric(df, column) if column else []
        return data
    if op is None:
        return data
    if chart.kind in ("bar", "barh", "lollipop", "doughnut", "pie",
                      "line", "area", "stacked_bar"):
        labels, values = _aggregate_pairs(df, op, chart)
        data["labels"], data["values"] = labels, values
    return data


def render_chart(chart: ChartMetadata, df: pd.DataFrame,
                 kpis: List[KpiResult]) -> str:
    """Render one chart to an SVG string (deterministic, XML-escaped)."""
    renderer = _RENDERERS.get(chart.kind)
    if renderer is None:
        return _svg(chart, f"<text x='50%' y='50%' text-anchor='middle' "
                           f"fill='#999'>unknown kind: "
                           f"{escape_xml(chart.kind)}</text>")
    data = _prepare(chart, df, kpis)
    kpi = _find_kpi(chart, kpis)
    return renderer(chart, df, kpi, data)


def render_all(chart_metadata: List[ChartMetadata], df: pd.DataFrame,
               kpis: List[KpiResult], charts_dir: str | Path
               ) -> Dict[str, Path]:
    """Render every chart and write runs/<run_id>/charts/<chart_id>.svg.

    Returns {chart_id: Path} — the caller records each path in
    chart_metadata.json (chart_path). Never raises on a bad chart.
    """
    out_dir = Path(charts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for chart in chart_metadata:
        try:
            svg = render_chart(chart, df, kpis)
        except Exception:  # noqa: BLE001 -- one bad chart never kills the run
            svg = _svg(chart, "<text x='50%' y='50%' text-anchor='middle' "
                              "fill='#999'>render error</text>")
        path = out_dir / f"{chart.chart_id}.svg"
        path.write_text(svg, encoding="utf-8")
        paths[chart.chart_id] = path
    return paths