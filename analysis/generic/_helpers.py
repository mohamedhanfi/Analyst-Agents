"""Shared helpers for the statistical suite (analysis/generic/*)."""
from __future__ import annotations

from typing import Any, Dict, List

import math
import numpy as np
import pandas as pd

from shared.schemas import DatasetUnderstanding

MAX_GROUPS = 20          # group-comparison cap (avoid blow-ups)
MAX_CATEGORIES = 50      # categorical checks cap


def measure_columns(understanding: DatasetUnderstanding,
                    df: pd.DataFrame) -> List[str]:
    return [c.name for c in understanding.columns
            if c.role == "measure" and c.name in df.columns]


def temporal_columns(understanding: DatasetUnderstanding,
                     df: pd.DataFrame) -> List[str]:
    return [c.name for c in understanding.columns
            if c.role == "temporal" and c.name in df.columns]


def categorical_columns(understanding: DatasetUnderstanding,
                        df: pd.DataFrame) -> List[str]:
    return [c.name for c in understanding.columns
            if c.role in ("dimension", "categorical")
            and c.name in df.columns
            and 2 <= int(df[c.name].nunique()) <= MAX_CATEGORIES]


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").dropna()


def fisher_ci(r: float, n: int, z: float = 1.96):
    """Approximate 95% CI via Fisher z-transform."""
    if n < 4 or abs(r) >= 1.0:
        return None, None
    z_r = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    return math.tanh(z_r - z * se), math.tanh(z_r + z * se)


def mean_variance_variant(series: pd.Series) -> bool:
    return bool(series.nunique() >= 2 and float(series.std()) > 0)


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
