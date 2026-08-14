"""Shared fixtures for analysis stage-5a unit tests."""
from __future__ import annotations

import pandas as pd
import pytest

from shared.schemas import ColumnUnderstanding, DatasetUnderstanding


@pytest.fixture
def make_understanding():
    """Build a DatasetUnderstanding that matches a given DataFrame."""

    def _build(df, measures=("revenue", "quantity"),
               temporal=("date",), dimensions=("category", "product")):
        columns = []
        for name in df.columns:
            if name in measures:
                role = "measure"
            elif name in temporal:
                role = "temporal"
            elif name in dimensions:
                role = "dimension"
            else:
                role = "identifier"
            columns.append(ColumnUnderstanding(
                name=name, role=role,
                dtype=str(df[name].dtype),
                nunique=int(df[name].nunique()),
                nullable=bool(df[name].isna().any()),
            ))
        return DatasetUnderstanding(
            detected_domain="sales",
            domain_confidence=0.9,
            temporal_columns=[c for c in temporal if c in df.columns],
            dimensions=[c for c in dimensions if c in df.columns],
            measures=[c for c in measures if c in df.columns],
            identifiers=["order_id"] if "order_id" in df.columns else [],
            columns=columns,
            has_temporal_data=bool(temporal),
        )

    return _build


SALES = pd.DataFrame({
    "date": ["2024-01-15", "2024-02-15", "2024-03-15", "2023-01-15",
             "2023-02-15", "2023-03-15"],
    "product": ["A", "B", "A", "B", "A", "B"],
    "category": ["X", "Y", "X", "Y", "X", "Y"],
    "revenue": [100, 200, 300, 50, 80, 120],
    "quantity": [1, 2, 3, 4, 5, 6],
})

GROUPS = pd.DataFrame({
    "category": ["X", "X", "X", "X", "X", "X",
                 "Y", "Y", "Y", "Y", "Y", "Y"],
    "revenue": [10, 12, 11, 13, 12, 11,
                30, 32, 31, 33, 32, 31],
})
