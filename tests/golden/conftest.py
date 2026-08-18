"""Shared fixtures for golden-dataset tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="session")
def sales_small_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "sales_small.csv")


@pytest.fixture(scope="session")
def sales_small_total() -> float:
    return 10000.0


@pytest.fixture(scope="session")
def sales_small_orders() -> int:
    return 100


@pytest.fixture(scope="session")
def sales_small_aov() -> float:
    return 100.0
