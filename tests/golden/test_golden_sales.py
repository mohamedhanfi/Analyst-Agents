"""Golden dataset tests — §6 fixture assertions.

Each test verifies a specific property of a golden fixture CSV, ensuring
the data is correct before it enters the pipeline.  These are DATA
VALIDATION tests, not pipeline tests (pipeline e2e is in tests/e2e/).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# sales_small — baseline correctness
# ---------------------------------------------------------------------------


class TestSalesSmall:
    def test_row_count(self, sales_small_df: pd.DataFrame) -> None:
        assert len(sales_small_df) == 100

    def test_revenue_sum(self, sales_small_df: pd.DataFrame,
                         sales_small_total: float) -> None:
        assert abs(sales_small_df["revenue"].sum() - sales_small_total) < 0.01

    def test_order_count(self, sales_small_df: pd.DataFrame,
                         sales_small_orders: int) -> None:
        assert len(sales_small_df) == sales_small_orders

    def test_aov(self, sales_small_df: pd.DataFrame,
                 sales_small_aov: float) -> None:
        aov = sales_small_df["revenue"].mean()
        assert abs(aov - sales_small_aov) < 0.01

    def test_no_missing_revenue(self, sales_small_df: pd.DataFrame) -> None:
        assert sales_small_df["revenue"].isna().sum() == 0

    def test_no_duplicates(self, sales_small_df: pd.DataFrame) -> None:
        assert sales_small_df.duplicated().sum() == 0

    def test_all_columns_present(self, sales_small_df: pd.DataFrame) -> None:
        expected = {"order_id", "date", "product", "category",
                    "revenue", "quantity", "unit_price"}
        assert expected.issubset(set(sales_small_df.columns))


# ---------------------------------------------------------------------------
# sales_missing — missingness injection
# ---------------------------------------------------------------------------


class TestSalesMissing:
    def test_has_missing_revenue(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_missing.csv")
        assert df["revenue"].isna().sum() > 0

    def test_has_missing_quantity(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_missing.csv")
        assert df["quantity"].isna().sum() > 0

    def test_row_count_matches(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_missing.csv")
        assert len(df) == 100

    def test_non_missing_rows_match_baseline(self) -> None:
        base = pd.read_csv(FIXTURES / "sales_small.csv")
        missing = pd.read_csv(FIXTURES / "sales_missing.csv")
        clean = missing.dropna(subset=["revenue", "quantity"])
        base_clean = base.loc[clean.index]
        assert abs(clean["revenue"].sum() - base_clean["revenue"].sum()) < 0.01


# ---------------------------------------------------------------------------
# sales_outliers — extreme value injection
# ---------------------------------------------------------------------------


class TestSalesOutliers:
    def test_has_extreme_values(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_outliers.csv")
        assert (df["revenue"] > 50000).sum() >= 5

    def test_row_count_exceeds_baseline(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_outliers.csv")
        assert len(df) > 100

    def test_baseline_rows_still_present(self) -> None:
        base = pd.read_csv(FIXTURES / "sales_small.csv")
        outliers = pd.read_csv(FIXTURES / "sales_outliers.csv")
        base_ids = set(base["order_id"])
        assert base_ids.issubset(set(outliers["order_id"]))


# ---------------------------------------------------------------------------
# sales_duplicates — exact duplicate injection
# ---------------------------------------------------------------------------


class TestSalesDuplicates:
    def test_has_duplicates(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_duplicates.csv")
        assert df.duplicated().sum() >= 10

    def test_row_count_exceeds_baseline(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_duplicates.csv")
        assert len(df) == 110

    def test_dedup_restores_baseline_totals(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_duplicates.csv")
        deduped = df.drop_duplicates()
        base = pd.read_csv(FIXTURES / "sales_small.csv")
        assert abs(deduped["revenue"].sum() - base["revenue"].sum()) < 0.01
        assert len(deduped) == len(base)


# ---------------------------------------------------------------------------
# sales_injection — SQL/script payloads
# ---------------------------------------------------------------------------


class TestSalesInjection:
    def test_has_injection_payloads(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_injection.csv")
        products = df["product"].tolist()
        has_script = any("<script>" in str(p) for p in products)
        has_sql = any("DROP TABLE" in str(p).upper() for p in products)
        assert has_script or has_sql

    def test_row_count_exceeds_baseline(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_injection.csv")
        assert len(df) > 100


# ---------------------------------------------------------------------------
# sales_pii — PII column injection
# ---------------------------------------------------------------------------


class TestSalesPii:
    def test_has_email_column(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_pii.csv")
        assert "customer_email" in df.columns

    def test_has_phone_column(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_pii.csv")
        assert "customer_phone" in df.columns

    def test_emails_are_valid_format(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_pii.csv")
        emails = df["customer_email"].dropna()
        assert all("@" in str(e) and "." in str(e) for e in emails)

    def test_row_count_matches(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_pii.csv")
        assert len(df) == 100


# ---------------------------------------------------------------------------
# hr — independent HR domain
# ---------------------------------------------------------------------------


class TestHr:
    def test_row_count(self) -> None:
        df = pd.read_csv(FIXTURES / "hr.csv")
        assert len(df) == 80

    def test_has_hr_columns(self) -> None:
        df = pd.read_csv(FIXTURES / "hr.csv")
        expected = {"employee_id", "name", "department", "salary",
                    "hire_date", "performance_score"}
        assert expected.issubset(set(df.columns))

    def test_departments_are_valid(self) -> None:
        df = pd.read_csv(FIXTURES / "hr.csv")
        valid = {"Engineering", "Marketing", "Sales", "Finance",
                 "HR", "Operations"}
        assert set(df["department"].unique()).issubset(valid)


# ---------------------------------------------------------------------------
# finance — independent finance domain
# ---------------------------------------------------------------------------


class TestFinance:
    def test_row_count(self) -> None:
        df = pd.read_csv(FIXTURES / "finance.csv")
        assert len(df) == 90

    def test_has_finance_columns(self) -> None:
        df = pd.read_csv(FIXTURES / "finance.csv")
        expected = {"account_id", "date", "debit", "credit",
                    "balance", "category"}
        assert expected.issubset(set(df.columns))

    def test_categories_are_valid(self) -> None:
        df = pd.read_csv(FIXTURES / "finance.csv")
        valid = {"Revenue", "Expense", "Transfer", "Investment", "Tax"}
        assert set(df["category"].unique()).issubset(valid)
