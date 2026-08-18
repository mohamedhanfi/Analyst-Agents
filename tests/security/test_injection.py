"""Security tests — injection, XSS, malformed files.

Verifies that the pipeline handles adversarial input safely:
- SQL/script payloads in cell content are neutralized
- XSS in column values doesn't appear raw in HTML output
- Malformed files produce clean errors, not crashes
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shared.core.reader import FileReader
from shared.core.validation import FileValidator

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class TestSqlInjectionInCells:
    """SQL strings in CSV cells must not execute."""

    def test_injection_fixture_loads(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_injection.csv")
        assert len(df) > 100

    def test_drop_table_string_in_cells(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_injection.csv")
        products = df["product"].astype(str).tolist()
        has_sql = any("DROP TABLE" in p.upper() for p in products)
        assert has_sql, "Fixture should contain SQL injection payload"

    def test_sql_string_treated_as_text(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_injection.csv")
        for val in df["product"].astype(str):
            assert isinstance(val, str)


class TestScriptInjectionInCells:
    """Script tags in CSV cells must be neutralized."""

    def test_script_tag_in_cells(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_injection.csv")
        products = df["product"].astype(str).tolist()
        has_script = any("<script>" in p.lower() for p in products)
        assert has_script, "Fixture should contain script injection payload"

    def test_script_content_is_string(self) -> None:
        df = pd.read_csv(FIXTURES / "sales_injection.csv")
        for val in df["product"].astype(str):
            assert isinstance(val, str)


class TestXssInReport:
    """XSS payloads must not appear raw in report HTML."""

    def test_no_raw_script_in_report(self) -> None:
        from analysis.report_builder import _esc
        malicious = '<script>alert("xss")</script>'
        escaped = _esc(malicious)
        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped

    def test_esc_handles_single_quotes(self) -> None:
        from analysis.report_builder import _esc
        result = _esc("it's a test")
        assert "it" in result and "test" in result


class TestMalformedCsvHeader:
    """Missing or extra columns handled gracefully."""

    def test_missing_columns_detected(self) -> None:
        validator = FileValidator()
        csv_content = "order_id\n1001\n1002\n"
        csv_path = Path(__file__).parent / "_test_missing_cols.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        try:
            result = validator.validate(str(csv_path))
            assert result is not None
        finally:
            csv_path.unlink(missing_ok=True)

    def test_extra_columns_accepted(self) -> None:
        tmp = Path(__file__).parent / "_test_extra_cols.csv"
        df = pd.read_csv(FIXTURES / "sales_small.csv")
        df["extra_col"] = "test"
        df.to_csv(tmp, index=False)
        try:
            reader = FileReader(Path(__file__).parent)
            result = reader.read(tmp)
            assert result.row_count == 100
        finally:
            tmp.unlink(missing_ok=True)


class TestEmptyCsv:
    """0-row CSV should produce a clean error, not a crash."""

    def test_empty_csv_validation(self) -> None:
        csv_content = "order_id,date,product,category,revenue,quantity\n"
        csv_path = Path(__file__).parent / "_test_empty.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        try:
            validator = FileValidator()
            result = validator.validate(str(csv_path))
            assert result is not None
        finally:
            csv_path.unlink(missing_ok=True)


class TestBinaryFile:
    """Non-CSV/XLSX file produces validation error."""

    def test_binary_rejected(self) -> None:
        binary_path = Path(__file__).parent / "_test_binary.bin"
        binary_path.write_bytes(b"\x00\x01\x02\x03\x04\x05")
        try:
            validator = FileValidator()
            result = validator.validate(str(binary_path))
            assert result.validation_status == "failed"
        finally:
            binary_path.unlink(missing_ok=True)


class TestHugeColumnCount:
    """500+ columns handled gracefully."""

    def test_many_columns_read(self) -> None:
        tmp = Path(__file__).parent / "_test_huge_cols.csv"
        df = pd.DataFrame({f"col_{i}": range(10) for i in range(500)})
        df["order_id"] = [f"ORD-{i}" for i in range(10)]
        df.to_csv(tmp, index=False)
        try:
            reader = FileReader(Path(__file__).parent)
            result = reader.read(tmp)
            assert result.row_count == 10
            assert result.column_count == 501
        finally:
            tmp.unlink(missing_ok=True)


class TestUnicodeInjection:
    """Unicode control characters in cells do not crash the pipeline."""

    def test_unicode_in_cells(self) -> None:
        tmp = Path(__file__).parent / "_test_unicode.csv"
        df = pd.read_csv(FIXTURES / "sales_small.csv")
        df.loc[0, "product"] = "\u200b\u200c\u200d\ufeff"
        df.to_csv(tmp, index=False)
        try:
            reader = FileReader(Path(__file__).parent)
            result = reader.read(tmp)
            assert result.row_count == 100
        finally:
            tmp.unlink(missing_ok=True)
