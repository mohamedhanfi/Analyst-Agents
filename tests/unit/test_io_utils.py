"""Unit tests for shared/core/io_utils.py + the report drill-down."""
from __future__ import annotations

import pandas as pd
import pytest

from shared.core.io_utils import (column_stats_chunked, convert_to_parquet,
                                   estimate_csv_rows, file_size_mb,
                                   iter_csv_chunks, parquet_cache_path,
                                   read_analysis_dataframe, read_dataframe)


def test_estimate_csv_rows(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    assert estimate_csv_rows(csv) == 3  # header + 2 data rows


def test_read_dataframe_roundtrip(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(
        csv, index=False)
    frame = read_dataframe(csv)
    assert frame["a"].tolist() == [1, 2, 3]
    assert frame["b"].tolist() == ["x", "y", "z"]


def test_iter_csv_chunks(tmp_path):
    csv = tmp_path / "d.csv"
    rows = [{"a": i, "b": i * 2} for i in range(7)]
    pd.DataFrame(rows).to_csv(csv, index=False)
    chunks = list(iter_csv_chunks(csv, chunk_size=3))
    assert len(chunks) == 3
    assert sum(len(c) for c in chunks) == 7


def test_column_stats_chunked(tmp_path):
    csv = tmp_path / "d.csv"
    rows = [{"a": i, "b": "x" if i % 2 else "y", "c": None if i == 0 else 1.0}
            for i in range(5)]
    pd.DataFrame(rows).to_csv(csv, index=False)
    stats = column_stats_chunked(csv, chunk_size=2)
    assert stats["rows"] == 5
    col_a = stats["columns"]["a"]
    assert col_a["missing"] == 0
    assert col_a["min"] == 0 and col_a["max"] == 4
    assert col_a["sum"] == 10
    assert col_a["nunique"] == 5
    assert stats["columns"]["b"]["nunique"] == 2
    assert stats["columns"]["c"]["missing"] == 1


def test_parquet_cache_when_engine_available(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv, index=False)
    target = convert_to_parquet(csv)
    if target is None:
        pytest.skip("no parquet engine installed")
    assert target == parquet_cache_path(csv)
    frame = read_dataframe(csv)
    assert frame["a"].tolist() == [1, 2, 3]


def test_read_analysis_dataframe_prefers_ready(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame({"v": [1]}).to_csv(processed / "analysis_ready.csv",
                                    index=False)
    pd.DataFrame({"v": [2]}).to_csv(processed / "cleaned_data.csv",
                                    index=False)
    frame = read_analysis_dataframe(tmp_path)
    assert frame["v"].tolist() == [1]
    (processed / "analysis_ready.csv").unlink()
    assert read_analysis_dataframe(tmp_path)["v"].tolist() == [2]
    (processed / "cleaned_data.csv").unlink()
    assert read_analysis_dataframe(tmp_path) is None


def test_file_size_mb_small(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("a\n1\n", encoding="utf-8")
    assert file_size_mb(csv) < 1.0