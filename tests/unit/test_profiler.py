"""Unit tests for shared/core/profiler.DataProfiler."""
from __future__ import annotations

import json
import pandas as pd

from shared.core.profiler import DataProfiler


DF = pd.DataFrame({
    "order_id": [1, 2, 3, 4],
    "revenue": [10.5, None, 30.0, 40.25],
    "city": ["Cairo", "Giza", "Cairo", "Cairo"],
    "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com"],
})


def test_profile_counts():
    p = DataProfiler().profile(DF, "sales.csv", "sha256:abc",
                               pii_columns=["email"])
    assert p.row_count == 4
    assert p.column_count == 4
    assert p.duplicate_rows == 0
    assert p.missing_values == {"revenue": 1}
    assert p.nunique["city"] == 2
    assert p.columns == ["order_id", "revenue", "city", "email"]


def test_pii_redacted_in_sample():
    p = DataProfiler().profile(DF, "sales.csv", "sha256:abc",
                               pii_columns=["email"])
    assert all(row["email"] == "[REDACTED]" for row in p.sample)
    assert p.sample[0]["city"] == "Cairo"


def test_pii_auto_detected_without_hint():
    p = DataProfiler().profile(DF, "sales.csv", "sha256:abc")
    assert "email" in p.pii_columns


def test_save_writes_json(tmp_path):
    p = DataProfiler().profile(DF, "sales.csv", "sha256:abc")
    path = DataProfiler.save(p, tmp_path / "run1")
    assert path.name == "data_profile.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["row_count"] == 4


def test_stratified_sample_covers_categories():
    """Stratified sampling (audit J): every major category appears in the
    20-row preview, unlike head(20) which only shows the first."""
    big = pd.DataFrame({
        "group": [f"g{i % 4}" for i in range(200)],
        "value": [i for i in range(200)],
    })
    p = DataProfiler().profile(big, "big.csv", "sha256:abc")
    assert len(p.sample) == 20
    groups_in_sample = {r["group"] for r in p.sample}
    assert groups_in_sample == {"g0", "g1", "g2", "g3"}


def test_stratified_sample_numeric_spread():
    """Without a categorical column the sample spreads across the value
    range (decile picks), not just the head."""
    big = pd.DataFrame({"value": [i for i in range(100)]})
    p = DataProfiler().profile(big, "num.csv", "sha256:abc")
    assert len(p.sample) == 20
    values = [r["value"] for r in p.sample]
    assert max(values) > 80   # covers the tail
    assert min(values) == 0   # and the head