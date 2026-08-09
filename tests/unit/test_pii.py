"""Unit tests for shared/core/pii.PiiDetector."""
from __future__ import annotations

import pandas as pd

from shared.core.pii import PiiDetector


def test_email_column_detected_by_name():
    df = pd.DataFrame({"customer_email": ["a@x.com", "b@y.co"], "val": [1, 2]})
    assert PiiDetector().detect(df) == ["customer_email"]


def test_phone_column_detected_by_name():
    df = pd.DataFrame({"phone_number": ["01012345678", "01098765432"]})
    assert PiiDetector().detect(df) == ["phone_number"]


def test_temporal_columns_never_pii():
    df = pd.DataFrame({
        "order_date": ["2024-01-01", "2024-01-02"],
        "created_at": ["2024-01-01", "2024-01-02"],
    })
    assert PiiDetector().detect(df) == []


def test_value_pattern_fallback():
    df = pd.DataFrame({"contact": ["call 01012345678", "no email here"]})
    assert PiiDetector().detect(df) == ["contact"]


def test_numeric_id_columns_not_pii():
    df = pd.DataFrame({"order_id": [12345678, 87654321],
                       "revenue": [100.5, 200.5]})
    assert PiiDetector().detect(df) == []


def test_redact_replaces_values():
    df = pd.DataFrame({"email": ["a@x.com"], "val": [1]})
    out = PiiDetector.redact(df, ["email"])
    assert out["email"].iloc[0] == "[REDACTED]"
    assert out["val"].iloc[0] == 1