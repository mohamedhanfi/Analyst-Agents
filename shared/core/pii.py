"""PiiDetector — rule-based PII detection, no LLM (§2.1).

Rules: column-name hints (email/phone/name/address/...), value patterns
(email regex, phone patterns), and an explicit exclusion of temporal
columns (dates are data, not PII — avoids false positives like order_date).
Exports a redact() helper so samples never leak PII.
"""
from __future__ import annotations

import re
from typing import List

import pandas as pd

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
PHONE_SEARCH_RE = re.compile(r"\+?\d{7,15}")

# Strong column-name hints (normalized: lowercase, no spaces/_/-)
NAME_RE = re.compile(r"email|e[-_.]?mail|phone|mobile|tel|cell|\bname\b|address|ssn|national[\s_-]?id|passport|credit[\s_-]?card|card[\s_-]?num|iban|bank[\s_-]?acc|salary")  # noqa: E501

# Columns whose names look sensitive but are NOT PII (dates/periods)
TEMPORAL_RE = re.compile(r"date|time|period|month|year|week|quarter|day\b|created|updated|timestamp")

REDACTED = "[REDACTED]"


class PiiDetector:
    def detect(self, df: pd.DataFrame, sample_size: int = 100) -> List[str]:
        pii_columns: List[str] = []
        for column in df.columns:
            name = str(column).strip()
            normalized = re.sub(r"[\s_\-]+", "", name.lower())
            if TEMPORAL_RE.search(normalized):
                continue
            if NAME_RE.search(normalized):
                pii_columns.append(name)
                continue
            # Value patterns only on text columns — a numeric ID column
            # (e.g. order_id with 8-digit values) is not PII.
            if pd.api.types.is_string_dtype(df[column]):
                base = df[column].dropna().astype(str).head(sample_size)
                if len(base) and (
                    any(EMAIL_RE.match(str(v)) for v in base)
                    or any(PHONE_SEARCH_RE.search(str(v)) for v in base)
                ):
                    pii_columns.append(name)
        return pii_columns

    @staticmethod
    def redact(df: pd.DataFrame, pii_columns: List[str]) -> pd.DataFrame:
        """Return a copy with PII cells replaced by [REDACTED]."""
        out = df.copy()
        for column in pii_columns:
            if column in out.columns:
                out[column] = REDACTED
        return out