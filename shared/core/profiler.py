"""DataProfiler — shape/types/nulls/duplicates + redacted sample (§2.1, §3.1).

All numbers come from pandas over the FULL dataset. The 20-row sample is
only ever a redacted preview for downstream LLM stages (Understanding);
raw cell content never leaves this module beyond the sanitized sample.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from shared.core.pii import PiiDetector
from shared.schemas import DataProfile

SAMPLE_ROWS = 20


class DataProfiler:
    def __init__(self, pii_detector: PiiDetector | None = None):
        self.pii_detector = pii_detector or PiiDetector()

    def profile(self,
                df: pd.DataFrame,
                file_name: str,
                file_hash: str,
                pii_columns: List[str] | None = None) -> DataProfile:
        pii_columns = (pii_columns if pii_columns is not None
                       else self.pii_detector.detect(df))

        missing = {str(col): int(n) for col, n in df.isna().sum().items() if n > 0}
        dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
        nunique = {str(col): int(df[col].nunique(dropna=False))
                   for col in df.columns}

        redacted = self.pii_detector.redact(df, pii_columns)
        sample = self._to_records(redacted.head(SAMPLE_ROWS))

        return DataProfile(
            file_name=file_name,
            file_hash=file_hash,
            row_count=int(len(df)),
            column_count=int(len(df.columns)),
            columns=[str(c) for c in df.columns],
            column_types=dtypes,
            missing_values=missing,
            duplicate_rows=int(df.duplicated().sum()),
            pii_columns=pii_columns,
            nunique=nunique,
            sample=sample,
            validation_status="passed",
        )

    @staticmethod
    def save(profile: DataProfile, run_dir: str | Path) -> Path:
        path = Path(run_dir) / "metadata" / "data_profile.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile.model_dump(), ensure_ascii=False,
                                   indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _to_records(df: pd.DataFrame) -> List[dict]:
        records: List[dict] = []
        for _, row in df.iterrows():
            rec = {}
            for col, value in row.items():
                if pd.isna(value):
                    rec[str(col)] = None
                elif isinstance(value, (int, float)):
                    rec[str(col)] = value if isinstance(value, int) else round(float(value), 6)
                else:
                    rec[str(col)] = str(value)
            records.append(rec)
        return records