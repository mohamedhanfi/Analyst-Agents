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
        sample = self._stratified_sample(redacted)

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
    def _stratified_sample(df: pd.DataFrame) -> List[dict]:
        """20-row preview spread across the data (audit J, stratified).

        Uses the most categorical column when one exists (strata = its top
        values, quota by frequency), otherwise deciles of the first numeric
        column; falls back to a spread (first/middle/last) slice.
        """
        if len(df) <= SAMPLE_ROWS:
            return DataProfiler._to_records(df)

        cat_col = DataProfiler._best_categorical(df)
        if cat_col is not None:
            try:
                counts = df[cat_col].value_counts()
                strata = counts.head(5).index.tolist()
                picks: List[int] = []
                for s in strata:
                    idx = df.index[df[cat_col] == s].tolist()
                    if idx:
                        step = max(1, len(idx) // 4)
                        picks.extend(idx[::step][:4])
                if len(picks) >= SAMPLE_ROWS:
                    picks = picks[:SAMPLE_ROWS]
                else:
                    picks = DataProfiler._fill_picks(df, picks)
                return DataProfiler._to_records(df.iloc[picks])
            except (TypeError, KeyError):
                pass  # fall through to decile / spread

        num_col = next((c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])), None)
        if num_col is not None and len(df) >= SAMPLE_ROWS:
            try:
                sorted_idx = df[num_col].sort_values().index.tolist()
                step = len(sorted_idx) / SAMPLE_ROWS
                picks = [sorted_idx[int(i * step)] for i in range(SAMPLE_ROWS)]
                return DataProfiler._to_records(df.iloc[picks])
            except (KeyError, TypeError):
                pass

        step = max(1, len(df) // SAMPLE_ROWS)
        picks = list(range(0, len(df), step))[:SAMPLE_ROWS]
        return DataProfiler._to_records(df.iloc[picks])

    @staticmethod
    def _best_categorical(df: pd.DataFrame) -> str | None:
        best, best_ratio = None, 0.0
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            try:
                n = int(df[col].nunique(dropna=False))
                ratio = n / max(1, len(df))
            except (TypeError, ValueError):
                continue
            if 0 < ratio <= 0.5 and n <= 50 and ratio >= best_ratio:
                best, best_ratio = col, ratio
        return best

    @staticmethod
    def _fill_picks(df: pd.DataFrame, picks: List[int]) -> List[int]:
        have = set(picks)
        for i in range(0, len(df), max(1, len(df) // SAMPLE_ROWS)):
            if len(picks) >= SAMPLE_ROWS:
                break
            if i not in have:
                picks.append(i)
                have.add(i)
        return picks

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