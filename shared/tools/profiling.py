"""CrewAI @tool wrappers — profiling + PII detection (stage 1 outputs)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from crewai.tools import tool

from shared.core.pii import PiiDetector
from shared.core.profiler import DataProfiler


def _load_df(extracted_csv: str) -> pd.DataFrame:
    return pd.read_csv(extracted_csv, encoding="utf-8-sig")


@tool("pii_detector_tool")
def pii_detector_tool(extracted_csv: str) -> str:
    """Detect PII columns by rules (names + value patterns), no LLM.

    Returns JSON {"pii_columns": [...]}. Temporal columns (dates) are
    never flagged. PII is redacted from every downstream sample.
    """
    df = _load_df(extracted_csv)
    columns = PiiDetector().detect(df)
    return json.dumps({"pii_columns": columns}, ensure_ascii=False, indent=2)


@tool("data_profiler_tool")
def data_profiler_tool(extracted_csv: str, pii_columns: str, run_dir: str,
                       file_name: str = "", file_hash: str = "") -> str:
    """Build the full DataProfile (shape/dtypes/nulls/dups/PII/counts)
    over the ENTIRE extracted dataset and write metadata/data_profile.json.

    pii_columns: JSON list like '["customer_email"]' (from pii_detector_tool).
    file_name/file_hash: pass from file_reader_tool when available.
    Returns the profile JSON — metadata only, PII cells are [REDACTED].
    """
    df = _load_df(extracted_csv)
    pii = json.loads(pii_columns or "[]")
    profile = DataProfiler().profile(
        df=df,
        file_name=file_name or Path(extracted_csv).stem,
        file_hash=file_hash,
        pii_columns=pii,
    )
    DataProfiler.save(profile, run_dir)
    return json.dumps(profile.model_dump(), ensure_ascii=False, indent=2)