"""CrewAI @tool wrappers — file validation / read / sheet extract."""
from __future__ import annotations

import json

from crewai.tools import tool

from shared.core.reader import FileReader
from shared.core.validation import FileValidator
from shared.utils import load_config


def _file_validator() -> FileValidator:
    cfg = load_config()
    limits = cfg.get("limits", {})
    return FileValidator(
        max_file_size_mb=float(limits.get("max_file_size_mb", 200.0)),
        min_rows=int(limits.get("min_rows", 5)),
    )


@tool("file_validator_tool")
def file_validator_tool(file_path: str) -> str:
    """Deeply validate a CSV/XLSX file before any processing.

    Checks in order: extension -> file size (computed from disk, never
    trusted from input) -> content signature -> parseability + row count.

    Returns a JSON ValidationResult on success.
    Raises ValueError with details when validation fails — treat that as
    a hard stop for the ingestion task.
    """
    result = _file_validator().validate(file_path)
    payload = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    if result.validation_status == "failed":
        raise ValueError(f"FILE_VALIDATION_FAILED\n{payload}")
    return payload


@tool("file_reader_tool")
def file_reader_tool(file_path: str, run_dir: str) -> str:
    """Read/discover a validated file into the run's extracted directory.

    CSV: extracted immediately to data/extracted/<name>.csv.
    XLSX: discovery only — returns sheet names + row counts so the agent
    can pick one; then use file_sheet_extract_tool.

    Returns a JSON ReadResult (metadata only — never cell content).
    Raises ValueError on unreadable/unsupported files.
    """
    result = FileReader(run_dir).read(file_path)
    return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)


@tool("file_sheet_extract_tool")
def file_sheet_extract_tool(file_path: str, sheet_name: str, run_dir: str) -> str:
    """Extract ONE XLSX sheet to data/extracted/<name>__<sheet>.csv.

    Call only after file_reader_tool reported it and the sheet was
    selected by the user/business questions.
    """
    extracted = FileReader(run_dir).extract_sheet(file_path, sheet_name)
    return json.dumps({"selected_sheet": sheet_name,
                       "extracted_path": extracted}, ensure_ascii=False)