"""Stage 1 — Ingestion: CrewAI agent + deterministic run_ingestion path (§2.1).

Golden rule: the LLM decides WHAT (validate / read / extract / profile steps),
the tools and the run_ingestion orchestrator DO the work. run_ingestion with
use_crew=True kicks off a real CrewAI Crew; use_crew=False runs the same steps
deterministically so the path is unit-testable without an API key. All outputs
land under runs/<run_id>/ with a RunLogger audit trail (§5).

CLI: python -m agents.ingestion_agent <file> [--crew]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from crewai import Agent, Crew, Process, Task

from shared.core.business_context import BusinessContextGatherer
from shared.core.pii import PiiDetector
from shared.core.profiler import DataProfiler
from shared.core.reader import FileReader
from shared.core.validation import FileValidator
from shared.llm import build_llm
from shared.logger import RunLogger
from shared.schemas import BusinessContext, DataProfile
from shared.tools import (
    data_profiler_tool,
    file_reader_tool,
    file_sheet_extract_tool,
    file_validator_tool,
    human_input_tool,
    pii_detector_tool,
)
from shared.utils import allocate_run_id, init_run_layout, load_config

STAGE = "ingestion"

INGESTION_TOOLS = [
    file_validator_tool,
    file_reader_tool,
    file_sheet_extract_tool,
    pii_detector_tool,
    data_profiler_tool,
    human_input_tool,
]


# ---------------------------------------------------------------------------
# CrewAI assembly
# ---------------------------------------------------------------------------


def build_ingestion_agent(cfg: Dict[str, Any]) -> Agent:
    a_cfg = cfg["agents"]["ingestion"]
    return Agent(
        role=a_cfg["role"],
        goal=a_cfg["goal"],
        backstory=a_cfg["backstory"],
        llm=build_llm(cfg, "ingestion"),
        tools=INGESTION_TOOLS,
        verbose=False,
        memory=False,
        allow_delegation=False,
        cache=False,
        max_iter=12,
    )


def build_ingestion_tasks(agent: Agent, file_path: str, run_dir: str,
                          cfg: Dict[str, Any]) -> List[Task]:
    file_path = str(file_path)
    run_dir = str(run_dir)

    validate_and_extract = Task(
        description=(
            f"Validate and extract the uploaded file '{file_path}' for run "
            f"'{run_dir}'.\n"
            f"Step 1: call file_validator_tool(file_path='{file_path}'). "
            "If it raises FILE_VALIDATION_FAILED, STOP and report exactly that "
            "error — the run is invalid.\n"
            f"Step 2: call file_reader_tool(file_path='{file_path}', "
            f"run_dir='{run_dir}').\n"
            "  - If the returned JSON has file_type 'csv', extraction is done.\n"
            "  - If file_type 'xlsx' with exactly ONE sheet, call "
            "file_sheet_extract_tool(file_path, sheet_name='<that sheet>', "
            "run_dir) to extract it.\n"
            "  - If file_type 'xlsx' with MULTIPLE sheets: first call "
            "human_input_tool with the prompt 'This file contains multiple "
            "sheets: <list the sheet names>. Which sheet should we analyse? "
            "(type the exact name): ', then call file_sheet_extract_tool with "
            "the sheet name the user chose. If the answer is empty or timed "
            "out, pick the sheet with the MOST rows and extract that one.\n"
            "Never invent values — use only the tool outputs."
        ),
        expected_output=(
            "A short confirmation: the extracted CSV path and, for xlsx, the "
            "selected sheet. Never include raw cell content."
        ),
        agent=agent,
    )

    gather_business_context = Task(
        description=(
            f"Gather business context for the dataset of file '{file_path}' "
            f"(run '{run_dir}').\n"
            "Ask the user the following questions ONE at a time using "
            "human_input_tool:\n"
            "1. 'What is the main goal of this analysis?'\n"
            "2. 'What business domain / industry is this dataset from (e.g. "
            "sales, finance, hr)?'\n"
            "3. 'Which business questions should the report answer (one per "
            "line, up to 5)?'\n"
            "4. 'What decisions will be taken based on this report?'\n"
            "If a human_input_tool call returns a JSON containing 'timed_out', "
            "stop asking and set generic_mode to true.\n"
            "Return ONLY a single JSON object with this exact shape:\n"
            '{"file_name": "<source file name>", "sheet_used": null, '
            '"business_questions": ["..."], "answers": {"question": "answer"}, '
            '"goal_summary": "...", "context_confidence": 0.0, '
            '"generic_mode": false}\n'
            "Set context_confidence between 0 and 1 based on how many answers "
            "you received; generic_mode true when any question timed out."
        ),
        expected_output=(
            "The strict JSON object described above. No prose, no raw cell "
            "content."
        ),
        agent=agent,
    )

    profile_dataset = Task(
        description=(
            f"Build the data profile for the extracted dataset of run "
            f"'{run_dir}'.\n"
            "Step 1: find the single extracted CSV under "
            f"'{run_dir}/data/extracted/' (list files there).\n"
            "Step 2: call pii_detector_tool(extracted_csv='<that path>').\n"
            "Step 3: call data_profiler_tool(extracted_csv='<that path>', "
            "pii_columns='<the JSON list returned in step 2>', "
            f"run_dir='{run_dir}', file_name='{Path(file_path).name}', "
            "file_hash='<the sha256 of the source file if you know it, else "
            "empty>').\n"
            "Report the profile JSON exactly as returned — the tool already "
            "wrote metadata/data_profile.json."
        ),
        expected_output=(
            "The full data_profile JSON as returned by data_profiler_tool."
        ),
        agent=agent,
    )

    return [validate_and_extract, gather_business_context, profile_dataset]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_ingestion(file_path: str,
                  run_dir: str | Path | None = None,
                  cfg: Dict[str, Any] | None = None,
                  logger: RunLogger | None = None,
                  use_crew: bool = False,
                  answer_provider: Callable[[str], str] | None = None,
                  timeout_seconds: float | None = None) -> Dict[str, Any]:
    cfg = cfg or load_config(require_key=bool(use_crew))
    if run_dir is None:
        run_id, run_dir = allocate_run_id()
    else:
        run_dir = Path(run_dir)
        run_id = run_dir.name
    run_dir = init_run_layout(run_dir)

    log = logger or RunLogger(run_dir, run_id)
    log.stage_start(STAGE)
    start = time.monotonic()
    try:
        if use_crew:
            summary = _run_crew(str(file_path), run_dir, cfg, log)
        else:
            summary = _run_deterministic(str(file_path), run_dir, cfg, log,
                                         answer_provider, timeout_seconds)
        summary.setdefault("run_id", run_id)
        status = summary.get("status", "failed")
    except Exception as exc:  # noqa: BLE001 -- a failed stage must not crash the run
        log.error(STAGE, f"ingestion failed: {exc}")
        status = "failed"
        summary = {"run_id": run_id, "stage": STAGE, "status": status,
                   "error": str(exc)}
    log.stage_end(STAGE, status, time.monotonic() - start)
    summary["log_path"] = str(run_dir / "logs" / "run.jsonl")
    return summary


def _run_deterministic(file_path: str, run_dir: Path, cfg: Dict[str, Any],
                       log: RunLogger,
                       answer_provider: Callable[[str], str] | None,
                       timeout_seconds: float | None) -> Dict[str, Any]:
    limits = cfg.get("limits", {})
    validator = FileValidator(
        max_file_size_mb=float(limits.get("max_file_size_mb", 200.0)),
        min_rows=int(limits.get("min_rows", 5)),
    )
    t0 = time.monotonic()
    validation = validator.validate(file_path)
    log.tool_call(STAGE, "file_validator_tool",
                  validation.validation_status, time.monotonic() - t0)
    if validation.validation_status == "failed":
        log.error(STAGE, "file validation failed")
        return {
            "stage": STAGE, "status": "failed",
            "file_name": validation.file_name,
            "validation_status": "failed",
            "errors": [e.model_dump() for e in validation.errors],
        }

    reader = FileReader(run_dir)
    t0 = time.monotonic()
    read = reader.read(file_path)
    log.tool_call(STAGE, "file_reader_tool", "passed", time.monotonic() - t0)

    timeout = timeout_seconds if timeout_seconds is not None \
        else float(limits.get("human_input_timeout_min", 5.0)) * 60
    gatherer = BusinessContextGatherer(timeout_seconds=timeout,
                                       input_func=answer_provider)
    sheet_names = [s.name for s in read.sheets] if read.file_type == "xlsx" else None
    context = gatherer.gather(file_name=read.file_name, sheet_names=sheet_names)
    log.tool_call(STAGE, "business_context_gatherer", "passed", 0.0)

    extracted_path = read.extracted_path
    if read.file_type == "xlsx":
        names = [s.name for s in read.sheets]
        if len(names) == 1:
            sheet = names[0]
        elif context.sheet_used in names:
            sheet = context.sheet_used
        else:
            sheet = max(read.sheets, key=lambda s: s.row_count).name
            context.sheet_used = sheet
            log.fallback(STAGE, "generic_mode_largest_sheet")
            log.info(STAGE, "no sheet chosen; used largest", sheet=sheet)
        t0 = time.monotonic()
        extracted_path = reader.extract_sheet(file_path, sheet)
        log.tool_call(STAGE, "file_sheet_extract_tool", "passed",
                      time.monotonic() - t0)
        context.sheet_used = sheet

    df = _load_df(extracted_path)
    t0 = time.monotonic()
    pii_columns = PiiDetector().detect(df)
    log.tool_call(STAGE, "pii_detector_tool", "passed", time.monotonic() - t0)
    t0 = time.monotonic()
    profile = DataProfiler().profile(df=df, file_name=read.file_name,
                                     file_hash=read.file_hash,
                                     pii_columns=pii_columns)
    DataProfiler.save(profile, run_dir)
    log.tool_call(STAGE, "data_profiler_tool", "passed", time.monotonic() - t0)
    BusinessContextGatherer.save(context, run_dir)

    return {
        "stage": STAGE, "status": "passed",
        "file_name": read.file_name, "file_hash": read.file_hash,
        "row_count": profile.row_count, "column_count": profile.column_count,
        "pii_columns": profile.pii_columns,
        "sheet_used": context.sheet_used, "generic_mode": context.generic_mode,
        "context_confidence": context.context_confidence,
        "extracted_path": str(extracted_path),
        "data_profile_path": str(run_dir / "metadata" / "data_profile.json"),
        "business_context_path": str(run_dir / "knowledge" / "business_context.json"),
        "errors": [],
    }


def _run_crew(file_path: str, run_dir: Path, cfg: Dict[str, Any],
              log: RunLogger) -> Dict[str, Any]:
    agent = build_ingestion_agent(cfg)
    tasks = build_ingestion_tasks(agent, file_path, str(run_dir), cfg)
    crew = Crew(agents=[agent], tasks=tasks, process=Process.sequential,
                verbose=False, cache=False)
    t0 = time.monotonic()
    result = crew.kickoff(inputs={})
    log.info(STAGE, "crew kickoff finished",
             duration_s=round(time.monotonic() - t0, 3))

    extracted_csv = _find_extracted_csv(run_dir)
    if extracted_csv is None:
        raise RuntimeError("No extracted CSV was produced by the crew.")

    profile = _finalize_profile(extracted_csv, run_dir, file_path)
    context = _finalize_context(result, Path(file_path).name, extracted_csv,
                                run_dir)

    return {
        "stage": STAGE, "status": "passed",
        "file_name": Path(file_path).name,
        "file_hash": profile.file_hash,
        "row_count": profile.row_count, "column_count": profile.column_count,
        "pii_columns": profile.pii_columns,
        "sheet_used": context.sheet_used, "generic_mode": context.generic_mode,
        "context_confidence": context.context_confidence,
        "extracted_path": str(extracted_csv),
        "data_profile_path": str(run_dir / "metadata" / "data_profile.json"),
        "business_context_path": str(run_dir / "knowledge" / "business_context.json"),
        "errors": [],
    }


def _finalize_profile(extracted_csv: Path, run_dir: Path,
                      source_file: str) -> DataProfile:
    profile_path = run_dir / "metadata" / "data_profile.json"
    if profile_path.exists():
        profile = DataProfile(
            **json.loads(profile_path.read_text(encoding="utf-8")))
    else:
        df = _load_df(str(extracted_csv))
        pii_columns = PiiDetector().detect(df)
        profile = DataProfiler().profile(
            df=df, file_name=Path(source_file).name,
            file_hash=_sha256(source_file), pii_columns=pii_columns)
    profile.file_hash = _sha256(source_file)
    profile.file_name = Path(source_file).name
    DataProfiler.save(profile, run_dir)
    return profile


def _finalize_context(result, file_name: str, extracted_csv: Path,
                      run_dir: Path) -> BusinessContext:
    outputs = getattr(result, "tasks_output", None) or []
    raw = ""
    if len(outputs) > 1:
        task = outputs[1]
        raw = str(getattr(task, "raw", "") or getattr(task, "output", "") or "")
    context = _parse_context_json(raw, file_name)
    if context.sheet_used is None:
        stem = extracted_csv.stem
        if "__" in stem:
            context.sheet_used = stem.split("__", 1)[1]
    BusinessContextGatherer.save(context, run_dir)
    return context


def _parse_context_json(raw: str, file_name: str) -> BusinessContext:
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            data = json.loads(raw[start:end + 1])
            data["file_name"] = file_name
            return BusinessContext(**data)
    except Exception:  # noqa: BLE001 -- fall back to generic mode
        pass
    return BusinessContext(file_name=file_name, generic_mode=True)


def _find_extracted_csv(run_dir: Path) -> Path | None:
    csvs = sorted((run_dir / "data" / "extracted").glob("*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_df(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ingestion_agent",
        description="Run Insight Forge stage 1 (ingestion) on a CSV/XLSX file.")
    parser.add_argument("file_path", help="path to the CSV or XLSX file")
    parser.add_argument("--crew", action="store_true",
                        help="run via the real CrewAI agent (requires API key)")
    parser.add_argument("--run-dir", default=None,
                        help="existing run dir to write into (default: new run)")
    args = parser.parse_args(argv)

    summary = run_ingestion(args.file_path, run_dir=args.run_dir,
                            use_crew=args.crew)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
