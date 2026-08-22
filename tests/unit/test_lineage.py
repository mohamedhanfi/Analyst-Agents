"""Tests for the lineage chain (raw -> validated -> repaired -> cleaned ->
analysis-ready): hashes, row counts and operation logs per artifact, plus
end-to-end verification that Stage 3 persists validated_data.csv and Stage 4
freezes analysis_ready.csv for the analysis stage."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from agents.cleaning_agent import run_cleaning
from agents.data_quality import run_data_quality
from agents.ingestion_agent import run_ingestion
from agents.understanding_agent import run_understanding
from shared.core.lineage import (append_steps, artifact_step, df_fingerprint,
                                 file_sha256, read_lineage, step,
                                 write_lineage)
from shared.utils import load_config


@pytest.fixture
def cfg():
    return load_config(require_key=False)


def _business_answers():
    return iter(["Track revenue", "sales", "Top products?", "Set targets"])


def _sales_rows(n=7, negatives=()):
    rows = []
    for i in range(n):
        rows.append({
            "date": f"2024-01-{i + 1:02d}",
            "product": "A" if i % 2 == 0 else "B",
            "revenue": -50.0 if i in negatives else 100.0 + i * 10,
            "quantity": i + 1,
        })
    return rows


def _build_stage4(tmp_path, cfg, df, name="r1"):
    csv = tmp_path / "sales.csv"
    df.to_csv(csv, index=False)
    provider = lambda _prompt: next(_business_answers())
    run_dir = tmp_path / name
    s1 = run_ingestion(str(csv), run_dir=run_dir, cfg=cfg,
                       answer_provider=provider)
    assert s1["status"] == "passed"
    s2 = run_understanding(run_dir, cfg=cfg)
    assert s2["status"] == "passed"
    s3 = run_data_quality(run_dir, cfg=cfg)
    assert s3["status"] in ("passed", "needs_repair")
    s4 = run_cleaning(run_dir, cfg=cfg)
    assert s4["status"] == "passed"
    return run_dir


def _build_stage3(tmp_path, cfg, df, name="r1"):
    """Stops after Stage 3 so the DQ lineage steps can be asserted alone."""
    csv = tmp_path / "sales.csv"
    df.to_csv(csv, index=False)
    provider = lambda _prompt: next(_business_answers())
    run_dir = tmp_path / name
    s1 = run_ingestion(str(csv), run_dir=run_dir, cfg=cfg,
                       answer_provider=provider)
    assert s1["status"] == "passed"
    s2 = run_understanding(run_dir, cfg=cfg)
    assert s2["status"] == "passed"
    s3 = run_data_quality(run_dir, cfg=cfg)
    assert s3["status"] in ("passed", "needs_repair")
    return run_dir


# ---------------------------------------------------------------------------
# Hash/fingerprint primitives
# ---------------------------------------------------------------------------


def test_file_sha256_stable_and_prefixed(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    h1 = file_sha256(p)
    assert h1.startswith("sha256:")
    assert h1 == file_sha256(p)


def test_df_fingerprint_deterministic_and_sensitive():
    a = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    b = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    c = pd.DataFrame({"x": [1, 3], "y": ["a", "b"]})
    assert df_fingerprint(a) == df_fingerprint(b)
    assert df_fingerprint(a) != df_fingerprint(c)


# ---------------------------------------------------------------------------
# Lineage file mechanics
# ---------------------------------------------------------------------------


def test_write_read_append(tmp_path):
    steps = [step("raw", "sales.csv", hash="sha256:abc", rows_after=10)]
    p = write_lineage(tmp_path, "sales.csv", "sha256:src", steps)
    assert p.name == "lineage.json"
    data = read_lineage(tmp_path)
    assert data["source_file"] == "sales.csv"
    assert data["steps"][0]["stage"] == "raw"

    append_steps(tmp_path, [step("cleaned", "data/processed/cleaned_data.csv",
                                 rows_after=9)])
    data = read_lineage(tmp_path)
    assert [s["stage"] for s in data["steps"]] == ["raw", "cleaned"]


def test_artifact_step_computes_hash_and_rows(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("a\n1\n2\n3\n", encoding="utf-8")
    entry = artifact_step(tmp_path, "validated", "x.csv")
    assert entry["hash"] == file_sha256(csv)
    assert entry["rows_after"] == 3


def test_read_lineage_missing_is_empty(tmp_path):
    assert read_lineage(tmp_path) == {"steps": []}


# ---------------------------------------------------------------------------
# End-to-end: Stage 3 persists validated, Stage 4 freezes analysis-ready
# ---------------------------------------------------------------------------


def test_stage3_writes_validated_and_lineage(tmp_path, cfg):
    rows = _sales_rows()
    rows.append(dict(rows[0]))  # exact duplicate -> dropped by deterministic repair
    run_dir = _build_stage3(tmp_path, cfg, pd.DataFrame(rows))

    processed = run_dir / "data" / "processed"
    validated = processed / "validated_data.csv"
    assert validated.is_file()

    lineage = read_lineage(run_dir)
    stages = [s["stage"] for s in lineage["steps"]]
    assert stages == ["raw", "validated", "repaired"]
    repaired = lineage["steps"][2]
    assert repaired["artifact"] == "data/processed/validated_data.csv"
    assert repaired["rows_before"] == repaired["rows_after"] + 1  # duplicate dropped
    assert any(op["op"] == "dedup" for op in repaired["ops"])


def test_stage4_writes_analysis_ready_and_full_chain(tmp_path, cfg):
    run_dir = _build_stage4(tmp_path, cfg, pd.DataFrame(_sales_rows()))

    processed = run_dir / "data" / "processed"
    ready = processed / "analysis_ready.csv"
    cleaned = processed / "cleaned_data.csv"
    assert ready.is_file()
    assert ready.read_bytes() == cleaned.read_bytes()

    lineage = read_lineage(run_dir)
    stages = [s["stage"] for s in lineage["steps"]]
    assert stages == ["raw", "validated", "repaired", "cleaned",
                      "analysis_ready"]
    assert lineage["steps"][3]["hash"].startswith("sha256:")
    assert lineage["steps"][4]["hash"] == lineage["steps"][3]["hash"]
    # the analysis stage consumes the frozen frame
    from agents.analysis import _find_cleaned_csv
    assert _find_cleaned_csv(run_dir).name == "analysis_ready.csv"


def test_lineage_hashes_trace_row_deltas(tmp_path, cfg):
    run_dir = _build_stage4(tmp_path, cfg, pd.DataFrame(_sales_rows(negatives=(1, 3))))

    lineage = read_lineage(run_dir)
    by_stage = {s["stage"]: s for s in lineage["steps"]}
    # two negative rows dropped during cleaning
    cleaned = by_stage["cleaned"]
    assert cleaned["rows_before"] == cleaned["rows_after"] + 2
    assert any(op["op"] == "drop_negative" for op in cleaned["ops"])


def test_analysis_reads_analysis_ready_not_cleaned(tmp_path, cfg):
    run_dir = _build_stage4(tmp_path, cfg, pd.DataFrame(_sales_rows()))
    # tamper with the cleaned file AFTER analysis_ready was frozen — the
    # analysis frame must come from the frozen copy.
    cleaned = run_dir / "data" / "processed" / "cleaned_data.csv"
    cleaned.write_text("x\n999\n", encoding="utf-8")
    from agents.analysis import _find_cleaned_csv
    assert _find_cleaned_csv(run_dir).name == "analysis_ready.csv"