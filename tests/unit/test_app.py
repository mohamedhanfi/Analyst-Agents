"""Tests for app.py — the stdlib web app (server-less unit coverage)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

import app as app_module


def test_run_job_keeps_csv_extension_on_decrypted_work_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The decrypted .work file must keep .csv/.xlsx — FileValidator only
    accepts SUPPORTED_EXTENSIONS and a bare `.work` suffix fails ingestion."""
    uploaded = tmp_path / "20260101_120000__sales_demo.csv"
    uploaded.write_bytes(b"ciphertext")
    job = {"run_id": "run_x", "run_dir": str(tmp_path / "run_x"),
           "file": uploaded, "use_crew": False, "encrypted": True}
    (tmp_path / "run_x").mkdir()
    calls: dict = {}

    def fake_decrypt(src: Path, dst: Path) -> None:
        dst.write_bytes(b"name,value\nA,1\n")

    def fake_run_pipeline(file_path: str, **kwargs: object) -> dict:
        calls["file_path"] = file_path
        calls["content"] = Path(file_path).read_bytes()
        return {"status": "passed"}

    with patch("shared.security.decrypt_file", fake_decrypt), \
         patch("crew.crew.run_pipeline", fake_run_pipeline):
        app_module._run_job(job)

    assert calls["file_path"].endswith(".work.csv")
    assert calls["content"] == b"name,value\nA,1\n"
    assert not uploaded.with_suffix(".work.csv").exists()  # cleaned up


def test_run_job_does_not_block_on_console_questions(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """In web mode the pipeline must never wait on stdin — the provided
    answer_provider returns "" -> Generic Analysis Mode."""
    job = {"run_id": "run_y", "run_dir": str(tmp_path / "run_y"),
           "file": tmp_path / "plain.csv", "use_crew": False,
           "encrypted": False}
    (tmp_path / "run_y").mkdir()
    (tmp_path / "plain.csv").write_text("name,value\nA,1\n")
    provider: object = None

    def fake_run_pipeline(file_path: str, **kwargs: object) -> dict:
        nonlocal provider
        provider = kwargs.get("answer_provider")
        assert provider is not None
        assert provider("What is the main goal?") == ""
        return {"status": "passed"}

    with patch("crew.crew.run_pipeline", fake_run_pipeline):
        app_module._run_job(job)

    assert provider is not None