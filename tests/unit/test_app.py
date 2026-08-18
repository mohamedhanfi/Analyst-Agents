"""Tests for app.py — the stdlib web app (server-less unit coverage)."""
from __future__ import annotations

import sys
import queue
import threading
import time
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
           "file": uploaded, "use_crew": True, "encrypted": True}
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


def test_run_job_questions_answered_interactively(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unanswered questions go interactive: the provider blocks until the
    web UI submits an answer through STATE.answer (never console input)."""
    job = {"run_id": "run_y", "run_dir": str(tmp_path / "run_y"),
           "file": tmp_path / "plain.csv", "use_crew": True,
           "encrypted": False}
    (tmp_path / "run_y").mkdir()
    (tmp_path / "plain.csv").write_text("name,value\nA,1\n")
    seen: dict = {}

    def fake_run_pipeline(file_path: str, **kwargs: object) -> dict:
        provider = kwargs.get("answer_provider")
        assert provider is not None

        def answerer() -> None:
            time.sleep(0.2)
            app_module.STATE.answer(
                "What is the main goal of this analysis?", "grow sales")

        threading.Thread(target=answerer, daemon=True).start()
        # gatherer calls input_func(prompt + " ") — provider must strip.
        seen["goal"] = provider("What is the main goal of this analysis? ")
        return {"status": "passed"}

    with patch("crew.crew.run_pipeline", fake_run_pipeline):
        app_module._run_job(job)

    assert seen["goal"] == "grow sales"


def test_start_pipeline_always_runs_crew_mode(tmp_path: Path) -> None:
    """The deterministic checkbox is gone — every job is use_crew=True."""
    q: queue.Queue = queue.Queue()
    with patch.object(app_module, "_QUEUE", q), \
         patch("crew.crew.run_pipeline",
               lambda **_: {"status": "passed"}):
        result = app_module._start_pipeline("plain.csv", b"name,value\nA,1\n")
    assert result["ok"] is True
    job = q.get_nowait()
    assert job["use_crew"] is True
    assert "answers" not in job


def test_connect_test_reports_llm_misconfiguration(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_module, "load_config",
        lambda require_key: (_ for _ in ()).throw(ValueError("no key")))
    assert app_module._connect_test() is not None


def test_connect_test_propagates_provider_error(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "shared.llm.test_connection",
        lambda cfg: "401 Unauthorized: invalid api key")
    assert app_module._connect_test() == "401 Unauthorized: invalid api key"


def test_state_ask_blocks_until_answered() -> None:
    st = app_module.State()
    results: dict = {}

    def asker() -> None:
        results["answer"] = st.ask("Q?", timeout_seconds=10)

    t = threading.Thread(target=asker, daemon=True)
    t.start()
    time.sleep(0.2)
    assert st.pending_question == "Q?"
    assert st.answer("Q?", "42")
    t.join(5)
    assert results["answer"] == "42"
    assert st.pending_question is None


def test_state_ask_times_out_and_ignores_stale_answer() -> None:
    st = app_module.State()
    t0 = time.monotonic()
    ans = st.ask("Q?", timeout_seconds=0.3)
    assert ans is None
    assert time.monotonic() - t0 < 5
    assert st.pending_question is None
    # stale answer for a cleared question is ignored
    assert st.answer("Q?", "late") is False


def test_llm_connection_test_ok() -> None:
    from shared.llm import test_connection
    from types import SimpleNamespace

    fake = SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content="pong"))])
    with patch("litellm.completion", lambda **_: fake):
        assert test_connection({}) is None


def test_llm_connection_test_failure() -> None:
    from shared.llm import test_connection

    with patch("litellm.completion",
               side_effect=RuntimeError("Connection refused")):
        assert "Connection refused" in test_connection({})
