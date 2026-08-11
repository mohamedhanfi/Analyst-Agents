"""Flow Review — live web viewer for the Insight Forge pipeline (§1).

Shows every stage, tool call and artifact IN REAL TIME as the pipeline
runs, and the data moving from one stage/agent to the next. Pure stdlib
(no Flask) — the browser polls a tiny JSON API while a background thread
drives the deterministic pipeline.

    python tests/Flow_review/app.py            # opens http://127.0.0.1:8765
    python tests/Flow_review/app.py --demo     # auto-start a sample run

If the port is taken (e.g. another local service), the app scans the next
10 ports and prints the URL it actually bound to.

Stage 3+ are not implemented yet — they render as dimmed "Task N" cards
and light up automatically once their agents land (they all write the same
run.jsonl audit trail via RunLogger, so this viewer never changes).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

from agents.ingestion_agent import run_ingestion
from agents.understanding_agent import run_understanding
from shared.utils import load_config

APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "runs"
UPLOADS_DIR = APP_DIR / "uploads"
HOST = "127.0.0.1"
PORT = 8765

STAGES: List[Dict[str, Any]] = [
    {"id": "ingestion", "num": 1, "name": "Ingestion", "implemented": True},
    {"id": "understanding", "num": 2, "name": "Understanding",
     "implemented": True},
    {"id": "data_quality", "num": 3, "name": "Data Quality",
     "implemented": False, "task": 4},
    {"id": "cleaning", "num": 4, "name": "Cleaning",
     "implemented": False, "task": 5},
    {"id": "analysis", "num": 5, "name": "Analysis",
     "implemented": False, "task": 7},
    {"id": "insights", "num": 6, "name": "Insights & Recommendations",
     "implemented": False, "task": 8},
    {"id": "report", "num": 7, "name": "Report",
     "implemented": False, "task": 9},
    {"id": "qa", "num": 8, "name": "QA",
     "implemented": False, "task": 10},
]

# stage id -> artifacts the viewer previews when the stage finishes
ARTIFACTS: Dict[str, List[str]] = {
    "ingestion": ["data/extracted/", "metadata/data_profile.json",
                  "knowledge/business_context.json"],
    "understanding": ["metadata/dataset_understanding.json",
                      "metadata/analysis_plan.json"],
}

_ANSWERS = iter(["Track revenue growth", "sales",
                 "Which category grows fastest?", "Set targets"])


def _auto_answer(prompt: str) -> str:
    """Deterministic stand-in for the user so runs never block on stdin."""
    if "sheet" in prompt.lower():
        return "Sales"
    try:
        return next(_ANSWERS)
    except StopIteration:
        return ""


class State:
    """Shared state between the API thread and the pipeline thread."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.busy = False
        self.run_id: str | None = None
        self.run_dir: Path | None = None
        self.stage_status: Dict[str, str] = {
            s["id"]: "pending" for s in STAGES}
        self.stage_duration: Dict[str, float] = {}
        self.error: str | None = None
        self.events: List[Dict[str, Any]] = []

    def reset(self, run_id: str, run_dir: Path) -> None:
        self.busy = True
        self.run_id = run_id
        self.run_dir = run_dir
        self.stage_status = {s["id"]: "pending" for s in STAGES}
        self.stage_duration = {}
        self.error = None
        self.events = []

    def set_stage(self, stage: str, status: str,
                  duration: float | None = None) -> None:
        with self.lock:
            self.stage_status[stage] = status
            if duration is not None:
                self.stage_duration[stage] = duration

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "busy": self.busy,
                "run_id": self.run_id,
                "stage_status": dict(self.stage_status),
                "stage_duration": dict(self.stage_duration),
                "error": self.error,
                "stages": STAGES,
            }


STATE = State()


def _run_pipeline(file_path: Path) -> None:
    try:
        _run_pipeline_impl(file_path)
    except Exception as exc:  # noqa: BLE001
        import traceback
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        (RUNS_DIR / "last_error.log").write_text(
            traceback.format_exc(), encoding="utf-8")
        STATE.error = str(exc)
        STATE.busy = False


def _run_pipeline_impl(file_path: Path) -> None:
    cfg = load_config(require_key=False)
    run_id = f"flow_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    STATE.reset(run_id, run_dir)

    def _stage(stage_id: str, call):
        STATE.set_stage(stage_id, "running")
        t0 = time.monotonic()
        try:
            summary = call()
            status = summary.get("status", "failed")
            STATE.set_stage(stage_id, status, time.monotonic() - t0)
            return summary
        except Exception as exc:  # noqa: BLE001
            STATE.error = f"{stage_id}: {exc}"
            STATE.set_stage(stage_id, "failed", time.monotonic() - t0)
            return {"status": "failed", "error": str(exc)}

    s1 = _stage("ingestion",
                lambda: run_ingestion(str(file_path), run_dir=run_dir,
                                      cfg=cfg, answer_provider=_auto_answer))
    if s1.get("status") == "passed":
        _stage("understanding",
               lambda: run_understanding(run_dir, cfg=cfg))
    else:
        STATE.set_stage("understanding", "skipped")
    STATE.busy = False


def _start_pipeline(file_name: str, payload: bytes) -> Dict[str, Any]:
    if STATE.busy:
        return {"ok": False, "error": "a run is already in progress"}
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe = Path(file_name).name
    dest = UPLOADS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}__{safe}"
    dest.write_bytes(payload)
    if STATE.run_dir:
        pass  # keep runs isolated; each start makes a fresh run dir
    thread = threading.Thread(target=_run_pipeline, args=(dest,),
                              daemon=True)
    thread.start()
    return {"ok": True, "file": str(dest)}


def _demo_pipeline() -> None:
    import io
    import pandas as pd

    rows = []
    for i in range(1, 121):
        rows.append([f"2024-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
                     f"P{i % 6}", ["W", "G", "S"][i % 3],
                     round(100 + i * 3.7, 2), i % 13,
                     f"note {i}"])
    df = pd.DataFrame(rows, columns=["order_date", "product", "category",
                                     "revenue", "quantity", "notes"])
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}__demo_sales.csv"
    df.to_csv(dest, index=False)
    time.sleep(0.5)
    thread = threading.Thread(target=_run_pipeline, args=(dest,),
                              daemon=True)
    thread.start()


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


def _read_log_events(run_dir: Path, offset: int) -> List[Dict[str, Any]]:
    log_path = run_dir / "logs" / "run.jsonl"
    if not log_path.exists():
        return []
    with open(log_path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    return [json.loads(line) for line in lines[offset:]]


class Handler(BaseHTTPRequestHandler):
    server_version = "FlowReview/0.1"

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"))

    # ------------------------------------------------------------- GET

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = (APP_DIR / "index.html").read_text(encoding="utf-8")
            return self._send(200, html.encode("utf-8"), "text/html")
        if parsed.path == "/api/state":
            state = STATE.snapshot()
            try:
                offset = int((parse_qs(parsed.query).get("offset") or [0])[0])
            except ValueError:
                offset = 0
            if STATE.run_dir:
                events = _read_log_events(STATE.run_dir, offset)
                state["events"] = events
                state["offset"] = offset + len(events)
            else:
                state["events"] = []
                state["offset"] = 0
            return self._json(state)
        if parsed.path == "/api/artifact":
            query = parse_qs(parsed.query)
            rel = (query.get("rel") or [""])[0]
            if not STATE.run_dir or not rel:
                return self._json({"error": "missing rel"}, 400)
            target = (STATE.run_dir / rel).resolve()
            if not target.is_relative_to(STATE.run_dir.resolve()):
                return self._json({"error": "outside run dir"}, 400)
            if target.is_dir():
                files = sorted(p.name for p in target.iterdir())
                return self._json({"dir": rel, "files": files})
            if target.is_file():
                if target.suffix == ".json":
                    return self._json({"path": rel, "content":
                        json.loads(target.read_text(encoding="utf-8"))})
                if target.suffix == ".csv":
                    lines = target.read_text(encoding="utf-8-sig") \
                        .splitlines()[:8]
                    return self._json({"path": rel, "preview": lines})
            return self._json({"error": "not found"}, 404)
        self._json({"error": "not found"}, 404)

    # ------------------------------------------------------------ POST

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/start":
            length = int(self.headers.get("Content-Length", 0))
            file_name = (parse_qs(parsed.query).get("name") or [""])[0]
            if not file_name or length == 0:
                return self._json({"ok": False,
                                   "error": "file upload missing"}, 400)
            payload = self.rfile.read(length)
            return self._json(_start_pipeline(file_name, payload))
        self._json({"error": "not found"}, 404)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # keep the console clean


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flow_review")
    parser.add_argument("--port", type=int, default=PORT,
                        help="start scanning from this port; if busy, the "
                             "next free port is used")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="auto-start a sample pipeline on launch")
    args = parser.parse_args(argv)

    server: ThreadingHTTPServer | None = None
    last_error: OSError | None = None
    for port in range(args.port, args.port + 10):
        try:
            server = ThreadingHTTPServer((HOST, port), Handler)
            break
        except OSError as exc:  # WinError 10013 / EADDRINUSE — port taken
            last_error = exc
    if server is None:
        print(f"ports {args.port}-{args.port + 9} are all in use — "
              f"last error: {last_error}", file=sys.stderr)
        return 1

    url = f"http://{HOST}:{server.server_address[1]}/"
    print(f"Flow Review running at {url}  (Ctrl+C to stop)")
    if args.demo:
        _demo_pipeline()
        print("Demo pipeline started — watch the stages move.")
    elif not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
