"""Insight Forge — web app.

User-facing front end for the full 8-stage pipeline: upload a CSV/XLSX,
watch the stages run live, then open the QA-verified HTML report.

Pure stdlib (no Flask) — the browser polls a tiny JSON API while a
single background worker drains a FIFO queue (one run at a time, so runs
never step on each other's files). Optional API-key auth: set
INSIGHT_FORGE_API_KEY in the environment and every run-starting POST
requires an X-API-Key header.

    python app.py                       # opens http://127.0.0.1:8000
    python app.py --demo                # auto-start a sample run on launch

If the port is taken, the app scans the next 10 ports and prints the URL
it actually bound to.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

# CrewAI/litellm ship an OpenTelemetry tracer that retries exporting span
# batches to a collector that is never running here ("Service Unavailable"
# spam). Disable the whole telemetry stack before anything imports it.
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
import queue
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from shared.utils import load_config

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
UPLOADS_DIR = ROOT / "uploads"
HOST = "127.0.0.1"
PORT = 8000
AUTH_KEY = os.environ.get("INSIGHT_FORGE_API_KEY", "")

STAGES: List[Dict[str, Any]] = [
    {"id": "ingestion", "num": 1, "name": "Ingestion"},
    {"id": "understanding", "num": 2, "name": "Understanding"},
    {"id": "data_quality", "num": 3, "name": "Data Quality"},
    {"id": "cleaning", "num": 4, "name": "Cleaning"},
    {"id": "analysis", "num": 5, "name": "Analysis"},
    {"id": "insights", "num": 6, "name": "Insights & Recommendations"},
    {"id": "report", "num": 7, "name": "Report"},
    {"id": "qa", "num": 8, "name": "QA"},
]

# stages that appear in run.jsonl but map onto an existing card
_ALIASES = {"data_quality_recheck": "cleaning"}


class State:
    """Shared state between the API thread and the worker thread."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.busy = False
        self.run_id: Optional[str] = None
        self.error: Optional[str] = None
        self.queued: List[str] = []
        self.pending_question: Optional[str] = None
        self._answer_ready = threading.Event()
        self._answer_value: Optional[str] = None

    def start(self, run_id: str) -> None:
        with self.lock:
            self.busy = True
            self.run_id = run_id
            self.error = None

    def finish(self) -> None:
        with self.lock:
            self.busy = False

    def fail(self, msg: str) -> None:
        with self.lock:
            self.error = msg
            self.busy = False

    def enqueue(self, run_id: str) -> None:
        with self.lock:
            self.queued.append(run_id)

    def dequeue(self, run_id: str) -> None:
        with self.lock:
            if run_id in self.queued:
                self.queued.remove(run_id)

    def ask(self, prompt: str, timeout_seconds: float) -> Optional[str]:
        """Block the worker until the web UI answers (or timeout)."""
        with self.lock:
            self.pending_question = prompt
            self._answer_value = None
        self._answer_ready.clear()
        self._answer_ready.wait(timeout_seconds)
        with self.lock:
            answer = self._answer_value
            self.pending_question = None
        return answer

    def answer(self, prompt: str, value: str) -> bool:
        """Web UI submits an answer for the currently pending question."""
        with self.lock:
            if self.pending_question != prompt:
                return False
            self._answer_value = value
            self._answer_ready.set()
        return True

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {"busy": self.busy, "run_id": self.run_id,
                    "error": self.error, "queued": list(self.queued),
                    "pending_question": self.pending_question}


STATE = State()
_QUEUE: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()


# ---------------------------------------------------------------------------
# Pipeline driver — single FIFO worker
# ---------------------------------------------------------------------------


def _web_ask(prompt: str) -> str:
    """Blocking ask routed through the web UI (registered as the global
    interactive provider + the ingestion answer_provider)."""
    return STATE.ask(prompt.strip(), timeout_seconds=_ASK_TIMEOUT) or ""


_ASK_TIMEOUT = 300.0


def _run_job(job: Dict[str, Any]) -> None:
    STATE.start(job["run_id"])
    work_file = job["file"]
    try:
        # §5/G: uploads are stored as Fernet ciphertext; decrypt to a
        # temporary working file that is deleted when the run finishes.
        # Keep the original extension — FileValidator only accepts .csv/.xlsx.
        if job.get("encrypted"):
            work_file = job["file"].with_suffix(".work" + job["file"].suffix)
            from shared.security import decrypt_file
            decrypt_file(job["file"], work_file)
        from crew.crew import run_pipeline
        # Every question the pipeline asks (ingestion, review gate, crew
        # human tool) is answered in the web UI — never on the console.
        def _provider(prompt: str) -> str:
            return _web_ask(prompt.strip())
        run_pipeline(file_path=str(work_file), use_crew=True,
                     output_dir=job["run_dir"],
                     answer_provider=_provider)
    except Exception as exc:  # noqa: BLE001 -- surface to the UI, never crash
        STATE.fail(f"{type(exc).__name__}: {exc}")
        return
    finally:
        if work_file != job["file"]:
            Path(work_file).unlink(missing_ok=True)
    STATE.finish()


def _worker_loop() -> None:
    while True:
        job = _QUEUE.get()
        if job is None:
            return
        try:
            _run_job(job)
        finally:
            STATE.dequeue(job["run_id"])


_WORKER = threading.Thread(target=_worker_loop, daemon=True)
_WORKER.start()


def _start_pipeline(file_name: str, payload: bytes) -> Dict[str, Any]:
    from shared.utils import allocate_run_id
    run_id, run_dir = allocate_run_id()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe = Path(file_name).name
    dest = UPLOADS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}__{safe}"
    dest.write_bytes(payload)
    encrypted = False
    from shared.security import encryption_enabled
    if encryption_enabled(load_config(require_key=False)):
        from shared.security import encrypt_bytes
        cipher = encrypt_bytes(payload)
        dest.write_bytes(cipher)
        encrypted = True
    job = {"run_id": run_id, "run_dir": run_dir, "file": dest,
           "use_crew": True, "encrypted": encrypted}
    with STATE.lock:
        queued = STATE.busy
        if queued:
            STATE.enqueue(run_id)
    _QUEUE.put(job)
    return {"ok": True, "file": str(dest), "run_id": run_id,
            "queued": queued, "encrypted": encrypted}


def _start_demo() -> Dict[str, Any]:
    demo = ROOT / "tests" / "fixtures" / "sales_demo.csv"
    if not demo.is_file():
        return {"ok": False, "error": "demo fixture not found"}
    return _start_pipeline("sales_demo.csv", demo.read_bytes())


def _connect_test() -> Optional[str]:
    """Pre-flight LLM check — return None when the API is reachable,
    otherwise the error message to show in the app."""
    try:
        cfg = load_config(require_key=True)
    except Exception as exc:  # noqa: BLE001 -- missing/malformed config
        return f"LLM is not configured: {exc}"
    from shared.llm import test_connection
    return test_connection(cfg)


# ---------------------------------------------------------------------------
# Read-only run access
# ---------------------------------------------------------------------------


def _read_log_events(run_dir: Path) -> List[Dict[str, Any]]:
    log_path = run_dir / "logs" / "run.jsonl"
    if not log_path.exists():
        return []
    with open(log_path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _resolve_run_dir(run_id: str) -> Optional[Path]:
    """Resolve a run_id to its directory, guarding against path traversal."""
    candidate = (RUNS_DIR / run_id).resolve()
    try:
        candidate.relative_to(RUNS_DIR.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_dir() else None


def _stage_status(run_dir: Path) -> Dict[str, Any]:
    """Rebuild stage statuses + error purely from run.jsonl."""
    status = {s["id"]: "pending" for s in STAGES}
    durations: Dict[str, float] = {}
    error: Optional[str] = None
    for ev in _read_log_events(run_dir):
        stage = _ALIASES.get(ev.get("stage", ""), ev.get("stage", ""))
        if ev.get("kind") == "stage_start" and stage in status:
            status[stage] = "running"
        elif ev.get("kind") == "stage_end" and stage in status:
            status[stage] = ev.get("status", "failed")
            if ev.get("duration_s") is not None:
                durations[stage] = ev["duration_s"]
            if ev.get("status") == "failed":
                error = ev.get("message") or f"{stage}: failed"
        elif ev.get("kind") == "error" and error is None:
            error = ev.get("message")
    return {"status": status, "durations": durations, "error": error}


def _run_summary(run_dir: Path) -> Dict[str, Any]:
    """Verdict/score + report path for a run (manifest first, QA fallback)."""
    manifest = _load_json(run_dir / "master_manifest.json")
    verdict = _load_json(run_dir / "metadata" / "qa_verdict.json")
    if manifest:
        return {
            "run_id": run_dir.name,
            "verdict": manifest.get("verdict", "NEEDS_REVISION"),
            "score": manifest.get("score"),
            "report_path": manifest.get("report_path"),
            "use_crew": manifest.get("use_crew", False),
            "duration_s": manifest.get("duration_s"),
        }
    if verdict:
        return {
            "run_id": run_dir.name,
            "verdict": verdict.get("verdict", "NEEDS_REVISION"),
            "score": verdict.get("score"),
            "report_path": None,
            "use_crew": False,
            "duration_s": None,
        }
    return {"run_id": run_dir.name, "verdict": None, "score": None,
            "report_path": None, "use_crew": False, "duration_s": None}


def _list_runs() -> List[Dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir() or d.name == "_template":
            continue
        info = _run_summary(d)
        info["modified"] = d.stat().st_mtime
        runs.append(info)
    runs.sort(key=lambda r: r["modified"], reverse=True)
    return runs


def _state_payload(run_id: Optional[str], historical: bool = False
                   ) -> Dict[str, Any]:
    """Live or historical pipeline state for the UI."""
    run_dir = None
    if run_id:
        run_dir = _resolve_run_dir(run_id)
        if run_dir is None:
            return {"error": "unknown run_id"}
    elif STATE.run_id:
        run_dir = _resolve_run_dir(STATE.run_id)

    payload: Dict[str, Any] = {}
    if not historical:
        payload.update(STATE.snapshot())
    else:
        payload.update({"busy": False, "run_id": run_id, "error": None,
                        "queued": []})
    payload["auth_required"] = bool(AUTH_KEY)

    if run_dir is not None:
        info = _stage_status(run_dir)
        summary = _run_summary(run_dir)
        payload["run_dir"] = str(run_dir)
        payload["stages"] = STAGES
        payload["status"] = info["status"]
        payload["durations"] = info["durations"]
        payload["stage_error"] = info["error"]
        payload.update(summary)
        insights = _load_json(run_dir / "outputs" / "insights.json")
        payload["insights"] = (insights or {}).get("insights", [])
        charts = run_dir / "charts"
        if not charts.is_dir():
            charts = run_dir / "outputs" / "charts"
        payload["chart_files"] = sorted(p.name for p in charts.iterdir()) \
            if charts.is_dir() else []
        report = run_dir / "report.html"
        payload["report_ready"] = report.is_file()
    return payload


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "InsightForgeApp/0.1"

    _CSP = ("default-src 'self'; script-src 'unsafe-inline' "
            "https://cdn.jsdelivr.net; "
            "style-src 'unsafe-inline' https://cdn.jsdelivr.net "
            "https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; connect-src 'self'")

    def _send(self, code: int, body: bytes,
              ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", self._CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"))

    def _auth_ok(self) -> bool:
        """Optional API-key auth (§8): enabled only via env var."""
        if not AUTH_KEY:
            return True
        return self.headers.get("X-API-Key") == AUTH_KEY

    def _reject(self) -> None:
        self._json({"ok": False, "error": "invalid or missing X-API-Key"},
                   401)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = _INDEX_HTML
            return self._send(200, html.encode("utf-8"), "text/html")
        if parsed.path == "/api/runs":
            return self._json({"runs": _list_runs()})
        if parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            run_id = (query.get("run_id") or [None])[0]
            historical = run_id is not None
            return self._json(_state_payload(run_id, historical))
        if parsed.path.startswith("/run/"):
            parts = parsed.path[len("/run/"):].split("/")
            run_id = parts[0]
            rel = "/".join(parts[1:]) or "report.html"
            run_dir = _resolve_run_dir(run_id)
            if run_dir is None:
                return self._json({"error": "unknown run_id"}, 404)
            target = (run_dir / rel).resolve()
            try:
                target.relative_to(run_dir.resolve())
            except ValueError:
                return self._json({"error": "outside run dir"}, 400)
            if not target.is_file():
                return self._json({"error": "not found"}, 404)
            suffix = target.suffix.lower()
            ctype = {
                ".html": "text/html; charset=utf-8",
                ".svg": "image/svg+xml",
                ".json": "application/json; charset=utf-8",
                ".csv": "text/csv; charset=utf-8",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".css": "text/css",
                ".js": "text/javascript",
            }.get(suffix, "application/octet-stream")
            return self._send(200, target.read_bytes(), ctype)
        self._json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        try:
            self._do_post()
        except Exception as exc:  # noqa: BLE001 -- never drop the connection
            try:
                self._json({"ok": False, "error": f"server error: {exc}"}, 500)
            except Exception:  # noqa: BLE001 -- socket already gone
                pass

    def _do_post(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/api/start", "/api/demo", "/api/answer"):
            if not self._auth_ok():
                return self._reject()
        if parsed.path == "/api/answer":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8", "replace")
            query = parse_qs(body)
            value = (query.get("answer") or [""])[0]
            answered = STATE.answer(STATE.pending_question or "", value)
            return self._json({"ok": answered})
        if parsed.path == "/api/start":
            query = parse_qs(parsed.query)
            length = int(self.headers.get("Content-Length", 0))
            file_name = (query.get("name") or [""])[0]
            if not file_name or length == 0:
                return self._json({"ok": False,
                                   "error": "file upload missing"}, 400)
            err = _connect_test()
            if err:
                return self._json({"ok": False, "error": err})
            payload = self.rfile.read(length)
            return self._json(_start_pipeline(file_name, payload))
        if parsed.path == "/api/demo":
            err = _connect_test()
            if err:
                return self._json({"ok": False, "error": err})
            return self._json(_start_demo())
        self._json({"error": "not found"}, 404)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # keep the console clean


# --------------------------------------------------------------------------
# Front end (single self-contained page — no external assets)
# --------------------------------------------------------------------------

_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Insight Forge</title>
<style>
  :root { --bg:#0f172a; --panel:#1e293b; --line:#334155; --fg:#e2e8f0;
          --muted:#94a3b8; --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444;
          --run:#3b82f6; --accent:#8b5cf6; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif; }
  header { display:flex; align-items:center; gap:14px; padding:18px 28px;
           border-bottom:1px solid var(--line); background:var(--panel); }
  header h1 { font-size:20px; margin:0; letter-spacing:.3px; }
  header .tag { color:var(--muted); font-size:13px; }
  .logo { width:34px; height:34px; border-radius:9px;
          background:linear-gradient(135deg,var(--run),var(--accent));
          display:grid; place-items:center; font-weight:700; color:#fff; }
  main { max-width:1100px; margin:0 auto; padding:24px 28px 60px; }
  .card { background:var(--panel); border:1px solid var(--line);
          border-radius:14px; padding:20px; margin-bottom:20px; }
  .qlabel { display:block; font-size:13px; color:var(--muted);
            margin:12px 0 4px; }
  .qinput { width:100%; padding:9px; border:1px solid var(--line);
            border-radius:10px; background:#0b1220; color:var(--fg);
            font:inherit; resize:vertical; }
  .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
  input[type=file] { flex:1; min-width:260px; padding:10px; border:1px dashed
          var(--line); border-radius:10px; background:#0b1220; color:var(--fg); }
  button { border:0; border-radius:10px; padding:10px 18px; cursor:pointer;
           font-weight:600; font-size:14px; }
  .primary { background:var(--run); color:#fff; }
  .ghost { background:#334155; color:var(--fg); }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .hint { color:var(--muted); font-size:12.5px; margin-top:10px; }
  .verdict { display:inline-flex; align-items:center; gap:10px;
             font-size:15px; font-weight:700; padding:8px 16px;
             border-radius:10px; background:#0b1220; border:1px solid var(--line); }
  .badge { padding:4px 10px; border-radius:999px; font-size:12.5px;
           font-weight:700; }
  .badge.ok { background:rgba(34,197,94,.15); color:var(--ok); }
  .badge.warn { background:rgba(245,158,11,.15); color:var(--warn); }
  .badge.bad { background:rgba(239,68,68,.15); color:var(--bad); }
  .badge.run { background:rgba(59,130,246,.15); color:var(--run); }
  .badge.muted { background:#334155; color:var(--muted); }
  .stages { display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr));
            gap:12px; }
  .stage { background:#0b1220; border:1px solid var(--line); border-radius:12px;
           padding:14px; }
  .stage .num { font-size:11px; color:var(--muted); letter-spacing:1px; }
  .stage .name { font-weight:600; margin:4px 0 8px; }
  .stage .dur { color:var(--muted); font-size:12px; float:right; }
  .stage.pending { opacity:.55; }
  .stage.running { border-color:var(--run); }
  .stage.passed { border-color:var(--ok); }
  .stage.failed { border-color:var(--bad); }
  .stage.failed, .stage.passed { opacity:1; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%;
         margin-right:6px; vertical-align:middle; }
  .dot.pending { background:#475569; }
  .dot.running { background:var(--run); animation:pulse 1s infinite; }
  .dot.passed { background:var(--ok); }
  .dot.failed { background:var(--bad); }
  @keyframes pulse { 50% { opacity:.3; } }
  .error { color:var(--bad); background:rgba(239,68,68,.08);
           border:1px solid rgba(239,68,68,.35); border-radius:10px;
           padding:12px 16px; margin-top:16px; font-size:14px; }
  a.link { color:var(--run); text-decoration:none; font-weight:600; }
  a.link:hover { text-decoration:underline; }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th, td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; font-size:12px;
       text-transform:uppercase; letter-spacing:.5px; }
  tr.clickable { cursor:pointer; }
  tr.clickable:hover td { background:#0b1220; }
  .insight { padding:10px 0; border-bottom:1px solid var(--line); font-size:14px; }
  .insight:last-child { border-bottom:0; }
  .insight .k { color:var(--accent); font-weight:700; font-size:12px;
                text-transform:uppercase; letter-spacing:.5px; }
  #busy { display:none; color:var(--muted); font-size:13px; margin-top:12px; }
  #busy.show { display:block; }
  .spin { display:inline-block; width:12px; height:12px; border:2px solid
          var(--line); border-top-color:var(--run); border-radius:50%;
          margin-right:8px; vertical-align:-2px; animation:spin 1s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .muted { color:var(--muted); }
  #question { display:none; position:fixed; right:20px; bottom:20px;
              width:min(430px, 94vw); z-index:60;
              box-shadow:0 10px 34px rgba(0,0,0,.55);
              border:1px solid var(--run); }
  #question .qhead { font-size:12px; color:var(--run); font-weight:700;
                     letter-spacing:1px; text-transform:uppercase;
                     margin-bottom:6px; }
</style>
</head>
<body>
<header>
  <div class="logo">IF</div>
  <h1>Insight Forge</h1>
  <span class="tag">8-stage analytics pipeline · QA-verified reports</span>
</header>
<main>
  <div class="card">
    <div class="row">
      <input type="file" id="file" accept=".csv,.xlsx">
      <button class="primary" id="run">Run pipeline</button>
      <button class="ghost" id="demo">Run demo dataset</button>
    </div>
    <div id="authRow" style="display:none;margin-top:12px">
      <input type="password" id="apikey" style="width:280px;padding:9px;
        border:1px solid var(--line);border-radius:10px;background:#0b1220;
        color:var(--fg)" placeholder="Server API key (required)">
      <span class="hint">The server requires an X-API-Key for runs.</span>
    </div>
    <div class="hint">Every run uses the LLM (CrewAI) — the API connection
      is tested automatically before the pipeline starts.</div>
    <div id="busy"><span class="spin"></span>Pipeline running…</div>
    <div id="queued" class="hint"></div>
    <div id="error"></div>
  </div>

  <div class="card" id="result" style="display:none">
    <div class="row" style="justify-content:space-between">
      <div class="verdict">Verdict: <span id="verdict"></span></div>
      <div>Score: <b id="score"></b> · <span class="muted" id="meta"></span></div>
    </div>
    <div id="open" style="margin-top:12px"></div>
    <div id="insights"></div>
    <div id="charts" class="row" style="margin-top:8px"></div>
  </div>

  <div class="card">
    <h2 style="font-size:16px;margin:0 0 12px">Stage progress</h2>
    <div class="stages" id="stages"></div>
    <div id="stageError"></div>
  </div>

  <div class="card">
    <h2 style="font-size:16px;margin:0 0 12px">Run history</h2>
    <table>
      <thead><tr><th>Run</th><th>Verdict</th><th>Score</th><th>Time</th></tr></thead>
      <tbody id="history"></tbody>
    </table>
  </div>

  <div class="card" id="question">
    <div class="qhead">Question from the pipeline</div>
    <div id="qText" style="font-size:14px;white-space:pre-wrap"></div>
    <textarea id="qAnswer" class="qinput" rows="3"
      placeholder="Type your answer here…"></textarea>
    <div class="row" style="margin-top:10px">
      <button class="primary" id="qSend">Send answer</button>
      <button class="ghost" id="qSkip">Skip (generic mode)</button>
    </div>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const STAGES = ["ingestion","understanding","data_quality","cleaning",
                "analysis","insights","report","qa"];
const NAMES  = {ingestion:"Ingestion", understanding:"Understanding",
  data_quality:"Data Quality", cleaning:"Cleaning", analysis:"Analysis",
  insights:"Insights & Recommendations", report:"Report", qa:"QA"};

function badge(v) {
  if (!v) return '<span class="badge muted">—</span>';
  const cls = v === "APPROVED" ? "ok" :
              v === "APPROVED_WITH_WARNINGS" ? "warn" :
              v === "NEEDS_REVISION" ? "bad" : "muted";
  return '<span class="badge ' + cls + '">' + v + '</span>';
}

function renderStages(status) {
  $("stages").innerHTML = STAGES.map(id => {
    const st = status[id] || "pending";
    const dur = window._dur && window._dur[id]
      ? " · " + window._dur[id].toFixed(1) + "s" : "";
    return '<div class="stage ' + st + '"><span class="dur">' + dur +
      '</span><span class="num">STAGE ' + (STAGES.indexOf(id) + 1) +
      '</span><div class="name"><span class="dot ' + st + '"></span>' +
      NAMES[id] + '</div></div>';
  }).join("");
}

function showResult(state) {
  const ok = !!state.verdict;
  $("result").style.display = ok ? "block" : "none";
  if (!ok) return;
  $("verdict").innerHTML = badge(state.verdict);
  $("score").textContent = state.score ?? "—";
  $("meta").textContent = state.run_id +
    (state.duration_s ? " · " + state.duration_s.toFixed(1) + "s" : "");
  $("open").innerHTML = state.report_ready
    ? '<a class="link" href="/run/' + state.run_id + '/report.html" ' +
      'target="_blank">Open the HTML report →</a>' : "";
  const ins = (state.insights || []).slice(0, 6);
  $("insights").innerHTML = ins.length
    ? '<h3 style="font-size:14px;margin:14px 0 4px">Insights</h3>' +
      ins.map(i => '<div class="insight"><span class="k">' +
        (i.kind || "INSIGHT") + '</span> ' + (i.summary || i.insight || "") +
        '</div>').join("")
    : "";
  $("charts").innerHTML = (state.chart_files || []).map(f =>
    '<a class="link" target="_blank" href="/run/' + state.run_id +
    '/outputs/charts/' + f + '">' + f + '</a>').join(" ");
}

function showQuestion(q) {
  $("qText").textContent = q;
  $("question").style.display = "block";
}
function hideQuestion() {
  $("question").style.display = "none";
  $("qAnswer").value = "";
}
function answerQ(value) {
  fetch("/api/answer", {method: "POST",
    body: "answer=" + encodeURIComponent(value),
    headers: apiHeaders()}).catch(() => {});
  hideQuestion();
}
$("qSend").addEventListener("click", () => answerQ($("qAnswer").value));
$("qSkip").addEventListener("click", () => answerQ(""));
$("qAnswer").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); answerQ($("qAnswer").value); }
});

function poll() {
  fetch("/api/state").then(r => r.json()).then(state => {
    if (state.error && !state.run_id) { return; }
    window._dur = state.durations || {};
    renderStages(state.status || {});
    $("authRow").style.display = state.auth_required ? "flex" : "none";
    if (state.pending_question) {
      showQuestion(state.pending_question);
    } else {
      hideQuestion();
    }
    const q = (state.queued || []).length;
    $("queued").textContent = q
      ? q + " run(s) queued — they start automatically after this one." : "";
    if (state.busy) {
      $("busy").classList.add("show");
      $("result").style.display = "none";
    } else {
      $("busy").classList.remove("show");
      showResult(state);
    }
    if (state.stage_error) {
      $("stageError").innerHTML =
        '<div class="error">' + state.stage_error + "</div>";
    }
    if (state.error) {
      $("error").innerHTML = '<div class="error">' + state.error + "</div>";
    }
    if (!state.busy && state.run_id) loadHistory();
  }).catch(() => {});
}

function loadHistory() {
  fetch("/api/runs").then(r => r.json()).then(d => {
    $("history").innerHTML = (d.runs || []).map(r =>
      "<tr class=clickable onclick=\"loadRun('" + r.run_id + "')\">" +
      "<td>" + r.run_id + "</td><td>" + badge(r.verdict) + "</td>" +
      "<td>" + (r.score ?? "—") + "</td><td>" +
      new Date(r.modified * 1000).toLocaleString() + "</td></tr>").join("");
  }).catch(() => {});
}

function loadRun(runId) {
  fetch("/api/state?run_id=" + encodeURIComponent(runId)).then(r => r.json())
    .then(state => {
      window._dur = state.durations || {};
      renderStages(state.status || {});
      showResult(state);
      $("busy").classList.remove("show");
    }).catch(() => {});
}

function apiHeaders() {
  const key = $("apikey").value.trim();
  return key ? {"X-API-Key": key} : {};
}

function fetchError(e) {
  // TypeError "Failed to fetch" = the server is unreachable, not a
  // pipeline/model error — those come back as JSON with a message.
  return '<div class="error"><b>Server unreachable.</b> Make sure '
    + 'python app.py is still running in its console, then reload this '
    + 'page and try again. Details: ' + e + '</div>';
}

function upload(file) {
  setRunLocked(true);
  fetch("/api/start?name=" + encodeURIComponent(file.name),
        {method: "POST", body: file, headers: apiHeaders()})
    .then(r => r.json().catch(() => ({ok: false,
      error: "server returned HTTP " + r.status})))
    .then(d => {
      setRunLocked(false);
      if (!d.ok) {
        $("error").innerHTML = '<div class="error">' + d.error + "</div>";
        return;
      }
      $("error").innerHTML = "";
      $("result").style.display = "none";
      window._dur = {};
      renderStages({});
      poll();
      setInterval(() => { if (!document.hidden) poll(); }, 900);
    }).catch(e => { setRunLocked(false);
      $("error").innerHTML = fetchError(e); });
}

function setRunLocked(on) {
  $("run").disabled = on;
  $("demo").disabled = on;
  $("error").innerHTML = on
    ? '<div class="hint">Testing LLM connection…</div>' : "";
}

$("run").addEventListener("click", () => {
  const f = $("file").files[0];
  if (!f) { $("error").innerHTML =
    '<div class="error">Choose a CSV or XLSX file first.</div>'; return; }
  upload(f);
});
$("demo").addEventListener("click", () => {
  setRunLocked(true);
  fetch("/api/demo", {method: "POST", headers: apiHeaders()})
    .then(r => r.json()).then(d => {
      setRunLocked(false);
      if (!d.ok) { $("error").innerHTML =
        '<div class="error">' + d.error + "</div>"; return; }
      $("error").innerHTML = "";
      window._dur = {}; renderStages({}); poll();
      setInterval(() => { if (!document.hidden) poll(); }, 900);
    }).catch(e => { setRunLocked(false);
      $("error").innerHTML = fetchError(e); });
});

renderStages({});
loadHistory();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="insight_forge_app")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT,
                        help="start scanning from this port; if busy, the "
                             "next free port is used")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="auto-start a sample run on launch")
    args = parser.parse_args(argv)

    # Fail fast on config problems instead of inside the run thread.
    cfg = load_config(require_key=False)
    global _ASK_TIMEOUT
    _ASK_TIMEOUT = float(cfg["limits"].get("human_input_timeout_min", 5.0)) * 60
    # Keep the console clean — every question goes to the web UI and
    # framework chatter is silenced (errors still surface in the app).
    for noisy in ("crewai", "litellm", "httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    # All human questions (ingestion, review gate, crew human tool) are
    # answered in the web UI — never on the worker's console stdin.
    from shared.core.business_context import set_interactive_provider
    set_interactive_provider(_web_ask)

    server: Optional[ThreadingHTTPServer] = None
    last_error: Optional[OSError] = None
    for port in range(args.port, args.port + 10):
        try:
            server = ThreadingHTTPServer((args.host, port), Handler)
            break
        except OSError as exc:  # WinError 10013 / EADDRINUSE — port taken
            last_error = exc
    if server is None:
        print(f"ports {args.port}-{args.port + 9} are all in use — "
              f"last error: {last_error}", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Insight Forge app running at {url}  (Ctrl+C to stop)")
    if args.demo:
        result = _start_demo()
        print("Demo pipeline started — watch the stages move."
              if result.get("ok") else f"demo failed: {result.get('error')}")
    elif not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())