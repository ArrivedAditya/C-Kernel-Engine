"""Black-box acceptance preflight must not run model work for stale artifacts."""

import json
import queue
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from server import http_lifecycle_acceptance as probe


def test_wrong_loaded_identity_rejects_before_generation(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(("GET", self.path))
            body = json.dumps({
                "model": "fixture-model",
                "serving_identity": "a" * 64,
                "session_library_sha256": "b" * 64,
                "server_instance_id": "fixture-instance",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            requests.append(("POST", self.path))
            self.send_error(500, "generation must not start")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        script = Path(__file__).resolve().parents[1] / "http_lifecycle_acceptance.py"
        report_path = tmp_path / "report.json"
        result = subprocess.run([
            sys.executable, str(script),
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
            "--model", "fixture-model",
            "--expected-serving-identity", "0" * 64,
            "--expected-session-library-sha256", "b" * 64,
            "--output", str(report_path),
        ], capture_output=True, text=True, timeout=10)
    finally:
        server.shutdown()
        worker.join(timeout=10)
        server.server_close()

    assert result.returncode == 1, result.stderr
    report = json.loads(report_path.read_text())
    assert report["status"] == "fail"
    assert report["cases"] == []
    assert "serving_identity" in report["error"]
    assert requests == [("GET", "/v1/cke/loaded-identity")]


class _Response:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Thread:
    def join(self, **_kwargs):
        pass


def _cancel_fixture(monkeypatch, *, terminal=True, stale_status=409):
    events = queue.Queue()
    events.put(("response.created", {"response": {"id": "resp_test"}}))
    done = threading.Event()
    done.set()
    state = {"events": ["response.created", "response.in_progress"],
             "error": None}
    if terminal:
        state["events"].append("response.cancelled")
    monkeypatch.setattr(probe, "stream_request", lambda *_a, **_kw: (
        events, done, state, _Thread()))
    calls = []

    def fake_request(method, path, **_kwargs):
        calls.append((method, path))
        if method == "GET":
            return _Response(200, {"status": "cancelled"})
        return _Response(200 if calls.count((method, path)) == 1 else stale_status)

    monkeypatch.setattr(probe, "request", fake_request)
    return calls


def test_cancel_probe_rejects_missing_terminal_state(monkeypatch):
    calls = _cancel_fixture(monkeypatch, terminal=False)
    with pytest.raises(RuntimeError, match="cancellation not acknowledged"):
        probe.run_cancel("long-input-timed-cancel", "many tokens", wait_for="response.created")
    assert calls == [("POST", "/responses/resp_test/cancel"),
                     ("GET", "/responses/resp_test")]


def test_cancel_probe_rejects_failed_followup(monkeypatch):
    calls = _cancel_fixture(monkeypatch)
    monkeypatch.setattr(probe, "followup", lambda _label: (_ for _ in ()).throw(
        RuntimeError("follow-up failed: HTTP 500")))
    with pytest.raises(RuntimeError, match="follow-up failed"):
        probe.run_cancel("decode", "prompt", wait_for="response.created")
    assert calls == [("POST", "/responses/resp_test/cancel"),
                     ("GET", "/responses/resp_test")]


def test_cancel_probe_rejects_stale_cancel_success(monkeypatch):
    calls = _cancel_fixture(monkeypatch, stale_status=200)
    monkeypatch.setattr(probe, "followup", lambda _label: {"status": "completed"})
    with pytest.raises(RuntimeError, match="stale cancellation not rejected"):
        probe.run_cancel("decode", "prompt", wait_for="response.created")
    assert calls[-1] == ("POST", "/responses/resp_test/cancel")
    assert len(calls) == 3


@pytest.mark.parametrize("trigger,evidence", [
    ("response.created", "in_progress_event_only_native_phase_unverified"),
    ("response.output_text.delta", "output_delta_observed"),
])
def test_cancel_phase_label_states_its_evidence(monkeypatch, trigger, evidence):
    _cancel_fixture(monkeypatch)
    if trigger == "response.output_text.delta":
        original_stream = probe.stream_request

        def stream_with_delta(*args, **kwargs):
            messages, done, state, thread = original_stream(*args, **kwargs)
            messages.put(("response.output_text.delta", {"delta": "1"}))
            state["events"].insert(2, "response.output_text.delta")
            return messages, done, state, thread

        monkeypatch.setattr(probe, "stream_request", stream_with_delta)
    monkeypatch.setattr(probe, "followup", lambda _label: {"status": "completed"})
    row = probe.run_cancel("phase-probe", "prompt", wait_for=trigger)
    assert row["trigger_event"] == trigger
    assert row["phase_evidence"] == evidence
    assert row["stale_cancel_http_status"] == 409


def test_probe_rejects_server_instance_change_after_cases(tmp_path, monkeypatch):
    identity = {"model": "fixture-model", "serving_identity": "a" * 64,
                "session_library_sha256": "b" * 64,
                "server_instance_id": "first"}
    reads = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/v1/cke/loaded-identity"
            reads.append(1)
            body = json.dumps({**identity, "server_instance_id":
                               "first" if len(reads) == 1 else "second"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setattr(probe, "run_cancel", lambda *a, **kw: {"label": a[0]})
    monkeypatch.setattr(probe, "run_disconnect", lambda *a, **kw: {"label": a[0]})
    monkeypatch.setattr(probe, "run_client_timeout", lambda *a, **kw: {"label": a[0]})
    report_path = tmp_path / "changed-instance.json"
    monkeypatch.setattr(sys, "argv", [
        "http_lifecycle_acceptance.py", "--endpoint",
        f"http://127.0.0.1:{server.server_port}/v1",
        "--model", "fixture-model", "--expected-serving-identity", "a" * 64,
        "--expected-session-library-sha256", "b" * 64,
        "--output", str(report_path),
    ])
    try:
        assert probe.main() == 1
    finally:
        server.shutdown()
        worker.join(timeout=10)
        server.server_close()
    report = json.loads(report_path.read_text())
    assert report["status"] == "fail"
    assert len(report["cases"]) == 4
    assert report["identity_before"]["server_instance_id"] == "first"
    assert report["identity_after"]["server_instance_id"] == "second"
    assert "changed during acceptance" in report["error"]
