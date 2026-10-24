"""Negative controls for the HTTP-only mixed-row acceptance gate."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from server import batch_http_acceptance as probe


def _identity():
    return {
        "schema": "cke.loaded_serving_identity.v1",
        "model": "gemma3",
        "serving_identity": "a" * 64,
        "session_library_sha256": "b" * 64,
        "server_instance_id": "instance-1",
        "effective_serving": {
            "execution_mode": "experimental_batch2_http",
            "mixed_execution": "one_decode_plus_one_prefill_row",
            "prefill_strategy": "decode_replay_chunked",
        },
    }


def _observation():
    return {
        "schema": "cke.batch2_observation.v1", "server_instance_id": "instance-1",
        "shared_steps": 0, "mixed_steps": 0, "poisoned": False,
        "active_slots": 0, "pending_requests": 0,
    }


@pytest.mark.parametrize("change", [
    lambda identity, observation: identity.update(serving_identity="wrong"),
    lambda identity, observation: identity.update(session_library_sha256="wrong"),
    lambda identity, observation: identity["effective_serving"].update(mixed_execution=None),
    lambda identity, observation: observation.update(server_instance_id="other"),
    lambda identity, observation: observation.update(mixed_steps=True),
    lambda identity, observation: observation.update(active_slots=1),
    lambda identity, observation: observation.update(pending_requests=1),
])
def test_preflight_fails_closed(change):
    identity, observation = _identity(), _observation()
    change(identity, observation)
    with pytest.raises(ValueError):
        probe._preflight(identity, observation, model="gemma3",
                         serving_sha="a" * 64, session_sha="b" * 64)


def test_phase_aware_probe_requires_matching_effective_serving_mode():
    identity = _identity()
    with pytest.raises(ValueError, match="mixed-row"):
        probe._preflight(identity, _observation(), model="gemma3",
                         serving_sha="a" * 64, session_sha="b" * 64,
                         prefill_strategy="prefill_phase_chunked")
    identity["effective_serving"].update(
        prefill_strategy="prefill_phase_chunked",
        mixed_execution="one_decode_plus_one_prefill_phase_row")
    probe._preflight(identity, _observation(), model="gemma3",
                     serving_sha="a" * 64, session_sha="b" * 64,
                     prefill_strategy="prefill_phase_chunked")


def test_wrong_session_hash_prevents_generation(tmp_path):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(("GET", self.path))
            if self.path.endswith("loaded-identity"):
                body = _identity()
            elif self.path.endswith("batch-observation"):
                body = _observation()
            else:
                self.send_error(404)
                return
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            calls.append(("POST", self.path))
            self.send_error(500, "generation must not start")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    report = tmp_path / "report.json"
    try:
        status = probe.main([
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
            "--model", "gemma3", "--expected-serving-identity", "a" * 64,
            "--expected-session-library-sha256", "c" * 64,
            "--output", str(report),
        ])
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
    assert status == 1
    assert "session_library_sha256" in json.loads(report.read_text())["error"]
    assert calls == [
        ("GET", "/v1/cke/loaded-identity"),
        ("GET", "/v1/cke/batch-observation"),
    ]


@pytest.mark.parametrize("fault", ("unrelated_global_steps", "swapped_text", "missing_peer"))
def test_pair_verdict_rejects_false_concurrency_or_cross_routing(fault):
    baseline_a = {"id": "isolated-a", "status": "incomplete", "output_text": "cedar"}
    baseline_b = {"id": "isolated-b", "status": "incomplete", "output_text": "circle"}
    actual_a = {**baseline_a, "id": "concurrent-a"}
    actual_b = {**baseline_b, "id": "concurrent-b"}
    observed_a = {"schema": "cke.batch2_response_observation.v1", "response_id": "concurrent-a",
                  "native_done": True, "shared_with": {"concurrent-b": 2},
                  "mixed_with": {"concurrent-b": 2}}
    observed_b = {"schema": "cke.batch2_response_observation.v1", "response_id": "concurrent-b",
                  "native_done": True, "shared_with": {"concurrent-a": 2},
                  "mixed_with": {"concurrent-a": 2}}
    if fault == "unrelated_global_steps":
        # Global counters could rise through an unrelated pair; these two
        # request IDs have no observed shared execution.
        observed_a["mixed_with"] = {}
        observed_b["mixed_with"] = {}
    elif fault == "swapped_text":
        actual_a["output_text"], actual_b["output_text"] = (
            actual_b["output_text"], actual_a["output_text"])
    else:
        observed_b["mixed_with"] = {"unrelated-response": 2}
    with pytest.raises(RuntimeError):
        probe._check_pair(actual_a, actual_b, baseline_a, baseline_b, observed_a, observed_b)


def test_pair_verdict_accepts_attributed_steps():
    baseline_a = {"id": "isolated-a", "status": "completed", "output_text": "cedar"}
    baseline_b = {"id": "isolated-b", "status": "completed", "output_text": "circle"}
    actual_a = {**baseline_a, "id": "concurrent-a"}
    actual_b = {**baseline_b, "id": "concurrent-b"}
    observed_a = {"schema": "cke.batch2_response_observation.v1", "response_id": "concurrent-a",
                  "native_done": True, "shared_with": {"concurrent-b": 1},
                  "mixed_with": {"concurrent-b": 1}}
    observed_b = {"schema": "cke.batch2_response_observation.v1", "response_id": "concurrent-b",
                  "native_done": True, "shared_with": {"concurrent-a": 1},
                  "mixed_with": {"concurrent-a": 1}}
    probe._check_pair(actual_a, actual_b, baseline_a, baseline_b, observed_a, observed_b)


@pytest.mark.parametrize("duplicate", (False, True))
def test_stream_terminal_validates_normal_eos_and_duplicate_terminal(duplicate):
    response_id = "resp_normal_eos"
    events = [
        ("response.created", {"response": {"id": response_id, "status": "in_progress"}}),
        ("response.in_progress", {"response": {"id": response_id, "status": "in_progress"}}),
        ("response.output_text.delta", {"delta": "done"}),
        ("response.completed", {"response": {"id": response_id, "status": "completed",
                                              "output_text": "done", "incomplete_details": None,
                                              "performance": {"stop_reason": "eos"}}}),
    ]
    if duplicate:
        events.append(events[-1])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for seq, (name, payload) in enumerate(events):
                data = {"type": name, "sequence_number": seq, **payload}
                self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        stream = probe._Stream(f"http://127.0.0.1:{server.server_port}/v1", "gemma3", "hello", 8, 5)
        if duplicate:
            with pytest.raises(RuntimeError, match="ordering"):
                stream.terminal(5)
        else:
            assert stream.terminal(5)["status"] == "completed"
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()


def test_failure_after_created_cancels_and_joins_reader(monkeypatch):
    calls = []
    class FakeStream:
        def __init__(self, *_args):
            self.response_id = "resp_started"
            self.done = threading.Event()
        def wait(self, wanted, _timeout):
            if wanted == "response.created":
                return {"response": {"id": self.response_id}}
            raise RuntimeError("injected after creation")
        def close(self, _timeout):
            calls.append("reader_closed")
            return True
    class FakeClient:
        def __init__(self, **_kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def post(self, url):
            calls.append(url)
            return SimpleNamespace(status_code=200)
    monkeypatch.setattr(probe, "_Stream", FakeStream)
    monkeypatch.setattr(probe.httpx, "Client", FakeClient)
    monkeypatch.setattr(probe, "_request", lambda *_args: {
        "id": "baseline", "status": "incomplete", "output_text": "text", "usage": {}})
    monkeypatch.setattr(probe, "_get", lambda _client, _endpoint, path: (
        _identity() if path.endswith("loaded-identity") else _observation()))
    args = SimpleNamespace(endpoint="http://test/v1", timeout_seconds=2, model="gemma3",
                           expected_serving_identity="a" * 64,
                           expected_session_library_sha256="b" * 64,
                           max_output_tokens=8, nonce="fixed")
    evidence = {}
    with pytest.raises(RuntimeError, match="injected after creation"):
        probe._run(args, evidence)
    assert calls == ["http://test/v1/responses/resp_started/cancel", "reader_closed"]
    assert evidence["cleanup_idle"] is True
    assert evidence["cleanup"] == [{"response_id": "resp_started", "cancel_status": 200,
                                     "reader_stopped": True}]
    assert evidence["requests"]["prompts"]["A"].endswith("fixedA.")


def test_socket_failure_after_created_releases_active_request(monkeypatch):
    active = threading.Event()
    cancelled = threading.Event()
    calls = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, body, code=200):
            encoded = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path.endswith("loaded-identity"):
                self._json(_identity())
            elif self.path.endswith("batch-observation"):
                self._json({**_observation(), "active_slots": int(active.is_set())})
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path.endswith("/cancel"):
                calls.append(self.path)
                active.clear()
                cancelled.set()
                self._json({"status": "cancelled"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            if not body.get("stream"):
                self._json({"id": "baseline", "status": "incomplete",
                            "output_text": "text", "usage": {},
                            "performance": {"stop_reason": "token_limit"},
                            "incomplete_details": {"reason": "max_output_tokens"}})
                return
            active.set()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for seq, name in enumerate(("response.created", "response.in_progress")):
                payload = {"type": name, "sequence_number": seq,
                           "response": {"id": "resp_started", "status": "in_progress"}}
                self.wfile.write(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
            cancelled.wait(5)

        def log_message(self, *_args):
            pass

    original_wait = probe._Stream.wait
    def fail_after_created(self, wanted, timeout):
        if wanted == "response.output_text.delta":
            raise RuntimeError("injected after creation")
        return original_wait(self, wanted, timeout)
    monkeypatch.setattr(probe._Stream, "wait", fail_after_created)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    args = SimpleNamespace(endpoint=f"http://127.0.0.1:{server.server_port}/v1",
                           timeout_seconds=5, model="gemma3",
                           expected_serving_identity="a" * 64,
                           expected_session_library_sha256="b" * 64,
                           max_output_tokens=8, nonce="fixed")
    evidence = {}
    try:
        with pytest.raises(RuntimeError, match="injected after creation"):
            probe._run(args, evidence)
        assert calls == ["/v1/responses/resp_started/cancel"]
        assert not active.is_set()
        assert evidence["cleanup_idle"] is True
        assert evidence["cleanup"][0]["reader_stopped"] is True
    finally:
        cancelled.set()
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
