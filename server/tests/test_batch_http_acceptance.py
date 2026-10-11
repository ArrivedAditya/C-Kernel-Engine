"""Negative controls for the HTTP-only mixed-row acceptance gate."""

from __future__ import annotations

import json
import threading
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
    }


@pytest.mark.parametrize("change", [
    lambda identity, observation: identity.update(serving_identity="wrong"),
    lambda identity, observation: identity.update(session_library_sha256="wrong"),
    lambda identity, observation: identity["effective_serving"].update(mixed_execution=None),
    lambda identity, observation: observation.update(server_instance_id="other"),
    lambda identity, observation: observation.update(mixed_steps=True),
])
def test_preflight_fails_closed(change):
    identity, observation = _identity(), _observation()
    change(identity, observation)
    with pytest.raises(ValueError):
        probe._preflight(identity, observation, model="gemma3",
                         serving_sha="a" * 64, session_sha="b" * 64)


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


@pytest.mark.parametrize("fault", ("no_mixed", "swapped_text", "poisoned"))
def test_pair_verdict_rejects_false_concurrency_or_cross_routing(fault):
    baseline_a = {"id": "isolated-a", "status": "incomplete", "output_text": "cedar"}
    baseline_b = {"id": "isolated-b", "status": "incomplete", "output_text": "circle"}
    actual_a = {**baseline_a, "id": "concurrent-a"}
    actual_b = {**baseline_b, "id": "concurrent-b"}
    before = {"shared_steps": 0, "mixed_steps": 0}
    after = {"shared_steps": 2, "mixed_steps": 2, "poisoned": False}
    if fault == "no_mixed":
        after["mixed_steps"] = 0
    elif fault == "swapped_text":
        actual_a["output_text"], actual_b["output_text"] = (
            actual_b["output_text"], actual_a["output_text"])
    else:
        after["poisoned"] = True
    with pytest.raises(RuntimeError):
        probe._check_pair(actual_a, actual_b, baseline_a, baseline_b, before, after)
