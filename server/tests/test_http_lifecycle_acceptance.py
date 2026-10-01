"""Black-box acceptance preflight must not run model work for stale artifacts."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


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
