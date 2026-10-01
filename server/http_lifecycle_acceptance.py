#!/usr/bin/env python3
"""Black-box cancellation and recovery acceptance for a running CKE endpoint.

This script uses only HTTP and a declared loaded-artifact identity. It does not
import the Python server or inspect its native session, so the same cases can
certify a future server implementation against the same generated runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx


TERMINALS = {"response.completed", "response.incomplete", "response.cancelled", "response.failed"}
BASE = ""
MODEL = ""
HOST = ""
PORT = 0
PREFILL_LINES = 1500


def request(method: str, path: str, *, payload: dict | None = None, timeout: float = 40.0) -> httpx.Response:
    with httpx.Client(timeout=timeout) as client:
        return client.request(method, BASE + path, json=payload)


def stream_request(prompt: str, *, max_tokens: int, disconnect_at: str | None = None,
                   disconnect_delay: float = 0.0):
    messages: queue.Queue = queue.Queue()
    done = threading.Event()
    state: dict = {"events": [], "error": None, "status_code": None}

    def worker() -> None:
        try:
            with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                with client.stream(
                    "POST", BASE + "/responses",
                    json={"model": MODEL, "input": prompt, "max_output_tokens": max_tokens,
                          "stream": True, "store": True},
                ) as response:
                    state["status_code"] = response.status_code
                    response.raise_for_status()
                    event = None
                    for line in response.iter_lines():
                        if line.startswith("event: "):
                            event = line[7:]
                        elif line.startswith("data: ") and event:
                            payload = json.loads(line[6:])
                            state["events"].append(event)
                            messages.put((event, payload))
                            if event == disconnect_at:
                                time.sleep(disconnect_delay)
                                break
                            if event in TERMINALS:
                                break
                            event = None
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            messages.put(("error", state["error"]))
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return messages, done, state, thread


def wait_event(messages: queue.Queue, name: str, *, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event, payload = messages.get(timeout=max(0.01, deadline - time.monotonic()))
        if event == "error":
            raise RuntimeError(payload)
        if event == name:
            return payload
        if event in TERMINALS:
            raise RuntimeError(f"terminal {event} arrived before {name}")
    raise TimeoutError(f"missing {name}")


def response_id_from_created(messages: queue.Queue) -> str:
    payload = wait_event(messages, "response.created")
    return payload["response"]["id"]


def followup(label: str) -> dict:
    start = time.monotonic()
    response = request("POST", "/responses", payload={
        "model": MODEL, "input": "Reply with one word: ready", "max_output_tokens": 2,
    })
    result = {"label": label, "http_status": response.status_code,
              "latency_seconds": round(time.monotonic() - start, 3)}
    if response.status_code == 200:
        body = response.json()
        result.update(status=body.get("status"), output_tokens=body.get("usage", {}).get("output_tokens"))
    else:
        result["error"] = response.text[:300]
    if (response.status_code != 200
            or result.get("status") not in {"completed", "incomplete"}
            or not (result.get("output_tokens") or 0) > 0):
        raise RuntimeError(f"{label}: follow-up failed: {result}")
    return result


def run_cancel(label: str, prompt: str, *, wait_for: str, delay: float = 0.0) -> dict:
    messages, done, state, thread = stream_request(prompt, max_tokens=512)
    response_id = response_id_from_created(messages)
    if wait_for != "response.created":
        wait_event(messages, wait_for, timeout=30.0)
    if delay:
        time.sleep(delay)
    start = time.monotonic()
    cancelled = request("POST", f"/responses/{response_id}/cancel", timeout=20.0)
    ack_latency = round(time.monotonic() - start, 3)
    if not done.wait(25.0):
        raise TimeoutError(f"{label}: stream worker did not finish")
    thread.join(timeout=1.0)
    stored = request("GET", f"/responses/{response_id}")
    row = {"label": label, "response_id": response_id,
           "cancel_http_status": cancelled.status_code,
           "cancel_ack_seconds": ack_latency,
           "stream_events": state["events"][-6:], "stream_error": state["error"],
           "stored_http_status": stored.status_code,
           "stored_status": stored.json().get("status") if stored.status_code == 200 else None}
    if (cancelled.status_code != 200 or stored.status_code != 200
            or row["stored_status"] != "cancelled" or state["error"] is not None
            or "response.cancelled" not in state["events"]):
        raise RuntimeError(f"{label}: cancellation not acknowledged: {row}")
    row["followup"] = followup(label + "-followup")
    stale = request("POST", f"/responses/{response_id}/cancel")
    row["stale_cancel_http_status"] = stale.status_code
    if stale.status_code != 409:
        raise RuntimeError(f"{label}: stale cancellation not rejected: {row}")
    return row


def run_disconnect(label: str, prompt: str) -> dict:
    messages, done, state, thread = stream_request(prompt, max_tokens=512,
                                                   disconnect_at="response.in_progress",
                                                   disconnect_delay=3.0)
    response_id = response_id_from_created(messages)
    if not done.wait(10.0):
        raise TimeoutError(f"{label}: client did not disconnect")
    thread.join(timeout=1.0)
    start = time.monotonic()
    errors = []
    while time.monotonic() - start < 30.0:
        try:
            next_result = followup(label + "-followup")
            stored = request("GET", f"/responses/{response_id}")
            stored_status = stored.json().get("status")
            if stored_status != "cancelled":
                raise RuntimeError(f"{label}: stored status is {stored_status!r}")
            return {"label": label, "response_id": response_id,
                    "stream_events": state["events"], "stream_error": state["error"],
                    "time_to_followup_completion_seconds": round(time.monotonic() - start, 3),
                    "followup": next_result, "stored_status": stored_status,
                    "busy_attempts": len(errors)}
        except RuntimeError as exc:
            errors.append(str(exc))
            time.sleep(0.5)
    raise RuntimeError(f"{label}: no follow-up recovery within 30s: {errors[-2:]}")


def run_client_timeout(label: str, prompt: str) -> dict:
    payload = json.dumps({"model": MODEL, "input": prompt, "max_output_tokens": 512,
                          "stream": True, "store": True}).encode()
    connection = socket.create_connection((HOST, PORT), timeout=30.0)
    try:
        connection.sendall(
            b"POST /v1/responses HTTP/1.1\r\nHost: "
            + f"{HOST}:{PORT}".encode() + b"\r\n"
            b"Content-Type: application/json\r\nContent-Length: "
            + str(len(payload)).encode() + b"\r\n\r\n" + payload
        )
        received = b""
        while b"response.in_progress" not in received:
            chunk = connection.recv(4096)
            if not chunk:
                raise RuntimeError(f"{label}: stream closed before in-progress")
            received += chunk
        match = re.search(rb"resp_[0-9a-f]{24}", received)
        if match is None:
            raise RuntimeError(f"{label}: response id missing")
        response_id = match.group().decode()
        deadline = time.monotonic() + 0.3
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            connection.settimeout(remaining)
            try:
                chunk = connection.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                raise RuntimeError(f"{label}: server closed before the client deadline")
    finally:
        connection.close()
    start = time.monotonic()
    errors = []
    while time.monotonic() - start < 30.0:
        try:
            result = followup(label + "-followup")
            stored = request("GET", f"/responses/{response_id}")
            row = {"label": label, "response_id": response_id, "client_timeout_seconds": 0.3,
                   "time_to_followup_completion_seconds": round(time.monotonic() - start, 3),
                   "followup": result, "stored_status": stored.json().get("status"),
                   "busy_attempts": len(errors)}
            if row["stored_status"] != "cancelled":
                raise RuntimeError(f"{label}: timed-out response did not become cancelled: {row}")
            return row
        except RuntimeError as exc:
            errors.append(str(exc))
            time.sleep(0.5)
    raise RuntimeError(f"{label}: no follow-up recovery within 30s: {errors[-2:]}")


def _check_identity(identity: dict, *, expected_serving: str,
                    expected_session: str, model: str) -> None:
    for key, expected in (
        ("serving_identity", expected_serving),
        ("session_library_sha256", expected_session),
        ("model", model),
    ):
        if identity.get(key) != expected:
            raise RuntimeError(f"loaded identity {key} differs from the required artifact")
    if not identity.get("server_instance_id"):
        raise RuntimeError("loaded identity has no server instance ID")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Local HTTP base URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-serving-identity", required=True)
    parser.add_argument("--expected-session-library-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    parser.add_argument("--prefill-lines", type=int, default=1500)
    args = parser.parse_args()
    parsed = urlparse(args.endpoint.rstrip("/"))
    if parsed.scheme != "http" or not parsed.hostname or parsed.path != "/v1":
        parser.error("endpoint must be an HTTP URL with /v1 as its path")
    if args.prefill_lines < 1:
        parser.error("prefill-lines must be positive")
    for label, digest in (
        ("expected-serving-identity", args.expected_serving_identity),
        ("expected-session-library-sha256", args.expected_session_library_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            parser.error(f"{label} must be a lowercase SHA-256 digest")
    if args.output.exists():
        parser.error("output report already exists; acceptance evidence is immutable")

    global BASE, MODEL, HOST, PORT, PREFILL_LINES
    BASE = args.endpoint.rstrip("/")
    MODEL = args.model
    HOST = parsed.hostname
    PORT = parsed.port or 80
    PREFILL_LINES = args.prefill_lines

    report = {
        "schema": "cke.http_lifecycle_acceptance.v1",
        "endpoint": BASE, "model": MODEL,
        "expected_serving_identity": args.expected_serving_identity,
        "expected_session_library_sha256": args.expected_session_library_sha256,
        "started_at_unix": time.time(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prefill_lines": PREFILL_LINES,
        "identity_before": None,
        "identity_after": None, "cases": [], "status": "fail",
    }
    try:
        before = request("GET", "/cke/loaded-identity")
        before.raise_for_status()
        report["identity_before"] = before.json()
        _check_identity(report["identity_before"],
                        expected_serving=args.expected_serving_identity,
                        expected_session=args.expected_session_library_sha256,
                        model=MODEL)
        long_prompt = "Test data follows.\n" + "alpha beta gamma delta\n" * PREFILL_LINES
        report["cases"].append(run_cancel(
            "long-prefill", long_prompt, wait_for="response.in_progress", delay=3.0))
        report["cases"].append(run_cancel(
            "decode", "Count upward forever, one number per line, beginning at one.",
            wait_for="response.output_text.delta"))
        report["cases"].append(run_disconnect("stream-disconnect", long_prompt))
        report["cases"].append(run_client_timeout("client-read-timeout", long_prompt))
        after = request("GET", "/cke/loaded-identity")
        after.raise_for_status()
        report["identity_after"] = after.json()
        _check_identity(report["identity_after"],
                        expected_serving=args.expected_serving_identity,
                        expected_session=args.expected_session_library_sha256,
                        model=MODEL)
        if report["identity_after"] != report["identity_before"]:
            raise RuntimeError("loaded identity changed during acceptance run")
        report["status"] = "pass"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = round(time.time() - report["started_at_unix"], 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "status": report["status"],
                      "error": report.get("error")}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
