#!/usr/bin/env python3
"""HTTP-only acceptance for the opt-in two-slot mixed decode path.

This checks a loaded artifact, independent Responses streams, actual mixed
generated steps, cancellation and recovery. It does not certify tool use,
ordinary-prefill parity, answer quality or a throughput improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import secrets
import threading
import time
from pathlib import Path

import httpx


TERMINALS = {"response.completed", "response.incomplete", "response.cancelled", "response.failed"}


def _get(client: httpx.Client, endpoint: str, path: str) -> dict:
    response = client.get(endpoint + path)
    response.raise_for_status()
    return response.json()


def _preflight(identity: dict, observation: dict, *, model: str,
               serving_sha: str, session_sha: str) -> None:
    for key, expected in (("model", model), ("serving_identity", serving_sha),
                          ("session_library_sha256", session_sha)):
        if identity.get(key) != expected:
            raise ValueError(f"loaded {key} does not match required artifact")
    if (identity.get("schema") != "cke.loaded_serving_identity.v1"
            or not identity.get("server_instance_id")
            or identity.get("effective_serving", {}).get("execution_mode") != "experimental_batch2_http"
            or identity["effective_serving"].get("mixed_execution") != "one_decode_plus_one_prefill_row"
            or identity["effective_serving"].get("prefill_strategy") != "decode_replay_chunked"):
        raise ValueError("loaded artifact is not the resolved mixed-row serving path")
    if (observation.get("schema") != "cke.batch2_observation.v1"
            or observation.get("server_instance_id") != identity["server_instance_id"]
            or observation.get("poisoned") is not False
            or any(type(observation.get(key)) is not int or observation[key] < 0
                   for key in ("shared_steps", "mixed_steps"))):
        raise ValueError("mixed-row execution observation is unavailable or invalid")


def _request(client: httpx.Client, endpoint: str, model: str, prompt: str,
             limit: int) -> dict:
    response = client.post(endpoint + "/responses", json={
        "model": model, "input": prompt, "max_output_tokens": limit,
        "temperature": 0, "store": True,
    })
    response.raise_for_status()
    body = response.json()
    if (body.get("status") not in ("completed", "incomplete")
            or not isinstance(body.get("output_text"), str)
            or not body["output_text"]
            or not isinstance(body.get("id"), str)):
        raise RuntimeError(f"invalid response result: {body.get('status')}")
    return body


def _check_pair(actual_a: dict, actual_b: dict, baseline_a: dict,
                baseline_b: dict, before: dict, after: dict) -> None:
    if (after.get("poisoned") is not False
            or after.get("mixed_steps", 0) <= before["mixed_steps"]
            or after.get("shared_steps", 0) <= before["shared_steps"]):
        raise RuntimeError("two clients completed without an observed mixed generated step")
    for label, actual, baseline in (("A", actual_a, baseline_a),
                                    ("B", actual_b, baseline_b)):
        if (actual.get("id") == baseline["id"]
                or actual.get("status") != baseline["status"]
                or actual.get("output_text") != baseline["output_text"]):
            raise RuntimeError(f"{label} differs from isolated greedy execution")


class _Stream:
    def __init__(self, endpoint: str, model: str, prompt: str, limit: int,
                 timeout: float):
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.done = threading.Event()
        self.names: list[str] = []
        self.deltas: list[str] = []
        self.result: dict | None = None
        self.error: str | None = None

        def run() -> None:
            try:
                with httpx.Client(timeout=httpx.Timeout(timeout, connect=10)) as client:
                    with client.stream("POST", endpoint + "/responses", json={
                        "model": model, "input": prompt, "max_output_tokens": limit,
                        "temperature": 0, "stream": True, "store": True,
                    }) as response:
                        response.raise_for_status()
                        name = None
                        for line in response.iter_lines():
                            if line.startswith("event: "):
                                name = line[7:]
                            elif line.startswith("data: ") and name:
                                payload = json.loads(line[6:])
                                if payload.get("type") != name:
                                    raise RuntimeError(f"SSE type differs from event {name}")
                                self.names.append(name)
                                if name == "response.output_text.delta":
                                    delta = payload.get("delta")
                                    if not isinstance(delta, str):
                                        raise RuntimeError("output delta is not text")
                                    self.deltas.append(delta)
                                self.events.put((name, payload))
                                if name in TERMINALS:
                                    self.result = payload.get("response")
                                    break
                                name = None
                        if self.result is None:
                            raise RuntimeError("stream ended without a terminal response")
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.events.put(("error", self.error))
            finally:
                self.done.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def wait(self, wanted: str, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                name, payload = self.events.get(timeout=max(.01, deadline - time.monotonic()))
            except queue.Empty as exc:
                raise TimeoutError(f"missing {wanted}") from exc
            if name == "error" or name in TERMINALS and name != wanted:
                raise RuntimeError(f"stream ended before {wanted}: {name}: {payload}")
            if name == wanted:
                return payload
        raise TimeoutError(f"missing {wanted}")

    def terminal(self, timeout: float) -> dict:
        if not self.done.wait(timeout):
            raise TimeoutError("stream did not terminate")
        self.thread.join(timeout=1)
        if self.error or self.result is None or self.names[-1] not in TERMINALS:
            raise RuntimeError(f"stream failed: {self.error or self.names}")
        if (self.names[0] != "response.created"
                or sum(name in TERMINALS for name in self.names) != 1
                or "".join(self.deltas) != self.result.get("output_text", "")):
            raise RuntimeError("stream event ordering or output deltas differ from terminal response")
        return self.result


def _run(args: argparse.Namespace) -> dict:
    endpoint = args.endpoint.rstrip("/")
    timeout = args.timeout_seconds
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=10)) as client:
        before = _get(client, endpoint, "/cke/loaded-identity")
        counters_before = _get(client, endpoint, "/cke/batch-observation")
        _preflight(before, counters_before, model=args.model,
                   serving_sha=args.expected_serving_identity,
                   session_sha=args.expected_session_library_sha256)

        nonce = secrets.token_hex(4)
        prompt_a = f"Write a long list of words about cedar trees. Marker {nonce}A."
        prompt_b = f"Explain how to draw a blue circle in five steps. Marker {nonce}B."
        baseline_a = _request(client, endpoint, args.model, prompt_a, args.max_output_tokens)
        baseline_b = _request(client, endpoint, args.model, prompt_b, args.max_output_tokens)

        a = _Stream(endpoint, args.model, prompt_a, args.max_output_tokens, timeout)
        a_created = a.wait("response.created", timeout)["response"]["id"]
        a.wait("response.output_text.delta", timeout)
        if a.done.is_set():
            raise RuntimeError("first response finished before second could join")
        b = _Stream(endpoint, args.model, prompt_b, args.max_output_tokens, timeout)
        b_created = b.wait("response.created", timeout)["response"]["id"]
        if a_created == b_created:
            raise RuntimeError("concurrent requests reused one response ID")
        a_result, b_result = a.terminal(timeout), b.terminal(timeout)
        if a_result.get("id") != a_created or b_result.get("id") != b_created:
            raise RuntimeError("terminal response ID differs from its stream owner")
        counters_after_pair = _get(client, endpoint, "/cke/batch-observation")
        _check_pair(a_result, b_result, baseline_a, baseline_b,
                    counters_before, counters_after_pair)

        # Cancel one active owner, then prove the same process serves another.
        c = _Stream(endpoint, args.model, prompt_a, args.max_output_tokens, timeout)
        c_id = c.wait("response.created", timeout)["response"]["id"]
        c.wait("response.output_text.delta", timeout)
        if c.done.is_set():
            raise RuntimeError("cancel target finished before cancellation")
        cancelled = client.post(endpoint + f"/responses/{c_id}/cancel")
        cancelled.raise_for_status()
        c_result = c.terminal(timeout)
        stored = _get(client, endpoint, f"/responses/{c_id}")
        stale_cancel = client.post(endpoint + f"/responses/{c_id}/cancel")
        if (c_result.get("id") != c_id or c_result.get("status") != "cancelled"
                or stored.get("status") != "cancelled" or stale_cancel.status_code != 409):
            raise RuntimeError("cancel target did not terminate as cancelled")
        followup = _request(client, endpoint, args.model, prompt_b, args.max_output_tokens)
        if followup["output_text"] != baseline_b["output_text"]:
            raise RuntimeError("follow-up differs from isolated greedy execution")

        after = _get(client, endpoint, "/cke/loaded-identity")
        counters_after = _get(client, endpoint, "/cke/batch-observation")
        if after != before or counters_after["server_instance_id"] != before["server_instance_id"]:
            raise RuntimeError("loaded artifact or server instance changed during acceptance")
        if (counters_after["poisoned"] or counters_after["active_slots"]
                or counters_after["pending_requests"]):
            raise RuntimeError("batch owner is poisoned or retained completed work")
        return {
            "schema": "cke.batch2_http_acceptance.v1", "status": "pass",
            "scope": "resolved KV-only chat HTTP/native mixed-row acceptance",
            "identity": before, "observation_before": counters_before,
            "observation_after_pair": counters_after_pair,
            "observation_after": counters_after,
            "cases": [
                {"case": "two_streams", "response_ids": [a_created, b_created],
                 "statuses": [a_result["status"], b_result["status"]],
                 "isolated_text_equal": True,
                 "mixed_steps_delta": counters_after_pair["mixed_steps"] - counters_before["mixed_steps"]},
                {"case": "cancel_and_recover", "cancelled_id": c_id,
                 "cancelled_status": c_result["status"],
                 "followup_status": followup["status"], "isolated_text_equal": True},
            ],
            "unsupported": ["tool_harness", "ordinary_prefill_parity", "batch_N", "speedup"],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-serving-identity", required=True)
    parser.add_argument("--expected-session-library-sha256", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {"schema": "cke.batch2_http_acceptance.v1", "status": "fail",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        if not 2 <= args.max_output_tokens <= 256 or not 1 <= args.timeout_seconds <= 600:
            raise ValueError("acceptance requires 2-256 output tokens and a 1-600 second timeout")
        report.update(_run(args))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status'].upper()}: {args.output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
