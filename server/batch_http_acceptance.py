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


def _digest(response: dict) -> dict:
    value = response.get("output_text", "")
    return {"status": response.get("status"),
            "created_at": response.get("created_at"),
            "completed_at": response.get("completed_at"),
            "text_sha256": hashlib.sha256(value.encode()).hexdigest(),
            "text_bytes": len(value.encode()),
            "output_tokens": response.get("usage", {}).get("output_tokens"),
            "stop_reason": response.get("performance", {}).get("stop_reason"),
            "incomplete_details": response.get("incomplete_details")}


def _preflight(identity: dict, observation: dict, *, model: str,
               serving_sha: str, session_sha: str,
               prefill_strategy: str = "decode_replay_chunked") -> None:
    for key, expected in (("model", model), ("serving_identity", serving_sha),
                          ("session_library_sha256", session_sha)):
        if identity.get(key) != expected:
            raise ValueError(f"loaded {key} does not match required artifact")
    mixed_execution = ("one_decode_plus_one_prefill_phase_row"
                       if prefill_strategy == "prefill_phase_chunked"
                       else "one_decode_plus_one_prefill_row")
    if (identity.get("schema") != "cke.loaded_serving_identity.v1"
            or not identity.get("server_instance_id")
            or identity.get("effective_serving", {}).get("execution_mode") != "experimental_batch2_http"
            or identity["effective_serving"].get("mixed_execution") != mixed_execution
            or identity["effective_serving"].get("prefill_strategy") != prefill_strategy):
        raise ValueError("loaded artifact is not the resolved mixed-row serving path")
    if (observation.get("schema") != "cke.batch2_observation.v1"
            or observation.get("server_instance_id") != identity["server_instance_id"]
            or observation.get("poisoned") is not False
            or any(type(observation.get(key)) is not int or observation[key] != 0
                   for key in ("active_slots", "pending_requests"))
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
    if (body["status"] == "completed" and body.get("performance", {}).get("stop_reason") != "eos"
            or body["status"] == "incomplete"
            and (body.get("performance", {}).get("stop_reason") != "token_limit"
                 or (body.get("incomplete_details") or {}).get("reason") != "max_output_tokens")):
        raise RuntimeError("response terminal reason does not match its status")
    return body


def _check_pair(actual_a: dict, actual_b: dict, baseline_a: dict,
                baseline_b: dict, observed_a: dict, observed_b: dict) -> None:
    aid, bid = actual_a.get("id"), actual_b.get("id")
    if any(observed.get("schema") != "cke.batch2_response_observation.v1"
           or observed.get("response_id") != own
           or observed.get("native_done") is not True
           or observed.get("shared_with", {}).get(peer, 0) <= 0
           or observed.get("mixed_with", {}).get(peer, 0) <= 0
           for observed, own, peer in ((observed_a, aid, bid), (observed_b, bid, aid))):
        raise RuntimeError("two clients completed without an observed mixed generated step")
    if (observed_a["mixed_with"][bid] != observed_b["mixed_with"][aid]
            or observed_a["shared_with"][bid] != observed_b["shared_with"][aid]):
        raise RuntimeError("paired generated-step observations disagree")
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
        self.sequence_numbers: list[int] = []
        self.deltas: list[str] = []
        self.delta_times: list[float] = []
        self.result: dict | None = None
        self.error: str | None = None
        self.response_id: str | None = None
        self.stop = threading.Event()
        self._response: httpx.Response | None = None

        def run() -> None:
            try:
                with httpx.Client(timeout=httpx.Timeout(timeout, connect=10)) as client:
                    with client.stream("POST", endpoint + "/responses", json={
                        "model": model, "input": prompt, "max_output_tokens": limit,
                        "temperature": 0, "stream": True, "store": True,
                    }) as response:
                        self._response = response
                        response.raise_for_status()
                        name = None
                        for line in response.iter_lines():
                            if self.stop.is_set():
                                break
                            if line.startswith("event: "):
                                name = line[7:]
                            elif line.startswith("data: ") and name:
                                payload = json.loads(line[6:])
                                if payload.get("type") != name:
                                    raise RuntimeError(f"SSE type differs from event {name}")
                                self.names.append(name)
                                self.sequence_numbers.append(payload.get("sequence_number"))
                                if name == "response.created":
                                    self.response_id = payload.get("response", {}).get("id")
                                if name == "response.output_text.delta":
                                    delta = payload.get("delta")
                                    if not isinstance(delta, str):
                                        raise RuntimeError("output delta is not text")
                                    self.deltas.append(delta)
                                    self.delta_times.append(time.monotonic())
                                self.events.put((name, payload))
                                if name in TERMINALS:
                                    self.result = payload.get("response")
                                name = None
                        if self.result is None and not self.stop.is_set():
                            raise RuntimeError("stream ended without a terminal response")
            except Exception as exc:
                if not self.stop.is_set():
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
        if (self.names[:2] != ["response.created", "response.in_progress"]
                or sum(name in TERMINALS for name in self.names) != 1
                or self.names[-1] not in TERMINALS
                or self.sequence_numbers != list(range(len(self.sequence_numbers)))
                or self.result.get("status") != self.names[-1].removeprefix("response.")
                or self.result.get("id") != self.response_id
                or (self.result.get("status") == "completed"
                    and (self.result.get("incomplete_details") is not None
                         or self.result.get("performance", {}).get("stop_reason") != "eos"))
                or (self.result.get("status") == "incomplete"
                    and ((self.result.get("incomplete_details") or {}).get("reason") != "max_output_tokens"
                         or self.result.get("performance", {}).get("stop_reason") != "token_limit"))
                or "".join(self.deltas) != self.result.get("output_text", "")):
            raise RuntimeError("stream event ordering or output deltas differ from terminal response")
        return self.result

    def close(self, timeout: float) -> bool:
        self.stop.set()
        try:
            if self._response is not None:
                self._response.close()
        except Exception:
            pass
        self.thread.join(timeout)
        return not self.thread.is_alive()


def _run(args: argparse.Namespace, evidence: dict) -> dict:
    endpoint = args.endpoint.rstrip("/")
    timeout = args.timeout_seconds
    streams: list[_Stream] = []
    failed = True
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=10)) as client:
        try:
            before = _get(client, endpoint, "/cke/loaded-identity")
            counters_before = _get(client, endpoint, "/cke/batch-observation")
            _preflight(before, counters_before, model=args.model,
                       serving_sha=args.expected_serving_identity,
                       session_sha=args.expected_session_library_sha256,
                       prefill_strategy=getattr(args, "expected_prefill_strategy",
                                                "decode_replay_chunked"))

            nonce = args.nonce or secrets.token_hex(4)
            prompt_a = f"Write a long list of words about cedar trees. Marker {nonce}A."
            prompt_b = f"Explain how to draw a blue circle in five steps. Marker {nonce}B."
            evidence["identity"] = before
            evidence["requests"] = {"nonce": nonce, "model": args.model,
                                    "temperature": 0, "store": True,
                                    "max_output_tokens": args.max_output_tokens,
                                    "prompts": {"A": prompt_a, "B": prompt_b}}
            baseline_a = _request(client, endpoint, args.model, prompt_a, args.max_output_tokens)
            baseline_b = _request(client, endpoint, args.model, prompt_b, args.max_output_tokens)
            evidence["baselines"] = {"A": _digest(baseline_a), "B": _digest(baseline_b)}
            counters_before = _get(client, endpoint, "/cke/batch-observation")
            if counters_before["active_slots"] or counters_before["pending_requests"]:
                raise RuntimeError("batch owner became busy before the pair")

            a = _Stream(endpoint, args.model, prompt_a, args.max_output_tokens, timeout)
            streams.append(a)
            a_created = a.wait("response.created", timeout)["response"]["id"]
            a.wait("response.output_text.delta", timeout)
            if a.done.is_set():
                raise RuntimeError("first response finished before second could join")
            b = _Stream(endpoint, args.model, prompt_b, args.max_output_tokens, timeout)
            streams.append(b)
            b_created = b.wait("response.created", timeout)["response"]["id"]
            if a_created == b_created:
                raise RuntimeError("concurrent requests reused one response ID")
            a_result, b_result = a.terminal(timeout), b.terminal(timeout)
            if a_result.get("id") != a_created or b_result.get("id") != b_created:
                raise RuntimeError("terminal response ID differs from its stream owner")
            counters_after_pair = _get(client, endpoint, "/cke/batch-observation")
            observed_a = _get(client, endpoint, f"/cke/batch-observation/{a_created}")
            observed_b = _get(client, endpoint, f"/cke/batch-observation/{b_created}")
            if any(observed.get("server_instance_id") != before["server_instance_id"]
                   for observed in (observed_a, observed_b)):
                raise RuntimeError("paired observation came from another server instance")
            _check_pair(a_result, b_result, baseline_a, baseline_b, observed_a, observed_b)
            evidence["observation_before_pair"] = counters_before
            evidence["observation_after_pair"] = counters_after_pair
            evidence["cases"] = [{"case": "two_streams", "response_ids": [a_created, b_created],
                                  "A": _digest(a_result), "B": _digest(b_result),
                                  "request_observations": [observed_a, observed_b]}]
            evidence["normal_eos"] = {
                "status": "not_observed" if all(x["status"] == "incomplete"
                                                for x in (a_result, b_result)) else "observed",
                "note": "The tested Gemma artifact exhausted the output limit; this run does not certify model-generated EOS."
                        if all(x["status"] == "incomplete" for x in (a_result, b_result)) else None,
            }

            # Cancel C while D remains active, then prove the same process serves another.
            c = _Stream(endpoint, args.model, prompt_a, args.max_output_tokens, timeout)
            streams.append(c)
            c_id = c.wait("response.created", timeout)["response"]["id"]
            c.wait("response.output_text.delta", timeout)
            if c.done.is_set():
                raise RuntimeError("cancel target finished before peer joined")
            d = _Stream(endpoint, args.model, prompt_b, args.max_output_tokens, timeout)
            streams.append(d)
            d_id = d.wait("response.created", timeout)["response"]["id"]
            if c.done.is_set() or d.done.is_set():
                raise RuntimeError("both requests must be active at cancellation")
            overlap = _get(client, endpoint, "/cke/batch-observation")
            if overlap["active_slots"] != 2:
                raise RuntimeError("cancellation requests were not simultaneously admitted")
            cancel_at = time.time()
            cancel_monotonic = time.monotonic()
            cancelled = client.post(endpoint + f"/responses/{c_id}/cancel")
            cancelled.raise_for_status()
            c_result = c.terminal(timeout)
            d.wait("response.output_text.delta", timeout)
            d_result = d.terminal(timeout)
            stored = _get(client, endpoint, f"/responses/{c_id}")
            stale_cancel = client.post(endpoint + f"/responses/{c_id}/cancel")
            if (c_result.get("id") != c_id or c_result.get("status") != "cancelled"
                    or stored.get("status") != "cancelled" or stale_cancel.status_code != 409):
                raise RuntimeError("cancel target did not terminate as cancelled")
            if (d_result.get("id") != d_id or d_result.get("status") != baseline_b["status"]
                    or d_result.get("output_text") != baseline_b["output_text"]):
                raise RuntimeError("peer did not continue independently after cancellation")
            if not any(at > cancel_monotonic for at in d.delta_times):
                raise RuntimeError("peer produced no text after the other request was cancelled")
            evidence["cases"].append({"case": "cancel_peer_and_recover", "cancelled_id": c_id,
                                      "peer_id": d_id, "cancel_requested_at": cancel_at,
                                      "peer_deltas_after_cancel": sum(at > cancel_monotonic
                                                                      for at in d.delta_times),
                                      "cancelled": _digest(c_result), "peer": _digest(d_result)})
            followup = _request(client, endpoint, args.model, prompt_b, args.max_output_tokens)
            if (followup["status"] != baseline_b["status"]
                    or followup["output_text"] != baseline_b["output_text"]):
                raise RuntimeError("follow-up differs from isolated greedy execution")
            evidence["cases"].append({"case": "followup", **_digest(followup)})

            after = _get(client, endpoint, "/cke/loaded-identity")
            counters_after = _get(client, endpoint, "/cke/batch-observation")
            if after != before or counters_after["server_instance_id"] != before["server_instance_id"]:
                raise RuntimeError("loaded artifact or server instance changed during acceptance")
            if (counters_after["poisoned"] or counters_after["active_slots"]
                    or counters_after["pending_requests"]):
                raise RuntimeError("batch owner is poisoned or retained completed work")
            failed = False
            return {
                "schema": "cke.batch2_http_acceptance.v1", "status": "pass",
                "scope": "resolved KV-only chat HTTP/native mixed-row acceptance",
                "identity": before, "observation_before": counters_before,
                "observation_after_pair": counters_after_pair,
                "observation_after": counters_after,
                "unsupported": ["tool_harness", "ordinary_prefill_parity", "batch_N", "speedup"]
                + (["model_generated_normal_eos"]
                   if evidence["normal_eos"]["status"] == "not_observed" else []),
            }
        finally:
            if failed:
                evidence["cleanup"] = []
                for stream in streams:
                    outcome = {"response_id": stream.response_id}
                    if stream.response_id and not stream.done.is_set():
                        try:
                            outcome["cancel_status"] = client.post(
                                endpoint + f"/responses/{stream.response_id}/cancel").status_code
                        except Exception as exc:
                            outcome["cancel_error"] = f"{type(exc).__name__}: {exc}"
                    try:
                        outcome["reader_stopped"] = stream.close(min(timeout, 5.0))
                    except Exception as exc:
                        outcome["reader_error"] = f"{type(exc).__name__}: {exc}"
                    evidence["cleanup"].append(outcome)
                if streams:
                    deadline = time.monotonic() + min(timeout, 10.0)
                    while time.monotonic() < deadline:
                        try:
                            state = _get(client, endpoint, "/cke/batch-observation")
                            if not state["active_slots"] and not state["pending_requests"]:
                                evidence["cleanup_idle"] = True
                                break
                        except Exception as exc:
                            evidence["cleanup_observation_error"] = str(exc)
                            break
                        time.sleep(.05)
                    else:
                        evidence["cleanup_idle"] = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-serving-identity", required=True)
    parser.add_argument("--expected-session-library-sha256", required=True)
    parser.add_argument("--expected-prefill-strategy",
                        choices=("decode_replay_chunked", "prefill_phase_chunked"),
                        default="decode_replay_chunked")
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--nonce", help="Fixed synthetic prompt marker for replay")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {"schema": "cke.batch2_http_acceptance.v1", "status": "fail",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        if not 2 <= args.max_output_tokens <= 256 or not 1 <= args.timeout_seconds <= 600:
            raise ValueError("acceptance requires 2-256 output tokens and a 1-600 second timeout")
        report.update(_run(args, report))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status'].upper()}: {args.output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
