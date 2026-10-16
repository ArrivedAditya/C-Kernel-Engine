#!/usr/bin/env python3
"""Assess a retained Muse Code Responses trace without replaying its tools.

Muse Code's transport trace records requests and SSE events. This evaluator
checks the protocol and task separately; a correct final sentence cannot hide
an extra tool invocation or a lost tool result. Loaded identity is certified
only when matching before/after snapshots and expected hashes are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "cke.muse_code_trace_acceptance.v1"
IDENTITY_SCHEMA = "cke.loaded_serving_identity.v1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _call(item: dict[str, Any]) -> tuple[str, str, str]:
    call_id = item.get("call_id")
    name = item.get("name")
    namespace = item.get("namespace")
    if namespace and isinstance(name, str) and not name.startswith(namespace + "."):
        name = namespace + "." + name
    arguments = item.get("arguments")
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str):
        raise ValueError("malformed function call identity")
    if not isinstance(arguments, str):
        raise ValueError("malformed function call arguments")
    try:
        parsed = json.loads(arguments, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if not isinstance(parsed, dict):
            raise ValueError("function call arguments must be an object")
        normalized = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("malformed function call arguments") from exc
    return call_id, name, normalized


def _reasoning(item: dict[str, Any]) -> str:
    parts = item.get("content") or []
    if not isinstance(parts, list):
        raise ValueError("malformed reasoning content")
    if any(not isinstance(p, dict) or p.get("type") != "reasoning_text"
           or not isinstance(p.get("text"), str) for p in parts):
        raise ValueError("unrenderable reasoning content")
    return "".join(p["text"] for p in parts)


def parse_trace(raw: bytes) -> list[dict[str, Any]]:
    """Parse only the two transport records needed for request-level evidence."""
    groups: list[dict[str, Any]] = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if line.startswith("REQUEST: "):
            try:
                request = json.loads(line[len("REQUEST: "):])
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {number}: malformed request JSON") from exc
            if not isinstance(request, dict):
                raise ValueError(f"line {number}: request is not an object")
            groups.append({"request": request, "events": []})
        elif line.startswith("SSE RESPONSE: event: "):
            if not groups:
                raise ValueError(f"line {number}: event before request")
            header, separator, data = line.partition("\\ndata: ")
            if not separator:
                raise ValueError(f"line {number}: event has no data")
            event_name = header[len("SSE RESPONSE: event: "):]
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {number}: malformed event JSON") from exc
            if not isinstance(event, dict) or event.get("type") != event_name:
                raise ValueError(f"line {number}: event type mismatch")
            groups[-1]["events"].append(event)
    if not groups:
        raise ValueError("trace has no requests")
    return groups


def _identity_verdict(before: Any, after: Any, *, model: str,
                      expected_serving: str | None,
                      expected_session: str | None) -> dict[str, str]:
    for label, identity in (("before", before), ("after", after)):
        if identity is None:
            continue
        if not isinstance(identity, dict) or identity.get("schema") != IDENTITY_SCHEMA:
            return {"status": "fail", "reason": f"malformed_{label}_identity"}
        if identity.get("model") != model:
            return {"status": "fail", "reason": f"{label}_model_mismatch"}
        if expected_serving is not None and identity.get("serving_identity") != expected_serving:
            return {"status": "fail", "reason": f"{label}_serving_identity_mismatch"}
        if expected_session is not None and identity.get("session_library_sha256") != expected_session:
            return {"status": "fail", "reason": f"{label}_session_identity_mismatch"}
    if before is not None and after is not None and before != after:
        return {"status": "fail", "reason": "loaded_identity_changed"}
    if before is None or after is None:
        return {"status": "incomplete", "reason": "before_and_after_identity_required"}
    if expected_serving is None or expected_session is None:
        return {"status": "incomplete", "reason": "expected_identity_hashes_required"}
    return {"status": "pass", "reason": "loaded_identity_stable_and_expected"}


def evaluate(raw: bytes, stdout: str, *, model: str, tool_name: str,
             expected_arguments: dict[str, Any], expected_answer: str,
             expected_tool_calls: int, client_exit_code: int | None,
             identity_before: Any = None, identity_after: Any = None,
             expected_serving: str | None = None,
             expected_session: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    groups = parse_trace(raw)
    responses: list[dict[str, Any] | None] = []
    seen_calls: dict[str, tuple[str, str]] = {}
    seen_outputs: dict[str, str] = {}
    prior_items: list[dict[str, Any]] = []
    expected_args = json.dumps(expected_arguments, sort_keys=True, separators=(",", ":"))
    tool_count = 0
    for index, group in enumerate(groups):
        request = group["request"]
        if request.get("model") != model:
            errors.append(f"request_{index}_model_mismatch")
        items = request.get("input")
        if not isinstance(items, list):
            errors.append(f"request_{index}_input_missing")
            items = []
        events = group["events"]
        terminals = [e for e in events if e.get("type") in {
            "response.completed", "response.failed", "response.incomplete", "response.cancelled"}]
        if (len(terminals) != 1 or terminals[0].get("type") != "response.completed"
                or events[-1] is not terminals[0]):
            errors.append(f"request_{index}_terminal_not_completed_once")
            responses.append(None)
            continue
        response = terminals[0].get("response")
        if not isinstance(response, dict) or response.get("status") != "completed":
            errors.append(f"request_{index}_response_not_completed")
            responses.append(None)
            continue
        output = response.get("output")
        if not isinstance(output, list):
            errors.append(f"request_{index}_output_missing")
            responses.append(None)
            continue
        done_calls = [e.get("item") for e in events
                      if e.get("type") == "response.output_item.done"
                      and isinstance(e.get("item"), dict)
                      and e["item"].get("type") == "function_call"]
        calls = [item for item in output if isinstance(item, dict)
                 and item.get("type") == "function_call"]
        try:
            if sorted(_call(x) for x in done_calls) != sorted(_call(x) for x in calls):
                errors.append(f"request_{index}_stream_call_mismatch")
        except ValueError:
            errors.append(f"request_{index}_malformed_call")
        for item in calls:
            try:
                call_id, name, arguments = _call(item)
            except ValueError:
                errors.append(f"request_{index}_malformed_call")
                continue
            tool_count += 1
            if call_id in seen_calls:
                errors.append("duplicate_call_id")
            seen_calls[call_id] = (name, arguments)
            if name != tool_name or arguments != expected_args:
                errors.append(f"request_{index}_unexpected_tool_or_arguments")
        if index and responses[index - 1] is None:
            errors.append(f"request_{index}_prior_response_missing")
        for item in prior_items:
            if item.get("type") == "reasoning":
                try:
                    text = _reasoning(item)
                    replayed = [x for x in items if isinstance(x, dict)
                                and x.get("type") == "reasoning"
                                and x.get("id") == item.get("id")]
                    matches = len(replayed) == 1 and _reasoning(replayed[0]) == text
                except ValueError:
                    errors.append(f"request_{index}_reasoning_replay_malformed")
                    continue
                if not matches:
                    errors.append(f"request_{index}_reasoning_replay_mismatch")
            elif item.get("type") == "function_call":
                try:
                    call_id, name, arguments = _call(item)
                    replayed = [_call(x) for x in items if isinstance(x, dict)
                                and x.get("type") == "function_call"
                                and x.get("call_id") == call_id]
                except ValueError:
                    errors.append(f"request_{index}_call_replay_malformed")
                    continue
                if replayed != [(call_id, name, arguments)]:
                    errors.append(f"request_{index}_call_replay_mismatch")
                results = [x.get("output") for x in items if isinstance(x, dict)
                           and x.get("type") == "function_call_output"
                           and x.get("call_id") == call_id]
                if len(results) != 1 or not isinstance(results[0], str) or not results[0]:
                    errors.append(f"request_{index}_tool_result_missing")
                elif call_id in seen_outputs and seen_outputs[call_id] != results[0]:
                    errors.append(f"request_{index}_tool_result_changed")
                else:
                    seen_outputs[call_id] = results[0]
        for item in items:
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                if item.get("call_id") not in seen_calls:
                    errors.append(f"request_{index}_unknown_tool_result")
        responses.append(response)
        prior_items.extend(x for x in output if isinstance(x, dict))
    completed = [r for r in responses if r is not None]
    if not completed:
        errors.append("no_completed_response")
    elif responses[-1] is not None:
        final_output = responses[-1].get("output", [])
        if any(isinstance(x, dict) and x.get("type") == "function_call"
               for x in final_output):
            errors.append("last_response_is_tool_call")
        if not any(isinstance(x, dict) and x.get("type") == "message"
                   for x in final_output):
            errors.append("last_response_has_no_message")
    final_text = responses[-1].get("output_text", "") if responses and responses[-1] else ""
    if not isinstance(final_text, str):
        final_text = ""
    if stdout.strip() != final_text.strip():
        errors.append("client_stdout_final_response_mismatch")
    protocol = {"status": "pass" if not errors else "fail", "errors": sorted(set(errors))}
    if protocol["status"] != "pass":
        task = {"status": "fail", "reason": "protocol_failure"}
    elif client_exit_code is None:
        task = {"status": "incomplete", "reason": "client_exit_code_missing"}
    elif client_exit_code != 0:
        task = {"status": "fail", "reason": "client_exit_nonzero"}
    elif tool_count != expected_tool_calls:
        task = {"status": "fail", "reason": "unexpected_tool_call_count"}
    elif expected_answer not in final_text:
        task = {"status": "fail", "reason": "expected_answer_missing"}
    else:
        task = {"status": "pass", "reason": "answer_and_tool_budget_match"}
    certification = _identity_verdict(
        identity_before, identity_after, model=model,
        expected_serving=expected_serving, expected_session=expected_session,
    )
    return {
        "schema": SCHEMA, "model": model,
        "trace_sha256": _sha256(raw),
        "stdout_sha256": _sha256(stdout.encode("utf-8")),
        "response_count": len(completed), "tool_call_count": tool_count,
        "max_observed_prompt_tokens": max(
            ((r.get("usage") or {}).get("input_tokens", 0) for r in completed), default=0),
        "evidence_scope": "transport_trace_only; tool effects and generated-library execution not independently verified",
        "protocol": protocol, "task": task, "certification": certification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tool-name", default="muse.bash")
    parser.add_argument("--expected-arguments-json", default='{"command":"pwd"}')
    parser.add_argument("--expected-answer", required=True)
    parser.add_argument("--expected-tool-calls", type=int, default=1)
    parser.add_argument("--client-exit-code", type=int)
    parser.add_argument("--identity-before", type=Path)
    parser.add_argument("--identity-after", type=Path)
    parser.add_argument("--expected-serving-identity")
    parser.add_argument("--expected-session-library-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_tool_calls < 1:
        parser.error("--expected-tool-calls must be positive")
    try:
        expected_arguments = json.loads(args.expected_arguments_json)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --expected-arguments-json: {exc}")
    before = json.loads(args.identity_before.read_text()) if args.identity_before else None
    after = json.loads(args.identity_after.read_text()) if args.identity_after else None
    report = evaluate(
        args.trace.read_bytes(), args.stdout.read_text(), model=args.model,
        tool_name=args.tool_name, expected_arguments=expected_arguments,
        expected_answer=args.expected_answer,
        expected_tool_calls=args.expected_tool_calls,
        client_exit_code=args.client_exit_code,
        identity_before=before, identity_after=after,
        expected_serving=args.expected_serving_identity,
        expected_session=args.expected_session_library_sha256,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"protocol": report["protocol"], "task": report["task"],
                      "certification": report["certification"]}))
    return 0 if all(report[key]["status"] == "pass" for key in
                    ("protocol", "task", "certification")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
