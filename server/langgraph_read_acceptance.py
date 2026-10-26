"""Bounded, artifact-bound LangGraph tool round trip against a CKE endpoint.

This is a file-read protocol probe, not a coding-task or numerical certificate.
The optional LangGraph dependencies are needed only for a live run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import urllib.request


PROMPT = "Read note.txt with the shell tool and report its exact value."
SYSTEM = ("/no_think\nYou are a concise coding assistant. Use the shell tool "
          "to read the requested file. Report its exact value. Do not guess.")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assess(messages: list[dict], calls: list[dict], expected: str) -> tuple[bool, str]:
    """Require real execution, correlated result, and a complete final answer."""
    if len(calls) != 1 or calls[0] != {"cmd": "cat note.txt", "stdout": expected}:
        return False, "tool_execution"
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        return False, "malformed_child_output"
    emitted = [call for message in messages for call in message.get("tool_calls", [])]
    if len(emitted) != 1 or emitted[0].get("name") != "exec_bash_command" or \
            emitted[0].get("args") != {"cmd": "cat note.txt"}:
        return False, "tool_request"
    call_id = emitted[0].get("id")
    call_positions = [index for index, message in enumerate(messages)
                      if message.get("tool_calls")]
    result_positions = [index for index, message in enumerate(messages)
                        if message.get("type") == "ToolMessage"]
    if (len(call_positions) != 1 or len(result_positions) != 1
            or messages[call_positions[0]].get("type") != "AIMessage"
            or result_positions[0] != call_positions[0] + 1
            or len(messages) - 1 != result_positions[0] + 1):
        return False, "event_order"
    results = [messages[result_positions[0]]]
    if not call_id or len(results) != 1 or results[0].get("tool_call_id") != call_id:
        return False, "result_correlation"
    try:
        result = json.loads(results[0]["content"])
    except (KeyError, ValueError, TypeError):
        return False, "result_payload"
    if result != {"stdout": expected}:
        return False, "result_payload"
    if not messages or messages[-1].get("type") != "AIMessage" or \
            messages[-1].get("tool_calls") or expected not in messages[-1].get("content", ""):
        return False, "final_answer"
    if messages[-1].get("finish_reason") not in ("stop", "eos"):
        return False, "terminal_state"
    return True, "pass"


def identity(base_url: str) -> dict:
    url = base_url.rstrip("/") + "/cke/loaded-identity"
    with urllib.request.urlopen(url, timeout=15) as response:
        doc = json.load(response)
    if doc.get("schema") != "cke.loaded_serving_identity.v1":
        raise ValueError("endpoint did not return a loaded serving identity")
    return doc


def child(workspace: Path, model: str, base_url: str, output_tokens: int) -> dict:
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    calls: list[dict] = []

    @tool
    def exec_bash_command(cmd: str) -> dict:
        """Read note.txt in the disposable workspace using the allowed command."""
        if cmd != "cat note.txt":
            calls.append({"cmd": cmd, "stdout": None})
            return {"error": "command is not allowed"}
        completed = subprocess.run(
            ["cat", "note.txt"], cwd=workspace, capture_output=True,
            text=True, timeout=10, check=False,
        )
        output = completed.stdout.strip() if completed.returncode == 0 else None
        calls.append({"cmd": cmd, "stdout": output})
        return {"stdout": output} if output is not None else {"error": completed.stderr}

    model_client = ChatOpenAI(
        model=model, base_url=base_url, api_key="cke-local",
        temperature=0, max_tokens=output_tokens,
    )
    agent = create_react_agent(model=model_client, tools=[exec_bash_command], prompt=SYSTEM)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": PROMPT}]},
        config={"recursion_limit": 6},
    )
    messages = [{
        "type": type(item).__name__,
        "content": item.content,
        "tool_calls": getattr(item, "tool_calls", None) or [],
        "tool_call_id": getattr(item, "tool_call_id", None),
        "finish_reason": (getattr(item, "response_metadata", {}) or {}).get("finish_reason"),
    } for item in result["messages"]]
    return {"calls": calls, "messages": messages}


def run(args: argparse.Namespace) -> dict:
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = run_dir / "workspace"
    workspace.mkdir(exist_ok=True)
    value = "cke-" + secrets.token_hex(6)
    (workspace / "note.txt").write_text(value + "\n")
    base = {
        "schema": "cke.langgraph_read_acceptance.v1", "status": "FAIL",
        "scope": "Restricted one-file LangGraph tool/result/answer probe; not edit/test or numerical parity",
        "model": args.model,
        "serving_identity": args.expected_serving_identity,
        "session_library_sha256": args.expected_session_sha256,
        "server_instance_id": None,
        "prompt_sha256": sha256((SYSTEM + "\n" + PROMPT).encode()),
        "fixture_sha256": sha256((value + "\n").encode()),
        "max_output_tokens": args.output_tokens, "attempts": [],
    }
    try:
        first = identity(args.base_url)
    except Exception as exc:
        return {**base, "gate": "identity_unavailable", "error_type": type(exc).__name__}
    if (not isinstance(first.get("server_instance_id"), str)
            or not first["server_instance_id"].strip()):
        return {**base, "gate": "missing_server_instance"}
    if first.get("model") != args.model or \
            first.get("serving_identity") != args.expected_serving_identity or \
            first.get("session_library_sha256") != args.expected_session_sha256:
        return {**base, "gate": "identity_mismatch",
                "server_instance_id": first["server_instance_id"]}
    attempts = []
    for number in range(1, args.attempts + 1):
        command = [sys.executable, __file__, "--child", str(workspace), args.model,
                   args.base_url, str(args.output_tokens)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=args.case_timeout, check=False)
            raw = completed.stdout.strip()
            trace = {}
            if completed.returncode != 0:
                passed, gate = False, "client_error"
            else:
                try:
                    trace = json.loads(raw.rsplit("\n", 1)[-1])
                    if (not isinstance(trace, dict)
                            or not isinstance(trace.get("messages"), list)
                            or not isinstance(trace.get("calls"), list)):
                        raise ValueError("child trace has invalid shape")
                    passed, gate = assess(trace["messages"], trace["calls"], value)
                except (AttributeError, IndexError, KeyError, json.JSONDecodeError,
                        TypeError, ValueError):
                    passed, gate = False, "malformed_child_output"
            attempt = {"number": number, "status": "PASS" if passed else "FAIL",
                       "gate": gate, "trace_sha256": sha256(raw.encode()),
                       "tool_calls": len(trace.get("calls", []))}
            (run_dir / f"attempt-{number}.json").write_text(raw)
            (run_dir / f"attempt-{number}.stderr").write_text(completed.stderr)
        except subprocess.TimeoutExpired:
            attempt = {"number": number, "status": "FAIL", "gate": "deadline",
                       "tool_calls": None}
        try:
            current = identity(args.base_url)
        except Exception:
            attempt.update({"status": "FAIL", "gate": "identity_unavailable",
                            "same_server_instance": False,
                            "same_serving_identity": False,
                            "same_session_library": False})
            attempts.append(attempt)
            break
        attempt["same_server_instance"] = current.get("server_instance_id") == first.get("server_instance_id")
        attempt["same_serving_identity"] = current.get("serving_identity") == first.get("serving_identity")
        attempt["same_session_library"] = current.get("session_library_sha256") == first.get("session_library_sha256")
        if not all((attempt["same_server_instance"], attempt["same_serving_identity"],
                    attempt["same_session_library"])):
            attempt["status"] = "FAIL"
            attempt["gate"] = "identity_changed"
        attempts.append(attempt)
    return {**base, "status": "PASS" if attempts and all(
        item["status"] == "PASS" for item in attempts) else "FAIL",
            "server_instance_id": first["server_instance_id"], "attempts": attempts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--expected-serving-identity")
    parser.add_argument("--expected-session-sha256")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--case-timeout", type=int, default=180)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("child_args", nargs="*")
    args = parser.parse_args()
    if args.child:
        workspace, model, base_url, tokens = args.child_args
        print(json.dumps(child(Path(workspace), model, base_url, int(tokens))))
        return 0
    if not args.base_url or not args.model or not args.run_dir or \
            not args.expected_serving_identity or not args.expected_session_sha256 or \
            args.attempts < 1 or args.output_tokens < 1 or args.case_timeout < 1:
        parser.error("endpoint, model, run directory, expected identities and positive limits are required")
    report = run(args)
    (args.run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
