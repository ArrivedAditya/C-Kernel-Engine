#!/usr/bin/env python3
"""Bounded Antigravity SDK acceptance against one already-running Gemma bundle.

The SDK runs the tool loop. CKE only generates and validates tool calls. This
script restricts all custom tool effects to a fresh disposable workspace and
requires a loaded-identity match before and after the tasks.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import secrets
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


SDK_VERSION = "0.1.20"


def _identity(endpoint: str, model: str, serving_sha: str, session_sha: str) -> dict:
    with urlopen(endpoint + "/cke/loaded-identity", timeout=10) as response:
        document = json.load(response)
    if (document.get("schema") != "cke.loaded_serving_identity.v1"
            or document.get("model") != model
            or document.get("serving_identity") != serving_sha
            or document.get("session_library_sha256") != session_sha
            or document.get("output_protocol") != "gemma4_dsl_v1"
            or document.get("effective_serving", {}).get("request_output_cap") != 256):
        raise ValueError("loaded artifact or effective serving configuration mismatch")
    return document


def _config(endpoint: str, model: str, workspace: Path, tools: list):
    from google.antigravity import LocalOpenAIAgentConfig
    from google.antigravity.types import BuiltinTools, CapabilitiesConfig, BudgetConfig
    from google.antigravity.hooks import policy

    return LocalOpenAIAgentConfig(
        model=model,
        base_url=endpoint,
        workspaces=[str(workspace)],
        system_instructions=(
            "Use only the supplied file tools. Never guess file contents. "
            "Tool responses may include timestamps; report the actual result value, "
            "not only the metadata. Retry a transient tool error once. "
            "Finish only after the task is done."
        ),
        tools=tools,
        capabilities=CapabilitiesConfig(
            enable_subagents=False, enabled_tools=[BuiltinTools.FINISH],
        ),
        policies=[*(policy.allow(tool.__name__) for tool in tools), policy.allow("finish")],
        budget_config=BudgetConfig(
            max_model_calls=6, max_tool_calls=5, max_total_tokens=6000,
        ),
    ).lightweight()


async def _chat(endpoint: str, model: str, workspace: Path, tools: list, prompt: str) -> str:
    from google.antigravity import Agent

    config = _config(endpoint, model, workspace, tools)
    async with Agent(config) as agent:
        response = await agent.chat(prompt)
        parts: list[str] = []
        async for token in response:
            parts.append(str(token))
    return "".join(parts)


async def _one_file(endpoint: str, model: str, root: Path, *, transient: bool) -> dict:
    case = "transient_tool_recovery" if transient else "one_file"
    workspace = root / case
    workspace.mkdir()
    value = ("RECOVER_" if transient else "READ_") + secrets.token_hex(5)
    (workspace / "a.txt").write_text(value)
    calls: list[str] = []

    def read_file(name: str) -> str:
        """Read one named file from the disposable workspace."""
        if name != "a.txt":
            raise ValueError("Only a.txt is available")
        calls.append(name)
        if transient and len(calls) == 1:
            raise RuntimeError("Transient read failure; retry read_file once")
        return (workspace / name).read_text()

    prompt = ("Read a.txt. The tool result contains a JSON `result` field. "
              "Reply with only that field's exact string value, without timestamps.")
    if transient:
        prompt += " If read_file fails transiently, retry once."
    answer = await _chat(endpoint, model, workspace, [read_file], prompt)
    passed = value in answer and len(calls) >= (2 if transient else 1)
    return {"case": case, "status": "pass" if passed else "fail",
            "expected": value, "answer": answer, "tool_calls": calls}


async def _two_files(endpoint: str, model: str, root: Path) -> dict:
    workspace = root / "two_files"
    workspace.mkdir()
    values = {name: prefix + secrets.token_hex(5)
              for name, prefix in (("a.txt", "ALPHA_"), ("b.txt", "BETA_"))}
    for name, value in values.items():
        (workspace / name).write_text(value)
    calls: list[str] = []

    def read_file(name: str) -> str:
        """Read one named file from the disposable workspace."""
        if name not in values:
            raise ValueError("Only a.txt and b.txt are available")
        calls.append(name)
        return (workspace / name).read_text()

    answer = await _chat(endpoint, model, workspace, [read_file],
                         "Read a.txt and b.txt, then report both exact values.")
    passed = all(value in answer for value in values.values()) and set(values) <= set(calls)
    return {"case": "two_files", "status": "pass" if passed else "fail",
            "expected": values, "answer": answer, "tool_calls": calls}


async def _edit_and_validate(endpoint: str, model: str, root: Path) -> dict:
    workspace = root / "edit_and_validate"
    workspace.mkdir()
    nonce = "EDIT_" + secrets.token_hex(5)
    page = workspace / "page.html"
    page.write_text("<!doctype html>\n<html><body><h1>Before</h1></body></html>\n")
    calls: list[dict] = []

    def checked(name: str) -> Path:
        if name != "page.html":
            raise ValueError("Only page.html is available")
        return page

    def read_file(name: str) -> str:
        """Read a named file in the disposable workspace."""
        calls.append({"tool": "read_file", "name": name})
        return checked(name).read_text()

    def edit_file(name: str, old: str, new: str) -> str:
        """Replace one exact substring in a named file."""
        path = checked(name)
        content = path.read_text()
        if content.count(old) != 1:
            raise ValueError("Expected exactly one old substring")
        path.write_text(content.replace(old, new, 1))
        calls.append({"tool": "edit_file", "name": name, "old": old, "new": new})
        return "Edited page.html"

    def validate_file(name: str) -> str:
        """Run the deterministic HTML heading validation."""
        content = checked(name).read_text()
        passed = (f"<h1>After {nonce}</h1>" in content
                  and content.count("<h1>") == 1
                  and content.endswith("</html>\n"))
        calls.append({"tool": "validate_file", "name": name, "passed": passed})
        return "PASS" if passed else "FAIL"

    answer = await _chat(
        endpoint, model, workspace, [read_file, edit_file, validate_file],
        f"Read page.html. Replace <h1>Before</h1> with <h1>After {nonce}</h1>. "
        "Then call validate_file for page.html and summarize the result.",
    )
    final = page.read_text()
    passed = (f"<h1>After {nonce}</h1>" in final
              and "Before" not in final
              and any(call["tool"] == "read_file" for call in calls)
              and any(call["tool"] == "edit_file" for call in calls)
              and any(call["tool"] == "validate_file" and call["passed"] for call in calls)
              and "PASS" in answer)
    return {"case": "edit_and_validate", "status": "pass" if passed else "fail",
            "nonce": nonce, "answer": answer, "tool_calls": calls,
            "final_sha256": hashlib.sha256(final.encode()).hexdigest(), "final_html": final}


async def _run(args) -> int:
    start = time.monotonic()
    identity_before = _identity(args.endpoint, args.model,
                                args.expected_serving_identity,
                                args.expected_session_library_sha256)
    report: dict = {
        "schema": "cke.gemma_antigravity_acceptance.v1",
        "started_at_unix": time.time(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sdk_version": SDK_VERSION,
        "model": args.model,
        "endpoint": args.endpoint,
        "case_timeout_seconds": args.case_timeout_seconds,
        "identity_before": identity_before,
        "cases": [],
        "status": "incomplete",
    }
    try:
        for case, task in (
            ("one_file", lambda: _one_file(args.endpoint, args.model, args.workspace, transient=False)),
            ("two_files", lambda: _two_files(args.endpoint, args.model, args.workspace)),
            ("edit_and_validate", lambda: _edit_and_validate(args.endpoint, args.model, args.workspace)),
            ("transient_tool_recovery", lambda: _one_file(
                args.endpoint, args.model, args.workspace, transient=True)),
        ):
            try:
                result = await asyncio.wait_for(task(), timeout=args.case_timeout_seconds)
            except TimeoutError as exc:
                raise TimeoutError(
                    f"{case} exceeded {args.case_timeout_seconds:g} seconds") from exc
            report["cases"].append(result)
        identity_after = _identity(args.endpoint, args.model,
                                   args.expected_serving_identity,
                                   args.expected_session_library_sha256)
        report["identity_after"] = identity_after
        report["status"] = "pass" if (all(case["status"] == "pass" for case in report["cases"])
                                      and identity_before == identity_after) else "fail"
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = round(time.monotonic() - start, 3)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"report": str(args.output), "status": report["status"],
                      "cases": [case["case"] for case in report["cases"]],
                      "error": report.get("error")}))
    return 0 if report["status"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-serving-identity", required=True)
    parser.add_argument("--expected-session-library-sha256", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    url = urlparse(args.endpoint)
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"} or not args.endpoint.endswith("/v1"):
        parser.error("endpoint must be an explicit localhost HTTP /v1 URL")
    if args.output.exists() or args.workspace.exists():
        parser.error("report and workspace must be new; retained evidence is immutable")
    if not 0 < args.case_timeout_seconds <= 3600:
        parser.error("case timeout must be greater than zero and at most one hour")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.workspace.mkdir(parents=True)
    if importlib.metadata.version("google-antigravity") != SDK_VERSION:
        parser.error(f"google-antigravity=={SDK_VERSION} is required")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
