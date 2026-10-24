"""Publisher prompt fixtures for the pinned Nemotron and Cohere chat artifacts.

These check the common renderer and circuit linkage. Real tool execution and
model quality need separate artifact-backed acceptance.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from server.live import _render_with_chat_templates, create_app


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
CASES = {
    "nemotron_h": {
        "fixture": "nemotron_nano_9b_v2_publisher_chat_template.jinja",
        "template_sha": "b7a3a520a4bc1beae6a25d615f9215580df8eea926f488256c2bef74039433c0",
        "hello_sha": "6e71ebeff3a1faf1133804545234c2f54235396f39b0120f70b23a37c83b175a",
        "multi_sha": "2ce674abce017008484fd0f088f55d494f528924a88ccc5af2f3c31c91630b6b",
    },
    "cohere2": {
        "fixture": "cohere_command_r7b_publisher_chat_template.jinja",
        "template_sha": "3732db8c378e5817a7b03183b0d80baaebce13a59b3a947336fd70572ce94ef3",
        "hello_sha": "9c65050030b33e663997425c409418802c4a75e467a221e8c7427a89901c6e9a",
        "multi_sha": "bd96304656f22a9ffadefe63af9602769efbfb1547f6faf4f7e8bd8c796981ba",
    },
}


class _ReadTool:
    def model_dump(self):
        return {"type": "function", "function": {
            "name": "read_file", "description": "Read a file",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]},
        }}


@pytest.mark.parametrize("family", CASES)
def test_circuit_linked_chat_only_profile(family: str) -> None:
    circuit = json.loads((ROOT / f"version/v8/circuits/{family}.json").read_text())
    declaration = circuit["serving"]
    profile = json.loads((ROOT / "version/v8" / declaration["profile_ref"]).read_text())
    variant = profile["variants"][declaration["default_variant"]]
    assert variant["chat"] == {"source": "publisher", "variant": "default"}
    assert variant["output_protocol"] == "none"
    assert circuit["contract"]["tokenizer_contract"]["bos_policy"] == "publisher_template_no_implicit_bos"


@pytest.mark.parametrize("family", CASES)
def test_publisher_chat_prompt_bytes(family: str) -> None:
    case = CASES[family]
    source = (FIXTURES / case["fixture"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == case["template_sha"]
    body = SimpleNamespace(tools=None, instructions=None, tool_choice="auto")
    for label, messages in (
        ("hello", [{"role": "user", "content": "Hello."}]),
        ("multi", [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]),
    ):
        prompt = _render_with_chat_templates(
            source.decode(), None, messages, body,
            renderer_tokens={}, tool_protocol="none",
        )
        assert prompt is not None
        assert hashlib.sha256(prompt.encode()).hexdigest() == case[f"{label}_sha"]


def test_nemotron_publisher_tool_prompt_bytes_without_tool_capability_claim() -> None:
    source = (FIXTURES / CASES["nemotron_h"]["fixture"]).read_text()
    prompt = _render_with_chat_templates(
        source, None, [{"role": "user", "content": "Read x"}],
        SimpleNamespace(tools=[_ReadTool()], instructions=None, tool_choice="auto"),
        renderer_tokens={}, tool_protocol="none",
    )
    assert prompt is not None
    # Pinned publisher AutoTokenizer.apply_chat_template, same tool schema.
    assert hashlib.sha256(prompt.encode()).hexdigest() == "f8b8eb646cce9b61a2adbdd6f36918c599c1d679bd32495f7334469c98e3d25f"


@pytest.mark.parametrize("family", CASES)
@pytest.mark.parametrize("stream", [False, True])
def test_chat_only_profile_rejects_tools_before_native_work(family: str, stream: bool) -> None:
    class Session:
        def generate(self, *args, **kwargs):
            raise AssertionError("generation must not start for undeclared tools")

        def cancel(self):
            pass

    source = (FIXTURES / CASES[family]["fixture"]).read_text()
    client = TestClient(create_app(
        Session(), model=family, chat_template=source,
        tool_protocol="none", renderer_tokens={},
    ))
    result = client.post("/v1/responses", json={
        "model": family, "input": "Read the file", "stream": stream,
        "tools": [{"type": "function", "name": "read_file", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }}],
    })
    assert result.status_code == 501
    assert result.json()["error"]["code"] == "tool_protocol_undeclared"


@pytest.mark.parametrize("family", CASES)
def test_pinned_artifact_token_ids_when_configured(family: str) -> None:
    """Asset-backed lane; an absent artifact is explicitly NOT_TESTED in CI."""
    env_name = "CKE_NEMOTRON_RUN" if family == "nemotron_h" else "CKE_COHERE_R7B_RUN"
    run_dir = os.environ.get(env_name)
    if not run_dir:
        pytest.skip(f"NOT_TESTED: set {env_name} to a freshly built run directory")
    from server.session_v8 import SessionV8

    cases = [
        ([{"role": "user", "content": "Hello."}],
         [10, 6775, 1267, 11, 4019, 1010, 22177, 1626, 11, 102897, 1010, 49250, 2077, 1561]
         if family == "nemotron_h" else
         [255000, 255008, 255001, 255000, 255006, 28339, 21, 255001,
          255000, 255007, 255021]),
    ]
    if family == "nemotron_h":
        cases.append(([
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ], [10, 6775, 1267, 11, 4019, 1010, 1065, 1010, 11, 102897,
            1010, 1066, 1010, 12, 1010, 11, 4019, 1010, 1067, 1010,
            11, 102897, 1010, 49250, 2077, 1561]))
    session = SessionV8.open(Path(run_dir), context_length=2048, num_threads=2)
    try:
        template = (Path(run_dir) / "chat_template.jinja").read_text()
        assert hashlib.sha256(template.encode()).hexdigest() == CASES[family]["template_sha"]
        body = SimpleNamespace(tools=None, instructions=None, tool_choice="auto")
        for messages, expected in cases:
            prompt = _render_with_chat_templates(
                template, None, messages, body, renderer_tokens={}, tool_protocol="none",
            )
            assert prompt is not None
            assert session.encode_ids(prompt) == expected
        if family == "nemotron_h":
            prompt = _render_with_chat_templates(
                template, None, [{"role": "user", "content": "Read x"}],
                SimpleNamespace(tools=[_ReadTool()], instructions=None, tool_choice="auto"),
                renderer_tokens={}, tool_protocol="none",
            )
            assert prompt is not None
            ids = session.encode_ids(prompt)
            assert len(ids) == 223
            assert hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() == "944d17d07971cccbf1542c78b2fe0ce2bca097ce18c9c82f1c66de52b37544c7"
    finally:
        session.close()
