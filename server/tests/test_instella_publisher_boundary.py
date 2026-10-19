"""Pinned Instella-Think publisher-template boundary for chat and tools.

The publisher template can serialize tool history, but ignores the supplied
tool definitions. Its presence alone must not advertise executable tools.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from server.live import _render_with_chat_templates, create_app
from server.schemas.tool_definitions import FunctionTool


FIXTURE = Path(__file__).parent / "fixtures/instella_moe_think_publisher_chat_template.jinja"
ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_SHA256 = "3b8267e54b67df6500f231ffa969955cc9d26b3444a291ced4eb96ca7d3ca1ea"
TOKENS = {
    "bos_token": "<｜begin▁of▁sentence｜>",
    "eos_token": "<｜end▁of▁sentence｜>",
}
TOOL = {"type": "function", "name": "read_file", "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}


def _template() -> str:
    source = FIXTURE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == TEMPLATE_SHA256
    return source.decode()


def _render(messages: list[dict], *, tools: bool) -> str:
    body = SimpleNamespace(
        tools=[FunctionTool.model_validate(TOOL)] if tools else None,
        instructions=None, tool_choice="auto",
    )
    result = _render_with_chat_templates(
        _template(), None, messages, body,
        renderer_tokens=TOKENS, tool_protocol="none",
    )
    assert result is not None
    return result


def test_publisher_template_does_not_advertise_supplied_tools() -> None:
    messages = [{"role": "user", "content": "Read x"}]
    expected = "<｜begin▁of▁sentence｜><｜User｜>Read x<｜Assistant｜>"
    assert _render(messages, tools=False) == expected
    assert _render(messages, tools=True) == expected


def test_instella_circuit_keeps_the_publisher_variant_chat_only() -> None:
    circuit = json.loads((ROOT / "version/v8/circuits/instella_moe.json").read_text())
    serving = circuit["serving"]
    profile = json.loads((ROOT / "version/v8" / serving["profile_ref"]).read_text())
    assert profile["variants"][serving["default_variant"]]["output_protocol"] == "none"


def test_publisher_template_tool_history_is_not_complete_single_call_envelope() -> None:
    messages = [
        {"role": "user", "content": "Read x"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": "x"}, separators=(",", ":")),
            },
        }]},
        {"role": "tool", "content": "hello", "tool_call_id": "call_1"},
    ]
    rendered = _render(messages, tools=True)
    assert "<｜tool▁call▁begin｜>function<｜tool▁sep｜>read_file\n```json\n{\"path\":\"x\"}" in rendered
    assert "<｜tool▁output▁begin｜>hello<｜tool▁output▁end｜>" in rendered
    assert "<｜tool▁calls▁end｜>" not in rendered
    assert "call_1" not in rendered


class _Session:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, system, user, *, max_tokens, temperature, top_p, on_token,
                 flags=0, stop_on_text=(), stop_at_eos=False):
        self.prompts.append(user)
        on_token(0, "Hello.")
        return {"prompt_tokens": 5, "generated_tokens": 1, "stop_reason": 1}

    def cancel(self) -> None:
        pass


def test_instella_chat_only_contract_rejects_tools_before_generation() -> None:
    session = _Session()
    client = TestClient(create_app(
        session, model="instella-test", chat_template=_template(),
        tool_protocol="none", renderer_tokens=TOKENS,
    ))
    rejected = client.post("/v1/responses", json={
        "model": "instella-test", "input": "Read x", "tools": [TOOL],
    })
    assert rejected.status_code == 501
    assert rejected.json()["error"]["code"] == "tool_protocol_undeclared"
    assert session.prompts == []
    recovered = client.post("/v1/responses", json={
        "model": "instella-test", "input": "Hello",
    })
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["output_text"] == "Hello."
    assert len(session.prompts) == 1
