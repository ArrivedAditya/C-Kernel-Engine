"""Nightly sentinel for the converted Qwen tool template, without a model download.

The fixture is the native tokenizer.chat_template from the pinned Qwen3.8
GGUF used in the Ryzen serving probe (SHA-256 below). These tests exercise the
same sidecar loader and Jinja renderer that the live server uses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.live import create_app
from server.runtime import load_manifest_templates, load_tool_protocol
from server.session_v8 import CK_SESSION_REQUEST_RAW_PROMPT


FIXTURE = Path(__file__).parent / "fixtures" / "qwen38_native_chat_template.jinja"
FIXTURE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
TOOL_VARIANT = Path(__file__).resolve().parents[1] / "templates" / "qwen38_tool_use_compat.jinja"
TOOL = {
    "type": "function",
    "name": "read_file",
    "description": "Read a file by absolute path.",
    "parameters": {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
        "additionalProperties": False,
    },
}


class RecordingSession:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.prompts: list[str] = []
        self.flags: list[int] = []

    def generate(self, system, user, *, max_tokens, temperature, top_p, on_token,
                 flags=0, stop_on_text=(), stop_at_eos=False):
        self.prompts.append(user)
        self.flags.append(flags)
        output = next(self.outputs)
        on_token(1, output)
        return {"prompt_tokens": 1, "generated_tokens": 1, "stop_reason": 1}

    def close(self):
        pass


def _bundle(tmp_path: Path) -> tuple[str, str]:
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    (tmp_path / "chat_template.jinja").write_bytes(raw)
    (tmp_path / "tool_protocol.json").write_text(json.dumps({
        "schema": "cke.v8.tool_protocol.v1",
        "protocol": "qwen_xml",
        "template_sha256": FIXTURE_SHA256,
    }), encoding="utf-8")
    # Embedded templates are intentionally ignored; the converted sidecar wins.
    (tmp_path / "config.json").write_text(
        json.dumps({"chat_template": "HARDCODED_FALLBACK"}), encoding="utf-8"
    )
    template, variants, contract = load_manifest_templates(tmp_path)
    assert variants is None and contract is None and template == raw.decode("utf-8")
    return template, load_tool_protocol(tmp_path, template, variants)


def test_native_jinja_drives_tool_request_and_result_continuation(tmp_path: Path) -> None:
    template, protocol = _bundle(tmp_path)
    session = RecordingSession([
        "<tool_call>\n<function=read_file>\n<parameter=file_path>\n"
        "/tmp/cke-qwen/q\n</parameter>\n</function>\n</tool_call>",
        "CKE_OK",
    ])
    client = TestClient(create_app(
        session, model="qwen-local", chat_template=template, tool_protocol=protocol,
    ))
    first = client.post("/v1/responses", json={
        "model": "qwen-local", "input": "Read /tmp/cke-qwen/q", "tools": [TOOL],
        "temperature": 0, "max_output_tokens": 96,
    })
    assert first.status_code == 200, first.text
    call = next(item for item in first.json()["output"] if item["type"] == "function_call")
    assert call["name"] == "read_file"
    assert json.loads(call["arguments"]) == {"file_path": "/tmp/cke-qwen/q"}
    assert "# Tools" in session.prompts[0]
    assert '"name": "read_file"' in session.prompts[0]
    assert "<tool_call>" in session.prompts[0]
    assert "HARDCODED_FALLBACK" not in session.prompts[0]
    assert session.flags[0] & CK_SESSION_REQUEST_RAW_PROMPT

    second = client.post("/v1/responses", json={
        "model": "qwen-local", "previous_response_id": first.json()["id"],
        "input": [{"type": "function_call_output", "call_id": call["call_id"],
                   "output": "CKE_OK"}],
        "tools": [TOOL], "temperature": 0, "max_output_tokens": 96,
    })
    assert second.status_code == 200, second.text
    assert second.json()["output_text"] == "CKE_OK"
    assert "<function=read_file>" in session.prompts[1]
    assert "<tool_response>\nCKE_OK\n</tool_response>" in session.prompts[1]
    assert session.flags[1] & CK_SESSION_REQUEST_RAW_PROMPT


def test_changed_template_rejects_bound_tool_protocol(tmp_path: Path) -> None:
    template, _ = _bundle(tmp_path)
    changed = template.replace("# Tools", "# Modified tools", 1)
    (tmp_path / "chat_template.jinja").write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="does not match selected chat template"):
        load_tool_protocol(tmp_path, changed, None)


def test_operator_jinja_variant_is_selected_without_inline_tool_marker(tmp_path: Path) -> None:
    native, _ = _bundle(tmp_path)
    variants_dir = tmp_path / "additional_chat_templates"
    variants_dir.mkdir()
    variant = TOOL_VARIANT.read_text(encoding="utf-8")
    (variants_dir / "tool_use.jinja").write_text(variant, encoding="utf-8")
    (tmp_path / "tool_protocol.json").write_text(json.dumps({
        "schema": "cke.v8.tool_protocol.v1",
        "protocol": "qwen_xml",
        "template_sha256": hashlib.sha256(variant.encode()).hexdigest(),
    }), encoding="utf-8")
    selected_native, variants, contract = load_manifest_templates(tmp_path)
    assert selected_native == native and contract is None
    assert variants == {"tool_use": variant}
    protocol = load_tool_protocol(tmp_path, selected_native, variants)
    session = RecordingSession([
        "<tool_call>\n<function=read_file>\n<parameter=file_path>\n"
        "/tmp/cke-qwen/q\n</parameter>\n</function>\n</tool_call>",
        "CKE_OK",
    ])
    client = TestClient(create_app(
        session, model="qwen-local", chat_template=selected_native,
        chat_templates=variants, tool_protocol=protocol,
    ))
    response = client.post("/v1/responses", json={
        "model": "qwen-local", "input": "Read /tmp/cke-qwen/q", "tools": [TOOL],
    })
    assert response.status_code == 200, response.text
    assert any(item["type"] == "function_call" for item in response.json()["output"])
    assert "Qwen XML function call" in session.prompts[0]
    assert "<tool_call>" not in session.prompts[0]
    assert session.flags[0] & CK_SESSION_REQUEST_RAW_PROMPT
    call = next(item for item in response.json()["output"] if item["type"] == "function_call")
    continuation = client.post("/v1/responses", json={
        "model": "qwen-local", "previous_response_id": response.json()["id"],
        "input": [{"type": "function_call_output", "call_id": call["call_id"],
                   "output": "CKE_OK"}],
        "tools": [TOOL],
    })
    assert continuation.status_code == 200, continuation.text
    assert continuation.json()["output_text"] == "CKE_OK"
    assert "Tool result: CKE_OK" in session.prompts[1]
    assert "<tool_call>" not in session.prompts[1]
    assert session.flags[1] & CK_SESSION_REQUEST_RAW_PROMPT
    with pytest.raises(ValueError, match="does not match selected chat template"):
        load_tool_protocol(tmp_path, selected_native, {"tool_use": variant + " changed"})


def test_broken_selected_jinja_fails_before_generation(tmp_path: Path) -> None:
    template, protocol = _bundle(tmp_path)
    session = RecordingSession(["should not run"])
    client = TestClient(create_app(
        session, model="qwen-local", chat_template=template + "{{ missing_feature() }}",
        tool_protocol=protocol,
    ))
    result = client.post("/v1/responses", json={
        "model": "qwen-local", "input": "Read /tmp/cke-qwen/q", "tools": [TOOL],
    })
    assert result.status_code == 422
    assert result.json()["error"]["code"] == "template_render_failed"
    assert not session.prompts
