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

from server.live import _extract_tool_calls_from_text, create_app, create_live_app_from_run_dir
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
        "protocol": "qwen_code_xml",
        "template_sha256": hashlib.sha256(variant.encode()).hexdigest(),
    }), encoding="utf-8")
    selected_native, variants, contract = load_manifest_templates(tmp_path)
    assert selected_native == native and contract is None
    assert variants == {"tool_use": variant}
    protocol = load_tool_protocol(tmp_path, selected_native, variants)
    session = RecordingSession([
        '<function_calls><invoke name="read_file">'
        '<parameter name="file_path">/tmp/cke-qwen/q</parameter>'
        '</invoke></function_calls>',
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
    assert "XML envelope" in session.prompts[0]
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


def test_qwen_code_xml_rejects_malformed_unknown_and_example_text() -> None:
    valid = ('<function_calls><invoke name="read_file">'
             '<parameter name="file_path">/tmp/a</parameter>'
             '</invoke></function_calls>')
    schema = {"read_file": TOOL["parameters"]}
    calls, code, _ = _extract_tool_calls_from_text(
        valid, {"read_file"}, tool_syntax="qwen_code_xml", tool_parameters=schema,
    )
    assert code is None and json.loads(calls[0]["arguments"]) == {"file_path": "/tmp/a"}
    for corrupted in (
        valid.replace('name="read_file"', 'name="unknown"'),
        valid.replace('name="file_path"', 'name="other"'),
        valid.replace('</invoke>', '<parameter name="file_path">b</parameter></invoke>'),
        valid.replace('</function_calls>', ''),
    ):
        bad, error, _ = _extract_tool_calls_from_text(
            corrupted, {"read_file"}, tool_syntax="qwen_code_xml",
            tool_parameters=schema,
        )
        assert not bad and error in {"unknown", "malformed"}
    example, error, _ = _extract_tool_calls_from_text(
        "For example, JSON like {\"name\":\"read_file\"} is text.",
        {"read_file"}, tool_syntax="qwen_code_xml", tool_parameters=schema,
    )
    assert example == [] and error is None
    native_form = ("I'll read this. <tool_call><function=read_file>"
                   "<parameter=file_path>/tmp/a</parameter></function></tool_call>")
    calls, code, _ = _extract_tool_calls_from_text(
        native_form, {"read_file"}, tool_syntax="qwen_code_xml", tool_parameters=schema,
    )
    assert code is None and json.loads(calls[0]["arguments"]) == {"file_path": "/tmp/a"}


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


def test_blank_selected_jinja_does_not_fall_back_for_simple_request() -> None:
    session = RecordingSession(["should not run"])
    client = TestClient(create_app(
        session, model="qwen-local", chat_template="{% if false %}ignored{% endif %}",
    ))
    result = client.post("/v1/responses", json={
        "model": "qwen-local", "input": "hello",
    })
    assert result.status_code == 422
    assert result.json()["error"]["code"] == "template_render_failed"
    assert not session.prompts


def test_template_loader_preserves_crlf_bytes_and_rejects_empty_sidecar(tmp_path: Path) -> None:
    raw = b"{{ messages[0].content }}\r\n"
    native = tmp_path / "chat_template.jinja"
    native.write_bytes(raw)
    (tmp_path / "tool_protocol.json").write_text(json.dumps({
        "schema": "cke.v8.tool_protocol.v1", "protocol": "qwen_xml",
        "template_sha256": hashlib.sha256(raw).hexdigest(),
    }), encoding="utf-8")
    template, variants, _ = load_manifest_templates(tmp_path)
    assert template.encode("utf-8") == raw
    assert load_tool_protocol(tmp_path, template, variants) == "qwen_xml"
    native.write_bytes(b" \r\n ")
    with pytest.raises(ValueError, match="native chat template is empty"):
        load_manifest_templates(tmp_path)
    native.write_bytes(raw)
    variants_dir = tmp_path / "additional_chat_templates"
    variants_dir.mkdir()
    (variants_dir / "tool_use.jinja").write_bytes(b" ")
    with pytest.raises(ValueError, match="chat template is empty"):
        load_manifest_templates(tmp_path)


def test_xml_string_arguments_preserve_payload_whitespace() -> None:
    schema = {"edit": {"type": "object", "properties": {
        "replacement": {"type": "string"},
    }, "required": ["replacement"], "additionalProperties": False}}
    examples = (
        ("<function_calls><invoke name=\"edit\"><parameter name=\"replacement\">"
         "  indented\n</parameter></invoke></function_calls>", "  indented\n"),
        ("<tool_call><function=edit><parameter=replacement>\n"
         "  indented\n\n</parameter></function></tool_call>", "  indented\n"),
        ("<function_calls><invoke name=\"edit\"><parameter name=\"replacement\">"
         "   </parameter></invoke></function_calls>", "   "),
    )
    for generated, expected in examples:
        calls, code, _ = _extract_tool_calls_from_text(
            generated, {"edit"}, tool_syntax="qwen_code_xml", tool_parameters=schema,
        )
        assert code is None
        assert json.loads(calls[0]["arguments"])["replacement"] == expected


def test_unsupported_schema_and_duplicate_json_keys_fail_closed() -> None:
    union_schema = {"read_file": {"type": "object", "properties": {
        "file_path": {"type": "string"},
        "offset": {"type": ["integer", "null"]},
    }}}
    generated = ("<function_calls><invoke name=\"read_file\">"
                 "<parameter name=\"file_path\">/tmp/a</parameter>"
                 "<parameter name=\"offset\">null</parameter>"
                 "</invoke></function_calls>")
    calls, code, message = _extract_tool_calls_from_text(
        generated, {"read_file"}, tool_syntax="qwen_code_xml",
        tool_parameters=union_schema,
    )
    assert code is None and json.loads(calls[0]["arguments"])["offset"] is None
    integer_call = generated.replace("null</parameter>", "42</parameter>")
    calls, code, _ = _extract_tool_calls_from_text(
        integer_call, {"read_file"}, tool_syntax="qwen_code_xml",
        tool_parameters=union_schema,
    )
    assert code is None and json.loads(calls[0]["arguments"])["offset"] == 42

    unsupported_schema = {"read_file": {"type": "object", "properties": {
        "file_path": {"type": ["string", "unknown"]},
    }}}
    calls, code, message = _extract_tool_calls_from_text(
        generated, {"read_file"}, tool_syntax="qwen_code_xml",
        tool_parameters=unsupported_schema,
    )
    assert calls == [] and code == "malformed" and "unsupported type" in message

    session = RecordingSession(["should not run"])
    client = TestClient(create_app(
        session, model="qwen-local", chat_template="{{ messages[0].content }}",
        tool_protocol="qwen_code_xml",
    ))
    tool = dict(TOOL, parameters=unsupported_schema["read_file"])
    response = client.post("/v1/responses", json={
        "model": "qwen-local", "input": "Read a file", "tools": [tool],
    })
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_tool_schema"
    assert not session.prompts

    good = client.post("/v1/responses", json={
        "model": "qwen-local", "input": "hello",
    })
    assert good.status_code == 200
    assert session.prompts

    duplicate = ('<tool_call>{"name":"read_file","arguments":'
                 '{"file_path":"/tmp/a","file_path":"/tmp/b"}}</tool_call>')
    calls, code, message = _extract_tool_calls_from_text(
        duplicate, {"read_file"}, tool_syntax="tool_call_json",
        tool_parameters={"read_file": TOOL["parameters"]},
    )
    assert calls == [] and code == "malformed" and "duplicate JSON key" in message


def test_qwen_shell_numeric_bounds_are_checked() -> None:
    schema = {"run_shell_command": {"type": "object", "properties": {
        "command": {"type": "string"},
        "timeout": {"type": ["integer", "null"], "minimum": 1, "maximum": 600000},
    }, "required": ["command"]}}
    session = RecordingSession(["ready"])
    client = TestClient(create_app(
        session, model="qwen-local", chat_template="{{ messages[0].content }}",
        tool_protocol="qwen_code_xml",
    ))
    tool = {"type": "function", "name": "run_shell_command", "parameters": schema["run_shell_command"]}
    response = client.post("/v1/responses", json={"model": "qwen-local", "input": "hello", "tools": [tool]})
    assert response.status_code == 200
    assert session.prompts

    for timeout in (0, 600001):
        generated = ("<function_calls><invoke name=\"run_shell_command\">"
                     "<parameter name=\"command\">pwd</parameter>"
                     f"<parameter name=\"timeout\">{timeout}</parameter>"
                     "</invoke></function_calls>")
        calls, code, _ = _extract_tool_calls_from_text(
            generated, {"run_shell_command"}, tool_syntax="qwen_code_xml",
            tool_parameters=schema,
        )
        assert calls == [] and code == "malformed"


def test_qwen_goal_string_bounds_are_checked() -> None:
    schema = {"update_goal": {"type": "object", "properties": {
        "reason": {"type": "string", "minLength": 2, "maxLength": 8},
    }, "required": ["reason"]}}
    session = RecordingSession(["ready"])
    client = TestClient(create_app(
        session, model="qwen-local", chat_template="{{ messages[0].content }}",
        tool_protocol="qwen_code_xml",
    ))
    response = client.post("/v1/responses", json={
        "model": "qwen-local", "input": "hello", "tools": [{
            "type": "function", "name": "update_goal", "parameters": schema["update_goal"],
        }],
    })
    assert response.status_code == 200
    for reason in ("x", "too lengthy"):
        generated = ("<function_calls><invoke name=\"update_goal\">"
                     f"<parameter name=\"reason\">{reason}</parameter>"
                     "</invoke></function_calls>")
        calls, code, _ = _extract_tool_calls_from_text(
            generated, {"update_goal"}, tool_syntax="qwen_code_xml",
            tool_parameters=schema,
        )
        assert calls == [] and code == "malformed"


def test_untemplated_app_requires_explicit_raw_mode() -> None:
    with pytest.raises(ValueError, match="requires a nonempty native Jinja template"):
        create_app(RecordingSession(["unused"]), model="qwen-local")
    create_app(RecordingSession(["raw"]), model="qwen-local", allow_untemplated=True)


def test_run_directory_rejects_missing_template_before_session_open(tmp_path: Path, monkeypatch) -> None:
    from server import session_v8

    monkeypatch.setattr(session_v8.SessionV8, "open", lambda *a, **k: pytest.fail("session opened"))
    with pytest.raises(ValueError, match="normal chat serving requires"):
        create_live_app_from_run_dir(tmp_path)
