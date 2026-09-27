"""Structured tool-call tests for ck_serve_v8.create_app."""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "version" / "v8" / "scripts"))

from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from ck_serve_v8 import create_app, CK_SESSION_REQUEST_RAW_PROMPT
from server.schemas.streaming import ResponseStreamEvent

# Reuse inline Qwen3 contract from test_live_app
QWEN3_CONTRACT = {
    "name": "qwen3",
    "tool_protocol": "bare_json",
    "turn_prefix": "<|im_start|>{role}\n",
    "turn_suffix": "<|im_end|>\n",
    "assistant_generation_prefix": "<|im_start|>assistant\n",
    "role_labels": {"system": "system", "user": "user", "assistant": "assistant"},
    "system_prompt_mode": "dedicated_turn",
    "system_prompt_separator": "\n\n",
    "default_system_prompt": "",
    "inject_default_system_prompt": False,
    "force_bos_text_if_tokenizer_add_bos_false": "",
    "last_user_prefix": "",
    "last_user_prefix_suppression_markers": ["/no_think"],
    "thinking_mode_default": "visible",
    "assistant_generation_prefix_by_thinking_mode": {
        "visible": "<|im_start|>assistant\n",
        "suppressed": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    },
    "last_user_prefix_by_thinking_mode": {"visible": "", "suppressed": "/no_think\n"},
    "stop_text_markers": ["<|im_end|>"],
}

DUMMY_CHAT_TEMPLATES = {"tool_use": "tool jinja", "default": "default"}
STREAM_EVENT_ADAPTER = TypeAdapter(ResponseStreamEvent)
ROLE_AWARE_TEMPLATE = """
{% for message in messages %}
<{{ message.role }}>{{ message.content }}
{% for call in message.get('tool_calls', []) %}
CALL {{ call.id }} {{ call.function.name }} {{ call.function.arguments | tojson }}
{% endfor %}</{{ message.role }}>
{% endfor %}
"""


class FakeSession:
    def __init__(self, chunks=("hello",), *, timing=None):
        self.chunks = list(chunks)
        self.timing = dict(timing or {})
        self.cancel_called = False
        self.last_flags: int = 0
        self.last_user: str | None = None

    def generate(self, system, user, *, max_tokens, temperature, top_p, on_token, flags=0, stop_on_text=(), stop_at_eos=False):
        self.last_flags = flags
        self.last_user = user
        for i, text in enumerate(self.chunks):
            on_token(i, text)
        result = {"prompt_tokens": 1, "generated_tokens": len(self.chunks), "stop_reason": 1}
        result.update(self.timing)
        return result

    def cancel(self):
        self.cancel_called = True

    def close(self):
        pass


def iter_sse(text):
    events = []
    for block in text.split("\n\n"):
        event = "message"
        data_lines = []
        for line in block.strip("\n").splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: "):])
        if not data_lines:
            continue
        events.append((event, json.loads("\n".join(data_lines))))
    return events


def assert_valid_stream(events):
    assert [payload["sequence_number"] for _, payload in events] == list(
        range(len(events))
    )
    for event, payload in events:
        assert payload["type"] == event
        STREAM_EVENT_ADAPTER.validate_python(payload)


def test_tool_call_single_non_stream():
    session = FakeSession(chunks=('{"name":"get_weather","arguments":{"location":"Paris"}}',))
    client = TestClient(create_app(session, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={
        "model": "fake-model",
        "input": "what is weather?",
        "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    # output should contain function_call
    fc = next(i for i in data["output"] if i["type"] == "function_call")
    assert fc["name"] == "get_weather"
    assert json.loads(fc["arguments"]) == {"location": "Paris"}


def test_dated_web_search_tool_type_reaches_unsupported_policy(monkeypatch):
    # Regression: the OpenAI dated wire type web_search_2025_08_26 must pass
    # schema validation and reach the explicit unsupported-tool 501 policy,
    # not be rejected earlier with a 422 from the discriminated union.
    #
    # The "reject before rendering" claim is made executable: with a chat
    # template configured (rendering would be possible), no Jinja rendering
    # runs and no model generation happens, for stream=false and stream=true
    # alike, and a mixed tool list that pairs a valid function tool with an
    # unsupported web-search tool fails the same way.
    import server.live as live

    class GuardedSession(FakeSession):
        def __init__(self):
            super().__init__(chunks=("must not run",))
            self.generate_calls = 0

        def generate(self, *args, **kwargs):
            self.generate_calls += 1
            raise AssertionError("model generation ran for an unsupported tool request")

    session = GuardedSession()
    render_calls: list[tuple] = []

    def _spy_render(*args, **kwargs):
        render_calls.append((args, kwargs))

    monkeypatch.setattr(live, "_render_with_chat_templates", _spy_render)

    client = TestClient(create_app(
        session,
        model="fake-model",
        chat_template="rendered: {{ messages | length }}",
    ))

    def _assert_unsupported(resp, tool_type):
        assert resp.status_code == 501, (tool_type, resp.text)
        error = resp.json()["error"]
        assert error["code"] == "unsupported_tool_type"
        assert tool_type in error["message"]

    for stream in (False, True):
        for tool_type in ("web_search", "web_search_2025_08_26"):
            _assert_unsupported(
                client.post("/v1/responses", json={
                    "model": "fake-model",
                    "input": "hi",
                    "stream": stream,
                    "tools": [{"type": tool_type, "external_web_access": True}],
                }),
                tool_type,
            )

    # A mixed list with one valid function tool plus one unsupported
    # web-search tool is rejected before generation as well.
    _assert_unsupported(
        client.post("/v1/responses", json={
            "model": "fake-model",
            "input": "hi",
            "tools": [
                {"type": "function", "name": "get_weather",
                 "parameters": {"type": "object",
                                "properties": {"location": {"type": "string"}}}},
                {"type": "web_search_2025_08_26", "external_web_access": True},
            ],
        }),
        "web_search_2025_08_26",
    )

    assert session.generate_calls == 0
    assert render_calls == [], "Jinja rendering must not run for unsupported tool requests"


NATIVE_TOOL_TEMPLATE = """{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}{% if tools %}<tools>{{ tools | tojson }}</tools>{% endif %}<tool_call>{"name": "probe"}</tool_call>"""


def test_native_tagged_tool_call_non_stream():
    session = FakeSession(
        chunks=(
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "server/README.md", "line_end": 20}}\n</tool_call>',
        )
    )
    client = TestClient(
        create_app(
            session,
            model="fake-model",
            chat_contract=QWEN3_CONTRACT,
            chat_templates={"tool_use": NATIVE_TOOL_TEMPLATE},
            tool_protocol="tagged_json",
        )
    )
    response = client.post(
        "/v1/responses",
        json={
            "model": "fake-model",
            "input": "read it",
            "tools": [
                {"type": "function", "name": "read_file", "parameters": {}}
            ],
        },
    )

    assert response.status_code == 200
    call = next(
        item for item in response.json()["output"] if item["type"] == "function_call"
    )
    assert call["name"] == "read_file"
    assert json.loads(call["arguments"]) == {
        "path": "server/README.md",
        "line_end": 20,
    }


def test_qwen_xml_tool_call_and_continuation():
    session = FakeSession(chunks=(
        '<tool_call><function=read_file><parameter=path>server/README.md</parameter></function></tool_call>',
    ))
    client = TestClient(create_app(
        session, model="fake-model", chat_contract=QWEN3_CONTRACT,
        chat_templates={"tool_use": NATIVE_TOOL_TEMPLATE}, tool_protocol="qwen_xml",
    ))
    response = client.post("/v1/responses", json={
        "model": "fake-model", "input": "read it",
        "tools": [{"type": "function", "name": "read_file", "parameters": {}}],
    })
    assert response.status_code == 200
    call = next(item for item in response.json()["output"] if item["type"] == "function_call")
    assert call["name"] == "read_file"
    assert json.loads(call["arguments"]) == {"path": "server/README.md"}
    session.chunks = ["The file was read."]
    followup = client.post("/v1/responses", json={
        "model": "fake-model", "previous_response_id": response.json()["id"],
        "input": [{"type": "function_call_output", "call_id": call["call_id"], "output": "file data"}],
        "tools": [{"type": "function", "name": "read_file", "parameters": {}}],
    })
    assert followup.status_code == 200
    assert "file data" in session.last_user


def test_qwen_xml_rejects_duplicate_parameter():
    from ck_serve_v8 import _extract_tool_calls_from_text

    text = "<tool_call><function=read_file><parameter=path>a</parameter><parameter=path>b</parameter></function></tool_call>"
    calls, code, message = _extract_tool_calls_from_text(text, {"read_file"}, tool_syntax="qwen_xml")
    assert calls == []
    assert code == "malformed"
    assert "duplicate" in message


def test_qwen_xml_respects_declared_parameter_types():
    from ck_serve_v8 import _extract_tool_calls_from_text

    text = (
        "<tool_call><function=inspect>"
        "<parameter=path>123</parameter>"
        "<parameter=line_end>20</parameter>"
        "<parameter=enabled>true</parameter>"
        "<parameter=options>{\"limit\":2}</parameter>"
        "</function></tool_call>"
    )
    schema = {"inspect": {"properties": {
        "path": {"type": "string"}, "line_end": {"type": "integer"},
        "enabled": {"type": "boolean"}, "options": {"type": "object"},
    }}}
    calls, code, _ = _extract_tool_calls_from_text(
        text, {"inspect"}, tool_syntax="qwen_xml", tool_parameters=schema,
    )
    assert code is None
    assert json.loads(calls[0]["arguments"]) == {
        "path": "123", "line_end": 20, "enabled": True, "options": {"limit": 2},
    }
    bad = text.replace("<parameter=line_end>20", "<parameter=line_end>twenty")
    calls, code, _ = _extract_tool_calls_from_text(
        bad, {"inspect"}, tool_syntax="qwen_xml", tool_parameters=schema,
    )
    assert calls == []
    assert code == "malformed"


def test_nonfinite_tool_arguments_are_not_executable():
    from ck_serve_v8 import _extract_tool_calls_from_text

    calls, code, _ = _extract_tool_calls_from_text(
        '{"name":"read_file","arguments":{"line":NaN}}',
        {"read_file"}, tool_syntax="json",
    )
    assert calls == []
    assert code == "malformed"


def test_broken_qwen_xml_delimiter_fails_closed():
    from ck_serve_v8 import _extract_tool_calls_from_text

    calls, code, message = _extract_tool_calls_from_text(
        "<tool_\n>\n<function=\n>\n</function>\n</tool_",
        {"read_file"}, tool_syntax="qwen_xml",
    )
    assert calls == []
    assert code == "malformed"
    assert "delimiter" in message


def test_tool_call_requires_declared_argument_names():
    from ck_serve_v8 import _extract_tool_calls_from_text

    schema = {"read_file": {
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"], "additionalProperties": False,
    }}
    wrong = "<tool_call><function=read_file><parameter=path>README.md</parameter></function></tool_call>"
    calls, code, message = _extract_tool_calls_from_text(
        wrong, {"read_file"}, tool_syntax="qwen_xml", tool_parameters=schema,
    )
    assert calls == []
    assert code == "malformed"
    assert "missing required" in message
    right = wrong.replace("parameter=path", "parameter=file_path")
    calls, code, _ = _extract_tool_calls_from_text(
        right, {"read_file"}, tool_syntax="qwen_xml", tool_parameters=schema,
    )
    assert code is None
    assert json.loads(calls[0]["arguments"]) == {"file_path": "README.md"}

    empty = '<tool_call>{"name":"read_file","arguments":""}</tool_call>'
    calls, code, message = _extract_tool_calls_from_text(
        empty, {"read_file"}, tool_parameters=schema,
    )
    assert calls == []
    assert code == "malformed"
    assert "missing required" in message


def test_legacy_qwen_xml_tags_rejected_as_malformed():
    session = FakeSession(
        chunks=(
            "<tool_call><function=read_file>"
            "<parameter=path>a</parameter><parameter=path>b</parameter>"
            "</function></tool_call>",
        )
    )
    client = TestClient(
        create_app(
            session,
            model="fake-model",
            chat_contract=QWEN3_CONTRACT,
            chat_templates={"tool_use": NATIVE_TOOL_TEMPLATE},
            tool_protocol="tagged_json",
        )
    )
    response = client.post(
        "/v1/responses",
        json={
            "model": "fake-model",
            "input": "read it",
            "tools": [
                {"type": "function", "name": "read_file", "parameters": {}}
            ],
        },
    )

    assert response.json()["status"] == "failed"
    assert "malformed tool call" in response.json()["error"]["message"]


def test_detect_template_tool_syntax():
    from server.live import _detect_template_tool_syntax

    assert (
        _detect_template_tool_syntax("<tool_call>{\"a\": 1}</tool_call>", None)
        == "tool_call_json"
    )
    assert (
        _detect_template_tool_syntax(None, {"tool_use": "x <TOOL_CALL> y"})
        == "tool_call_json"
    )
    assert _detect_template_tool_syntax("plain {{ messages }}", None) == "json"
    assert _detect_template_tool_syntax(None, None) == "json"


def test_json_example_in_assistant_text_is_not_a_tool_call():
    from server.live import _extract_tool_calls_from_text

    example = 'An example is {"name":"read_file","arguments":{"path":"x"}}.'
    assert _extract_tool_calls_from_text(
        example, {"read_file"}, tool_syntax="json"
    ) == ([], None, None)
    assert _extract_tool_calls_from_text(
        '{"name":"read_file","arguments":{"path":"x"}}',
        {"read_file"}, tool_syntax="tool_call_json",
    ) == ([], None, None)


def test_unrelated_template_does_not_declare_tool_support():
    from server.live import _has_tool_support

    assert not _has_tool_support(None, {"default": "plain {{ messages }}"})
    assert not _has_tool_support("this prose mentions tools", None)


def test_tool_call_single_stream():
    session = FakeSession(chunks=('{"name":"get_weather","arguments":{"location":"Paris"}}',))
    client = TestClient(create_app(session, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={
        "model": "fake-model",
        "input": "hi",
        "stream": True,
        "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {}}}],
    })
    assert resp.status_code == 200
    events = iter_sse(resp.text)
    assert_valid_stream(events)
    assert not any(ev == "response.output_text.delta" for ev, _ in events)
    # The delta contains validated function arguments, not the raw envelope.
    deltas = [(ev, p) for ev, p in events if ev == "response.function_call_arguments.delta"]
    assert len(deltas) == 1
    assert json.loads(deltas[0][1]["delta"]) == {"location": "Paris"}
    done = [p for ev, p in events if ev == "response.function_call_arguments.done"]
    assert json.loads(done[0]["arguments"]) == {"location": "Paris"}
    # added -> delta -> done reference the same item.
    added = [p for ev, p in events if ev == "response.output_item.added"]
    assert len(added) == 1
    assert added[0]["item"]["id"] == deltas[0][1]["item_id"] == done[0]["item_id"]
    completed = next(p["response"] for ev, p in events if ev == "response.completed")
    fc = next(i for i in completed["output"] if i["type"] == "function_call")
    assert fc["name"] == "get_weather"
    assert fc["id"] == added[0]["item"]["id"]


def test_tool_call_malformed_failed():
    session = FakeSession(chunks=('{"name":"get_weather","arguments":',))  # malformed JSON
    client = TestClient(create_app(session, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={
        "model": "fake-model",
        "input": "hi",
        "tools": [{"type": "function", "name": "get_weather", "parameters": {}}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert data["error"] is not None
    # streaming also failed
    session2 = FakeSession(chunks=('{"name":"get_weather","arguments":',))
    client2 = TestClient(create_app(session2, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp2 = client2.post("/v1/responses", json={
        "model": "fake-model", "input": "hi", "stream": True,
        "tools": [{"type": "function", "name": "get_weather", "parameters": {}}],
    })
    events = iter_sse(resp2.text)
    assert_valid_stream(events)
    assert any(ev == "response.failed" for ev, _ in events)
    assert any(ev == "error" for ev, _ in events)


def test_tool_call_unknown_tool_failed():
    session = FakeSession(chunks=('{"name":"unknown_tool","arguments":{}}',))
    client = TestClient(create_app(session, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={
        "model": "fake-model", "input": "hi",
        "tools": [{"type": "function", "name": "get_weather", "parameters": {}}],
    })
    assert resp.json()["status"] == "failed"
    assert "unknown tool" in resp.json()["error"]["message"]


def test_tool_call_parallel_false_incomplete():
    session = FakeSession(chunks=('[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]',))
    client = TestClient(create_app(session, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp = client.post("/v1/responses", json={
        "model": "fake-model", "input": "hi",
        "tools": [
            {"type": "function", "name": "a", "parameters": {}},
            {"type": "function", "name": "b", "parameters": {}},
        ],
        "parallel_tool_calls": False,
    })
    data = resp.json()
    assert data["status"] == "incomplete"
    assert data["incomplete_details"]["reason"] == "max_tool_calls"
    fcs = [i for i in data["output"] if i["type"] == "function_call"]
    assert len(fcs) == 1
    assert fcs[0]["name"] == "a"
    # streaming variant
    session2 = FakeSession(chunks=('[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]',))
    client2 = TestClient(create_app(session2, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES))
    resp2 = client2.post("/v1/responses", json={
        "model": "fake-model", "input": "hi", "stream": True,
        "tools": [
            {"type": "function", "name": "a", "parameters": {}},
            {"type": "function", "name": "b", "parameters": {}},
        ],
        "parallel_tool_calls": False,
    })
    events = iter_sse(resp2.text)
    assert_valid_stream(events)
    assert any(ev == "response.incomplete" for ev, _ in events)
    completed = next(p["response"] for ev, p in events if ev == "response.incomplete")
    assert len([i for i in completed["output"] if i["type"] == "function_call"]) == 1


def test_assistant_message_without_tool_calls_renders():
    # Regression: a plain assistant-role message in `input` must not 422.
    # Native templates read `message.tool_calls` unguarded, which raised
    # under StrictUndefined when the key was missing.
    session = FakeSession(chunks=("done",))
    client = TestClient(
        create_app(
            session,
            model="fake-model",
            chat_contract=QWEN3_CONTRACT,
            chat_templates={"tool_use": NATIVE_TOOL_TEMPLATE},
        )
    )
    resp = client.post(
        "/v1/responses",
        json={
            "model": "fake-model",
            "input": [
                {"type": "message", "role": "assistant", "content": "thinking..."},
                {"type": "message", "role": "user", "content": "hi"},
            ],
            "tools": [
                {"type": "function", "name": "get_weather", "parameters": {}}
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


def test_plain_text_history_renders_followup_turn():
    # Regression: a stored assistant turn without tool calls must render
    # on the follow-up turn (previous_response_id path).
    session = FakeSession(chunks=("plain answer",))
    client = TestClient(
        create_app(
            session,
            model="fake-model",
            chat_contract=QWEN3_CONTRACT,
            chat_templates={"tool_use": NATIVE_TOOL_TEMPLATE},
        )
    )
    tools = [{"type": "function", "name": "get_weather", "parameters": {}}]
    first = client.post(
        "/v1/responses",
        json={"model": "fake-model", "input": "hi", "tools": tools},
    ).json()
    assert first["status"] == "completed"
    assert "function_call" not in [
        i["type"] for i in first["output"]
    ]
    session.chunks = ("follow-up answer",)
    second = client.post(
        "/v1/responses",
        json={
            "model": "fake-model",
            "previous_response_id": first["id"],
            "input": "and then?",
            "tools": tools,
        },
    )
    assert second.status_code == 200
    assert second.json()["status"] == "completed"


def test_function_output_continues_stored_tool_history():
    session = FakeSession(
        chunks=(
            '{"name":"get_weather","arguments":{"location":"Paris"}}',
        )
    )
    client = TestClient(
        create_app(
            session,
            model="fake-model",
            chat_contract=QWEN3_CONTRACT,
            chat_templates={"tool_use": ROLE_AWARE_TEMPLATE},
        )
    )
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object"},
        }
    ]
    first = client.post(
        "/v1/responses",
        json={"model": "fake-model", "input": "Weather?", "tools": tools},
    ).json()
    call = next(item for item in first["output"] if item["type"] == "function_call")

    session.chunks = ["It is sunny."]
    second = client.post(
        "/v1/responses",
        json={
            "model": "fake-model",
            "previous_response_id": first["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "sunny, 22 C",
                }
            ],
            "tools": tools,
        },
    )

    assert second.status_code == 200
    assert "<user>Weather?" in session.last_user
    assert f"CALL {call['call_id']} get_weather" in session.last_user
    assert "<tool>sunny, 22 C" in session.last_user


def test_function_output_rejects_unknown_history_and_call_id():
    session = FakeSession()
    client = TestClient(
        create_app(
            session,
            model="fake-model",
            chat_contract=QWEN3_CONTRACT,
            chat_templates={"tool_use": ROLE_AWARE_TEMPLATE},
        )
    )
    output = {
        "type": "function_call_output",
        "call_id": "call_missing",
        "output": "result",
    }
    missing_response = client.post(
        "/v1/responses",
        json={
            "model": "fake-model",
            "previous_response_id": "resp_missing",
            "input": [output],
        },
    )
    assert missing_response.status_code == 404

    unknown_call = client.post(
        "/v1/responses",
        json={"model": "fake-model", "input": [output]},
    )
    assert unknown_call.status_code == 400
    assert "unknown call_id" in unknown_call.json()["detail"]


def test_tool_call_cancellation_stream():
    import concurrent.futures

    class BlockingSession:
        def __init__(self):
            self.cancel_called = False

        def generate(self, system, user, *, max_tokens, temperature, top_p, on_token, flags=0, stop_on_text=(), stop_at_eos=False):
            for i in range(5):
                rc = on_token(i, f'chunk{i} ')
                if rc != 0:
                    break
                time.sleep(0.05)
            return {"prompt_tokens": 1, "generated_tokens": 5, "stop_reason": 1}

        def cancel(self):
            self.cancel_called = True

        def close(self):
            pass

    session = BlockingSession()
    app = create_app(session, model="fake-model", chat_contract=QWEN3_CONTRACT, chat_templates=DUMMY_CHAT_TEMPLATES)
    client = TestClient(app)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut = pool.submit(client.post, "/v1/responses", json={"model": "fake-model", "input": "hi", "stream": True, "tools": [{"type": "function", "name": "a", "parameters": {}}]})
        for _ in range(20):
            with app.state.active_streams_lock:
                if app.state.active_streams:
                    break
            time.sleep(0.05)
        assert app.state.active_streams
        active_id = next(iter(app.state.active_streams))
        resp = client.post(f"/v1/responses/{active_id}/cancel")
        assert resp.status_code == 200
        result = fut.result(timeout=5)
        assert result.status_code == 200
        events = iter_sse(result.text)
        assert_valid_stream(events)
        assert events[-1][0] == "response.cancelled"


def test_tool_call_cancellation_reports_native_failure_and_timeout():
    import concurrent.futures

    class ControlledSession:
        def __init__(self, *, cancel_error=None):
            self.cancel_error = cancel_error
            self.release = threading.Event()

        def generate(
            self,
            system,
            user,
            *,
            max_tokens,
            temperature,
            top_p,
            on_token,
            flags=0,
            stop_on_text=(),
            stop_at_eos=False,
        ):
            self.release.wait(timeout=5)
            return {"prompt_tokens": 1, "generated_tokens": 0, "stop_reason": 3}

        def cancel(self):
            if self.cancel_error is not None:
                raise self.cancel_error

        def close(self):
            pass

    for session, expected_status in (
        (ControlledSession(cancel_error=RuntimeError("cancel failed")), 500),
        (ControlledSession(), 504),
    ):
        app = create_app(
            session,
            model="fake-model",
            chat_contract=QWEN3_CONTRACT,
            chat_templates=DUMMY_CHAT_TEMPLATES,
            cancel_wait_seconds=0.01,
        )
        client = TestClient(app)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(
                client.post,
                "/v1/responses",
                json={"model": "fake-model", "input": "hi", "stream": True},
            )
            for _ in range(50):
                with app.state.active_streams_lock:
                    if app.state.active_streams:
                        break
                time.sleep(0.01)
            assert app.state.active_streams
            response_id = next(iter(app.state.active_streams))
            cancelled = client.post(f"/v1/responses/{response_id}/cancel")
            assert cancelled.status_code == expected_status
            session.release.set()
            future.result(timeout=5)
