"""Gemma publisher protocol boundaries without model-generated output assumptions."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from server.gemma_tool_protocol import GemmaToolSyntaxError, parse_gemma_tool_calls
from server.live import _extract_tool_calls_from_text, _render_with_chat_templates, _validate_tool_schema_subset, create_app
from server.schemas.tool_definitions import FunctionTool


TOOL = {"type": "function", "name": "read_file", "description": "Read file contents",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"], "additionalProperties": False}}
CALL = '<|tool_call>call:read_file{path:<|"|>/tmp/hello.txt<|"|>}<tool_call|><|tool_response>'


@pytest.mark.parametrize("source,expected", [
    (CALL, {"path": "/tmp/hello.txt"}),
    ('<|tool_call>call:read_file{path:<|"|>  a,{b}\n<|"|>}<tool_call|>',
     {"path": "  a,{b}\n"}),
    ('<|tool_call>call:read_file{path:<|"|>α🙂<|"|>,line:3,ok:true,meta:{n:null}}<tool_call|>',
     {"path": "α🙂", "line": 3, "ok": True, "meta": {"n": None}}),
])
def test_gemma_delimited_arguments(source, expected):
    assert parse_gemma_tool_calls(source)[0]["arguments"] == expected


@pytest.mark.parametrize("source", [
    'Example: ' + CALL,
    '```\n' + CALL + '\n```',
    '<|tool_call>call:read_file{path:<|"|>x<|"|>}',
    '<|tool_call>call:read_file{path:<|"|>x<|"|>,path:<|"|>y<|"|>}<tool_call|>',
    '<|tool_call>call:read_file{path:<|"|>x}<tool_call|>',
    CALL + ' extra text',
    '<|tool_response>',
])
def test_gemma_rejects_malformed_or_nonterminal_calls(source):
    with pytest.raises(GemmaToolSyntaxError):
        parse_gemma_tool_calls(source)


@pytest.mark.parametrize("number", ["1e999", "9" * 4400, "2" * 129])
def test_gemma_rejects_nonfinite_or_oversized_numbers_at_any_depth(number):
    for arguments in (f"{{n:{number}}}", f"{{nested:[{{n:{number}}}]}}"):
        with pytest.raises(GemmaToolSyntaxError):
            parse_gemma_tool_calls(
                f"<|tool_call>call:probe{arguments}<tool_call|>"
            )


def test_gemma_multiple_calls_and_declared_name_validation():
    calls = parse_gemma_tool_calls(CALL.replace("<|tool_response>", "") * 2 + "<|tool_response>")
    assert len(calls) == 2
    wrong = CALL.replace("read_file", "run_shell")
    parsed, code, _ = _extract_tool_calls_from_text(
        wrong, {"read_file"}, tool_syntax="gemma4_dsl_v1")
    assert not parsed and code == "unknown"


def test_gemma_template_tool_shape_is_explicitly_protocol_selected():
    template = "{% for tool in tools %}{{ tool.function.name }}:{{ tool.function.parameters.properties.path.type }}{% endfor %}"
    body = SimpleNamespace(tools=[FunctionTool.model_validate(TOOL)], instructions=None, tool_choice=None)
    assert _render_with_chat_templates(
        template, None, [{"role": "user", "content": "hi"}], body,
        tool_protocol="gemma4_dsl_v1") == "read_file:string"


def test_gemma_template_keeps_nested_tool_schema():
    nested = {"type": "function", "name": "probe", "parameters": {
        "type": "object", "properties": {"query": {"type": "object", "properties": {
            "terms": {"type": "array", "items": {"type": "string"}}}}}}}
    template = ("{{ tools[0].function.parameters.properties.query.properties."
                "terms['items'].type }}")
    body = SimpleNamespace(tools=[FunctionTool.model_validate(nested)],
                           instructions=None, tool_choice=None)
    assert _render_with_chat_templates(
        template, None, [{"role": "user", "content": "hi"}], body,
        tool_protocol="gemma4_dsl_v1") == "string"


def test_pinned_json_schema_declaration_is_annotation_only():
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema",
              "type": "object", "properties": {"AbsolutePath": {"type": "string"}},
              "required": ["AbsolutePath"], "additionalProperties": False}
    _validate_tool_schema_subset(schema)
    with pytest.raises(ValueError, match="unsupported \\$schema"):
        _validate_tool_schema_subset({**schema, "$schema": "https://example.invalid/custom"})
    with pytest.raises(ValueError, match="unsupported \\$schema"):
        _validate_tool_schema_subset({"type": "object", "properties": {"path": {
            "type": "string", "$schema": schema["$schema"]}}})


class _Session:
    def __init__(self):
        self.outputs = iter([CALL, "The file says cobalt."])
        self.prompts = []
        self.stops = []

    def generate(self, system, user, *, max_tokens, temperature, top_p, on_token,
                 flags=0, stop_on_text=(), stop_at_eos=False):
        self.prompts.append(user)
        self.stops.append(stop_on_text)
        on_token(0, next(self.outputs))
        return {"prompt_tokens": 1, "generated_tokens": 1, "stop_reason": 1}

    def cancel(self):
        pass


def test_gemma_http_tool_result_continuation():
    session = _Session()
    template = """{% for tool in tools %}{{ tool.function.name }} {% endfor %}
{% for message in messages %}{{ message.role }}:{{ message.content }}
{% for call in message.get('tool_calls', []) %}{{ call.function.name }}={{ call.function.arguments }}{% endfor %}
{% endfor %}"""
    client = TestClient(create_app(session, model="gemma-test", chat_template=template,
                                   tool_protocol="gemma4_dsl_v1",
                                   stop_on_text=["<|tool_response>", "<eos>"]))
    first = client.post("/v1/responses", json={
        "model": "gemma-test", "input": "read it", "tools": [TOOL],
    })
    assert first.status_code == 200, first.text
    call = next(item for item in first.json()["output"] if item["type"] == "function_call")
    assert json.loads(call["arguments"]) == {"path": "/tmp/hello.txt"}
    second = client.post("/v1/responses", json={
        "model": "gemma-test", "previous_response_id": first.json()["id"],
        "input": [{"type": "function_call_output", "call_id": call["call_id"], "output": "cobalt"}],
        "tools": [TOOL],
    })
    assert second.status_code == 200, second.text
    assert second.json()["output_text"] == "The file says cobalt."
    assert "cobalt" in session.prompts[1]
    assert all("<|tool_response>" in stops for stops in session.stops)


def test_gemma_streamed_call_split_at_delimiters():
    class SplitSession(_Session):
        def generate(self, system, user, *, max_tokens, temperature, top_p, on_token,
                     flags=0, stop_on_text=(), stop_at_eos=False):
            self.prompts.append(user)
            self.stops.append(stop_on_text)
            for index, chunk in enumerate(("<|tool_", "call>call:read_file{path:<|\"|>/tmp/hello.txt",
                                           "<|\"|>}<tool_call|><|tool_response>")):
                on_token(index, chunk)
            return {"prompt_tokens": 1, "generated_tokens": 3, "stop_reason": 1}

    session = SplitSession()
    client = TestClient(create_app(session, model="gemma-test", chat_template="{{ messages[-1].content }}",
                                   tool_protocol="gemma4_dsl_v1"))
    response = client.post("/v1/responses", json={
        "model": "gemma-test", "input": "read it", "tools": [TOOL], "stream": True,
    })
    assert response.status_code == 200, response.text
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert any(event["type"] == "response.output_item.done" and
               event.get("item", {}).get("type") == "function_call" for event in events)
    assert any(event["type"] == "response.completed" for event in events)
    assert not any("<|tool_call>" in json.dumps(event) for event in events)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("number", ["1e999", "9" * 4400])
def test_gemma_http_rejects_bad_number_then_serves_valid_call(stream, number):
    bad = f"<|tool_call>call:read_file{{n:{number}}}<tool_call|>"
    session = _Session()
    session.outputs = iter([bad, CALL])
    client = TestClient(create_app(
        session, model="gemma-test", chat_template="{{ messages[-1].content }}",
        tool_protocol="gemma4_dsl_v1",
    ))
    request = {"model": "gemma-test", "input": "read it", "tools": [TOOL],
               "stream": stream}
    first = client.post("/v1/responses", json=request)
    assert first.status_code == 200, first.text
    if stream:
        events = [json.loads(line[6:]) for line in first.text.splitlines()
                  if line.startswith("data: ")]
        assert any(event["type"] == "response.failed" for event in events)
        assert not any(event.get("item", {}).get("type") == "function_call"
                       for event in events)
    else:
        assert first.json()["status"] == "failed"
        assert not any(item["type"] == "function_call"
                       for item in first.json()["output"])
    follow_up = client.post("/v1/responses", json=request)
    assert follow_up.status_code == 200, follow_up.text
    if stream:
        events = [json.loads(line[6:]) for line in follow_up.text.splitlines()
                  if line.startswith("data: ")]
        assert any(event["type"] == "response.output_item.done"
                   and event.get("item", {}).get("type") == "function_call"
                   for event in events)
    else:
        assert follow_up.json()["status"] == "completed"
        assert any(item["type"] == "function_call"
                   for item in follow_up.json()["output"])
