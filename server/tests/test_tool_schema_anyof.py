"""Deep Agents' nullable tool fields stay checked through CKE admission and parsing."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.live import (
    _extract_tool_calls_from_text,
    _tool_value_matches_schema,
    _validate_tool_schema_subset,
    create_app,
)
from version.v8.scripts.ck_chat_completions_v8 import add_chat_completions_route


FIXTURE = Path(__file__).parent / "fixtures/deepagents_0_7_32_tool_parameters.json"
CAPTURED = json.loads(FIXTURE.read_text())
SCHEMAS = CAPTURED["tool_schemas"]


def test_all_captured_deepagents_schemas_are_supported():
    assert CAPTURED["client"] == "deepagents 0.7.32 / langchain-openai 1.7.2"
    assert len(SCHEMAS) == 8
    for name, schema in SCHEMAS.items():
        _validate_tool_schema_subset(schema, f"tool {name!r} parameters")


def test_nullable_paths_and_positive_integer_alternatives():
    path = SCHEMAS["glob"]["properties"]["path"]
    count = SCHEMAS["grep"]["properties"]["max_count"]
    for schema in (path, count):
        _validate_tool_schema_subset(schema)
        assert _tool_value_matches_schema(None, schema)
    assert _tool_value_matches_schema("src", path)
    assert not _tool_value_matches_schema(3, path)
    assert _tool_value_matches_schema(1, count)
    assert not _tool_value_matches_schema(0, count)
    assert not _tool_value_matches_schema("1", count)
    assert not _tool_value_matches_schema(True, count)


@pytest.mark.parametrize("schema", [
    {"anyOf": []},
    {"anyOf": {}},
    {"anyOf": [{"type": "string"}, False]},
    {"anyOf": [{"type": "string"}, {"$ref": "#/definitions/anything"}]},
])
def test_malformed_or_unsupported_alternative_fails_closed(schema):
    with pytest.raises(ValueError):
        _validate_tool_schema_subset(schema)


def test_anyof_conjoins_sibling_constraints():
    schema = {"anyOf": [{"type": "integer"}, {"type": "null"}], "enum": [None, 1]}
    _validate_tool_schema_subset(schema)
    assert _tool_value_matches_schema(None, schema)
    assert _tool_value_matches_schema(1, schema)
    assert not _tool_value_matches_schema(2, schema)


@pytest.mark.parametrize("arguments,code", [
    ({"pattern": "*.py", "path": None}, None),
    ({"pattern": "*.py", "path": "src"}, None),
    ({"pattern": "*.py", "path": 3}, "malformed"),
])
def test_generated_nullable_arguments_are_checked(arguments, code):
    payload = f'<TOOLCALL>[{{"name":"glob","arguments":{json.dumps(arguments)}}}]</TOOLCALL>'
    calls, actual, _ = _extract_tool_calls_from_text(
        payload, {"glob"}, tool_syntax="nemotron_toolcall_v1",
        tool_parameters={"glob": SCHEMAS["glob"]},
    )
    assert actual == code
    assert len(calls) == (0 if code else 1)


@pytest.mark.parametrize("stream", [False, True])
def test_http_accepts_captured_schemas_before_undeclared_protocol(stream):
    class NoModelWork:
        def generate(self, *args, **kwargs):
            raise AssertionError("chat-only rejection must precede generation")

    client = TestClient(create_app(
        NoModelWork(), model="chat-only", chat_template="{{ messages[-1].content }}",
        tool_protocol=None, extra_route_registrar=add_chat_completions_route,
    ))
    tools = [{"type": "function", "function": {"name": name, "parameters": schema}}
             for name, schema in SCHEMAS.items()]
    result = client.post("/v1/chat/completions", json={
        "model": "chat-only", "messages": [{"role": "user", "content": "Hello"}],
        "tools": tools, "stream": stream,
    })
    assert result.status_code == 501
    assert result.json()["error"]["code"] == "tool_protocol_undeclared"
