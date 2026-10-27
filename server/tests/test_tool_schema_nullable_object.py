"""Muse Code's nullable structured-result tool schema stays checked."""

import pytest
from fastapi.testclient import TestClient

from server.live import (
    _tool_value_matches_schema,
    _validate_tool_schema_subset,
    create_app,
)


OUTPUT_SCHEMA = {
    "type": ["object", "null"],
    "description": "Optional structured result",
    "additionalProperties": False,
    "properties": {
        "schema_ref": {"type": "string", "maxLength": 256},
        "required_fields": {
            "type": "array", "items": {"type": "string", "maxLength": 128},
            "maxItems": 16,
        },
    },
    "required": ["schema_ref", "required_fields"],
}
PARAMETERS = {
    "type": "object", "properties": {"output_schema": OUTPUT_SCHEMA},
}


def test_nullable_object_and_array_limit_match_captured_muse_contract():
    _validate_tool_schema_subset(PARAMETERS)
    assert _tool_value_matches_schema({"output_schema": None}, PARAMETERS)
    assert _tool_value_matches_schema({"output_schema": {
        "schema_ref": "test", "required_fields": ["value"] * 16,
    }}, PARAMETERS)
    for invalid in (
        {"schema_ref": "test", "required_fields": ["value"] * 17},
        {"schema_ref": "test", "required_fields": [], "extra": 1},
        {"required_fields": []},
        {"schema_ref": "test", "required_fields": [3]},
    ):
        assert not _tool_value_matches_schema({"output_schema": invalid}, PARAMETERS)


@pytest.mark.parametrize("bad", [True, -1, 1.5, "16"])
def test_invalid_max_items_fails_before_generation(bad):
    schema = {"type": "array", "items": {"type": "string"}, "maxItems": bad}
    with pytest.raises(ValueError, match="maxItems"):
        _validate_tool_schema_subset(schema)


def test_max_items_requires_array_type():
    with pytest.raises(ValueError, match="maxItems"):
        _validate_tool_schema_subset({"type": "string", "maxItems": 16})


@pytest.mark.parametrize("stream", [False, True])
def test_http_admits_nullable_object_before_chat_only_rejection(stream):
    class NoModelWork:
        def generate(self, *args, **kwargs):
            raise AssertionError("chat-only rejection must precede generation")

    client = TestClient(create_app(
        NoModelWork(), model="chat-only", chat_template="{{ messages[-1].content }}",
        tool_protocol=None,
    ))
    response = client.post("/v1/responses", json={
        "model": "chat-only", "input": "Hello", "stream": stream,
        "tools": [{"type": "namespace", "name": "muse", "tools": [{
            "type": "function", "name": "subagent_spawn", "description": "Spawn",
            "parameters": PARAMETERS,
        }]}],
    })
    assert response.status_code == 501, response.text
    assert response.json()["error"]["code"] == "tool_protocol_undeclared"
