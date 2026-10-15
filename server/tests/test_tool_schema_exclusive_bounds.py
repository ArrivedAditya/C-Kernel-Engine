"""OpenCode-style numeric tool bounds preserve fail-closed tool admission."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.live import (
    _tool_value_matches_schema,
    _validate_tool_schema_subset,
    create_app,
)
from version.v8.scripts.ck_chat_completions_v8 import add_chat_completions_route


TIMEOUT = {
    "type": "integer",
    "minimum": -9007199254740991,
    "exclusiveMinimum": 0,
    "maximum": 9007199254740991,
}

FIXTURE = Path(__file__).parent / "fixtures" / "opencode_1_19_1_tools.json"


def test_all_captured_opencode_tool_schemas_fit_supported_subset():
    captured = json.loads(FIXTURE.read_text())
    assert captured["client"] == "OpenCode 1.19.1"
    assert captured["route"] == "/v1/chat/completions"
    assert captured["stream"] is True
    assert len(captured["tool_schemas"]) == 10
    for name, schema in captured["tool_schemas"].items():
        _validate_tool_schema_subset(schema, f"tool {name!r} parameters")


def test_exclusive_numeric_bounds_accept_opencode_timeout_and_enforce_it():
    _validate_tool_schema_subset(TIMEOUT)
    assert _tool_value_matches_schema(1, TIMEOUT)
    assert not _tool_value_matches_schema(0, TIMEOUT)
    assert not _tool_value_matches_schema(-1, TIMEOUT)
    assert not _tool_value_matches_schema(9007199254740992, TIMEOUT)

    maximum = {"type": "number", "exclusiveMaximum": 2.5}
    _validate_tool_schema_subset(maximum)
    assert _tool_value_matches_schema(2.4, maximum)
    assert not _tool_value_matches_schema(2.5, maximum)


@pytest.mark.parametrize("bound", [True, float("inf"), "0"])
def test_invalid_exclusive_bounds_rejected(bound):
    with pytest.raises(ValueError, match="finite number"):
        _validate_tool_schema_subset({"type": "integer", "exclusiveMinimum": bound})


@pytest.mark.parametrize("schema", [
    {"type": "integer", "minimum": 10, "exclusiveMaximum": 5},
    {"type": "integer", "exclusiveMinimum": 5, "maximum": 5},
])
def test_contradictory_mixed_bounds_rejected(schema):
    with pytest.raises(ValueError, match="lower bound exceeds upper bound"):
        _validate_tool_schema_subset(schema)


def test_opencode_schema_reaches_undeclared_protocol_without_model_work():
    class NoModelWork:
        def generate(self, *args, **kwargs):
            raise AssertionError("undeclared tools must be rejected before generation")

    client = TestClient(create_app(
        NoModelWork(), model="instella-test", chat_template="{{ messages[-1].content }}",
        tool_protocol=None, extra_route_registrar=add_chat_completions_route,
    ))
    tool = {"type": "function", "function": {
        "name": "bash", "parameters": {"$schema": "https://json-schema.org/draft/2020-12/schema",
                                      "type": "object", "properties": {"timeout": TIMEOUT}},
    }}
    response = client.post("/v1/chat/completions", json={
        "model": "instella-test", "messages": [{"role": "user", "content": "read a file"}],
        "tools": [tool], "stream": True,
    })
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "tool_protocol_undeclared"
