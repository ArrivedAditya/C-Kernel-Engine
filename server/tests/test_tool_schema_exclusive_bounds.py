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


@pytest.mark.parametrize("bound", [10**400, -(10**400)])
def test_large_integer_bounds_do_not_overflow_float_finiteness_check(bound):
    schema = {"type": "integer", "exclusiveMinimum": bound}
    _validate_tool_schema_subset(schema)
    assert not _tool_value_matches_schema(bound, schema)
    assert _tool_value_matches_schema(bound + 1, schema)


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


@pytest.mark.parametrize("stream", [False, True])
def test_opencode_schema_reaches_undeclared_protocol_without_model_work(monkeypatch, stream):
    def unexpected_render(*args, **kwargs):
        raise AssertionError("undeclared tools must be rejected before rendering")

    monkeypatch.setattr("server.live._render_with_chat_templates", unexpected_render)

    class NoModelWork:
        def generate(self, *args, **kwargs):
            raise AssertionError("undeclared tools must be rejected before generation")

    client = TestClient(create_app(
        NoModelWork(), model="instella-test", chat_template="{{ messages[-1].content }}",
        tool_protocol=None, extra_route_registrar=add_chat_completions_route,
    ))
    captured = json.loads(FIXTURE.read_text())
    tools = [{"type": "function", "function": {"name": name, "parameters": schema}}
             for name, schema in captured["tool_schemas"].items()]
    response = client.post("/v1/chat/completions", json={
        "model": "instella-test", "messages": [{"role": "user", "content": "read a file"}],
        "tools": tools, "stream": stream,
    })
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "tool_protocol_undeclared"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("bound_key", ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"])
@pytest.mark.parametrize("bound", [10**400, -(10**400)])
def test_large_integer_tool_bound_rejects_before_model_work_and_recovers(
    monkeypatch, stream, bound_key, bound,
):
    import server.live as live

    render_calls = []
    real_render = live._render_with_chat_templates

    def track_render(*args, **kwargs):
        render_calls.append(True)
        return real_render(*args, **kwargs)

    monkeypatch.setattr(live, "_render_with_chat_templates", track_render)

    class RecordingSession:
        generation_calls = 0

        def generate(self, *args, on_token, **kwargs):
            self.generation_calls += 1
            on_token(1, "OK")
            return {"prompt_tokens": 1, "generated_tokens": 1, "stop_reason": 1}

    session = RecordingSession()

    client = TestClient(create_app(
        session, model="instella-test", chat_template="{{ messages[-1].content }}",
        tool_protocol=None, extra_route_registrar=add_chat_completions_route,
    ))
    response = client.post("/v1/chat/completions", json={
        "model": "instella-test", "messages": [{"role": "user", "content": "read a file"}],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {
            "type": "object", "properties": {"limit": {
                "type": "integer", bound_key: bound,
            }},
        }}}], "stream": stream,
    })
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "tool_protocol_undeclared"
    assert not render_calls
    assert session.generation_calls == 0

    follow_up = client.post("/v1/chat/completions", json={
        "model": "instella-test", "messages": [{"role": "user", "content": "say OK"}],
        "stream": False,
    })
    assert follow_up.status_code == 200
    assert follow_up.json()["choices"][0]["message"]["content"] == "OK"
    assert render_calls
    assert session.generation_calls == 1
