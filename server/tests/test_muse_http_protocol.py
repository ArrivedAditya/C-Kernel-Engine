"""Muse channels through both public HTTP routes, with no model weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from version.v8.scripts.ck_serve_v8 import create_app


TEMPLATE = (Path(__file__).parent / "fixtures/muse_glimmer_publisher_chat_template.jinja").read_text()
TOKENS = {"bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>"}
TOOL = {"type": "function", "name": "read_file", "description": "Read a file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
}}
CALL = (' to=self<|message|>Read it first.<|eom|>'
        '<|start|>assistant to=read_file<|message|><atem:function_calls>\n'
        '<atem:invoke name="read_file">\n'
        '<atem:parameter name="path">/tmp/a</atem:parameter>\n'
        '</atem:invoke>\n</atem:function_calls><|eot|>')
FINAL = ' to=user<|message|>The file says cobalt.<|eot|>'
NAMESPACE_TOOL = {"type": "namespace", "name": "muse", "description": "Local tools", "tools": [
    {**TOOL, "name": "read_file"},
]}
NAMESPACE_CALL = CALL.replace("to=read_file", "to=muse.read_file").replace(
    'name="read_file"', 'name="muse.read_file"'
)


class _Session:
    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)
        self.prompts: list[str] = []
        self.stops: list[tuple[str, ...]] = []
        self.max_tokens_seen: list[int] = []

    def generate(self, system, user, *, max_tokens, temperature, top_p, on_token,
                 flags=0, stop_on_text=(), stop_at_eos=False):
        self.prompts.append(user)
        self.stops.append(tuple(stop_on_text))
        self.max_tokens_seen.append(max_tokens)
        output = next(self.outputs)
        for index, part in enumerate((output[:7], output[7:29], output[29:])):
            on_token(index, part)
        return {"prompt_tokens": 10, "generated_tokens": 3, "stop_reason": 1}

    def cancel(self):
        pass


def _client(outputs: list[str], *, request_output_cap: int | None = None) -> tuple[TestClient, _Session]:
    session = _Session(outputs)
    return TestClient(create_app(
        session, model="muse-test", chat_template=TEMPLATE,
        tool_protocol="muse_atem_v1", renderer_tokens=TOKENS,
        stop_on_text=["<|eot|>", "<|end_of_text|>"],
        context_length=32768, request_output_cap=request_output_cap,
    )), session


def _events(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ")]


def _completed_payload(response, stream: bool) -> dict:
    if stream:
        return next(event["response"] for event in _events(response)
                    if event["type"] == "response.completed")
    return response.json()


@pytest.mark.parametrize("stream", [False, True])
def test_muse_tool_round_trip_and_channel_isolation(stream: bool) -> None:
    client, session = _client([CALL, FINAL])
    first = client.post("/v1/responses", json={
        "model": "muse-test", "input": "Read /tmp/a", "tools": [TOOL], "stream": stream,
    })
    assert first.status_code == 200, first.text
    if stream:
        events = _events(first)
        assert any(event["type"] == "response.completed" for event in events)
        calls = [event["item"] for event in events
                 if event["type"] == "response.output_item.done"
                 and event.get("item", {}).get("type") == "function_call"]
        assert all("<|message|>" not in json.dumps(event) and "<atem:" not in json.dumps(event)
                   for event in events)
        assert len(calls) == 1
        call = calls[0]
        response_id = next(event["response"]["id"] for event in events
                           if event["type"] == "response.completed")
    else:
        payload = first.json()
        assert payload["status"] == "completed"
        call = next(item for item in payload["output"] if item["type"] == "function_call")
        response_id = payload["id"]
        assert "Read it first" not in payload.get("output_text", "")
    assert json.loads(call["arguments"]) == {"path": "/tmp/a"}
    second = client.post("/v1/responses", json={
        "model": "muse-test", "previous_response_id": response_id,
        "input": [{"type": "function_call_output", "call_id": call["call_id"], "output": "cobalt"}],
        "tools": [TOOL],
    })
    assert second.status_code == 200, second.text
    assert second.json()["output_text"] == "The file says cobalt."
    assert "cobalt" in session.prompts[1]
    assert all("<|eom|>" not in stops for stops in session.stops)


@pytest.mark.parametrize("stream", [False, True])
def test_muse_code_namespace_tool_round_trip(stream: bool) -> None:
    client, session = _client([NAMESPACE_CALL, FINAL])
    first = client.post("/v1/responses", json={
        "model": "muse-test", "input": "Read /tmp/a", "tools": [NAMESPACE_TOOL],
        "stream": stream,
    })
    assert first.status_code == 200, first.text
    if stream:
        events = _events(first)
        assert any(e["type"] == "response.completed" for e in events)
        calls = [e["item"] for e in events if e["type"] == "response.output_item.done"
                 and e.get("item", {}).get("type") == "function_call"]
        response_id = next(e["response"]["id"] for e in events
                           if e["type"] == "response.completed")
    else:
        payload = first.json()
        calls = [item for item in payload["output"] if item["type"] == "function_call"]
        response_id = payload["id"]
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert calls[0]["namespace"] == "muse"
    assert json.loads(calls[0]["arguments"]) == {"path": "/tmp/a"}
    assert '"name": "muse.read_file"' in session.prompts[0]
    assert '"description": "Local tools"' in session.prompts[0]
    second = client.post("/v1/responses", json={
        "model": "muse-test", "previous_response_id": response_id,
        "input": [{"type": "function_call_output", "call_id": calls[0]["call_id"],
                   "output": "cobalt"}], "tools": [NAMESPACE_TOOL],
    })
    assert second.status_code == 200, second.text
    assert second.json()["output_text"] == "The file says cobalt."
    assert '<tool_output name="muse.read_file">' in session.prompts[1]


@pytest.mark.parametrize("stream", [False, True])
def test_muse_code_explicit_reasoning_and_tool_history_is_preserved(
    stream: bool,
) -> None:
    client, session = _client([FINAL])
    response = client.post("/v1/responses", json={
        "model": "muse-test", "stream": stream, "tools": [NAMESPACE_TOOL],
        "input": [
            {"type": "message", "role": "user", "content": "Read /tmp/a"},
            {"type": "reasoning", "content": [{
                "type": "reasoning_text", "text": "I should read the file."
            }]},
            {"type": "function_call", "namespace": "muse", "name": "read_file",
             "call_id": "call_1", "arguments": '{"path":"/tmp/a"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "cobalt"},
        ],
    })
    assert response.status_code == 200, response.text
    if stream:
        assert any(event["type"] == "response.completed" for event in _events(response))
    else:
        assert response.json()["output_text"] == "The file says cobalt."
    assert len(session.prompts) == 1
    prompt = session.prompts[0]
    assert (
        "<|start|>assistant to=self<|message|>I should read the file.<|eom|>"
        "<|start|>assistant to=muse.read_file<|message|>"
    ) in prompt
    assert '<tool_output name="muse.read_file">\ncobalt\n</tool_output>' in prompt


def test_muse_code_unrenderable_reasoning_history_fails_before_generation() -> None:
    client, session = _client([FINAL])
    rejected = client.post("/v1/responses", json={
        "model": "muse-test", "input": [{
            "type": "reasoning", "encrypted_content": "unavailable"
        }],
    })
    assert rejected.status_code == 400, rejected.text
    assert session.prompts == []
    recovered = client.post("/v1/responses", json={
        "model": "muse-test", "input": "Hello",
    })
    assert recovered.status_code == 200, recovered.text
    assert len(session.prompts) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_muse_response_reasoning_replay_matches_stored_continuation(stream: bool) -> None:
    client, session = _client([NAMESPACE_CALL, FINAL, FINAL])
    first = client.post("/v1/responses", json={
        "model": "muse-test", "input": "Read /tmp/a", "tools": [NAMESPACE_TOOL],
        "reasoning": {"effort": "high"}, "stream": stream,
    })
    assert first.status_code == 200, first.text
    payload = _completed_payload(first, stream)
    assert [item["type"] for item in payload["output"]] == ["reasoning", "function_call"]
    call = payload["output"][1]
    result = {"type": "function_call_output", "call_id": call["call_id"],
              "output": "cobalt"}
    common = {"model": "muse-test", "tools": [NAMESPACE_TOOL],
              "reasoning": {"effort": "high"}, "stream": stream}
    explicit = client.post("/v1/responses", json={
        **common, "input": [
            {"type": "message", "role": "user", "content": "Read /tmp/a"},
            *payload["output"], result,
        ],
    })
    stored = client.post("/v1/responses", json={
        **common, "previous_response_id": payload["id"], "input": [result],
    })
    assert explicit.status_code == stored.status_code == 200
    assert "Read it first." in session.prompts[1]
    assert session.prompts[1] == session.prompts[2]


@pytest.mark.parametrize("stream", [False, True])
def test_muse_empty_response_reasoning_can_be_replayed(stream: bool) -> None:
    client, session = _client([FINAL, FINAL, FINAL])
    first = client.post("/v1/responses", json={
        "model": "muse-test", "input": "Hello", "reasoning": {"effort": "high"},
        "stream": stream,
    })
    assert first.status_code == 200, first.text
    payload = _completed_payload(first, stream)
    assert payload["output"][0]["type"] == "reasoning"
    assert payload["output"][0]["content"] == []
    common = {"model": "muse-test", "reasoning": {"effort": "high"},
              "stream": stream}
    explicit = client.post("/v1/responses", json={
        **common, "input": [
            {"type": "message", "role": "user", "content": "Hello"},
            *payload["output"],
            {"type": "message", "role": "user", "content": "Continue"},
        ],
    })
    stored = client.post("/v1/responses", json={
        **common, "previous_response_id": payload["id"], "input": "Continue",
    })
    assert explicit.status_code == stored.status_code == 200
    assert session.prompts[1] == session.prompts[2]


@pytest.mark.parametrize("variant", [
    {"type": "namespace", "name": "bad.name", "tools": [TOOL]},
    {"type": "namespace", "name": "muse", "tools": []},
    {"type": "namespace", "name": "muse", "tools": [{**TOOL, "name": "bad/name"}]},
    {"type": "namespace", "name": "muse", "tools": [{**TOOL, "defer_loading": True}]},
    {"type": "namespace", "name": "muse", "tools": [TOOL, TOOL]},
])
def test_invalid_namespace_fails_before_generation(variant: dict) -> None:
    client, session = _client([NAMESPACE_CALL])
    response = client.post("/v1/responses", json={
        "model": "muse-test", "input": "Read", "tools": [variant],
    })
    assert response.status_code in (400, 501), response.text
    assert session.prompts == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("name", ["123tools", "-tools"])
def test_namespace_name_must_be_parseable_before_generation(
    stream: bool, name: str,
) -> None:
    client, session = _client([NAMESPACE_CALL])
    bad = client.post("/v1/responses", json={
        "model": "muse-test", "input": "Read", "stream": stream,
        "tools": [{**NAMESPACE_TOOL, "name": name}],
    })
    assert bad.status_code == 400, bad.text
    assert session.prompts == []
    good = client.post("/v1/responses", json={
        "model": "muse-test", "input": "Read", "stream": stream,
        "tools": [NAMESPACE_TOOL],
    })
    assert good.status_code == 200, good.text
    if stream:
        assert any(e["type"] == "response.completed" for e in _events(good))
    else:
        assert good.json()["status"] == "completed"
    assert len(session.prompts) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("selector,generated,expected_status", [
    ({"type": "function", "namespace": "muse", "name": "read_file"},
     NAMESPACE_CALL, "completed"),
    ({"type": "function", "namespace": "other", "name": "read_file"},
     NAMESPACE_CALL, "failed"),
    ({"type": "function", "namespace": "muse", "name": "read_file"},
     FINAL, "failed"),
])
def test_forced_namespaced_tool_choice_is_enforced(
    stream: bool, selector: dict, generated: str, expected_status: str,
) -> None:
    client, session = _client([generated])
    other = {**NAMESPACE_TOOL, "name": "other"}
    response = client.post("/v1/responses", json={
        "model": "muse-test", "input": "Read", "stream": stream,
        "tools": [NAMESPACE_TOOL, other], "tool_choice": selector,
    })
    assert response.status_code == 200, response.text
    if stream:
        events = _events(response)
        assert events[-1]["type"] == f"response.{expected_status}"
    else:
        assert response.json()["status"] == expected_status
    assert len(session.prompts) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_unsupported_tool_choice_is_rejected_before_generation(stream: bool) -> None:
    client, session = _client([NAMESPACE_CALL])
    response = client.post("/v1/responses", json={
        "model": "muse-test", "input": "Read", "stream": stream,
        "tools": [NAMESPACE_TOOL], "tool_choice": {"type": "mcp", "name": "read_file"},
    })
    assert response.status_code == 400, response.text
    assert session.prompts == []


def test_namespace_collision_with_flat_tool_fails_before_generation() -> None:
    client, session = _client([NAMESPACE_CALL])
    response = client.post("/v1/responses", json={
        "model": "muse-test", "input": "Read", "tools": [NAMESPACE_TOOL,
            {**TOOL, "name": "muse.read_file"}],
    })
    assert response.status_code == 400, response.text
    assert session.prompts == []


@pytest.mark.parametrize("stream", [False, True])
def test_muse_unknown_namespace_call_does_not_publish_tool(stream: bool) -> None:
    client, _ = _client([NAMESPACE_CALL.replace("muse.read_file", "other.read_file"), FINAL])
    request = {"model": "muse-test", "input": "Read", "tools": [NAMESPACE_TOOL],
               "stream": stream}
    first = client.post("/v1/responses", json=request)
    assert first.status_code == 200, first.text
    if stream:
        events = _events(first)
        assert any(e["type"] == "response.failed" for e in events)
        assert not any(e.get("item", {}).get("type") == "function_call" for e in events)
    else:
        assert first.json()["status"] == "failed"
        assert not any(i["type"] == "function_call" for i in first.json()["output"])
    second = client.post("/v1/responses", json=request)
    assert second.status_code == 200, second.text
    if stream:
        assert any(e["type"] == "response.completed" for e in _events(second))
    else:
        assert second.json()["status"] == "completed"


def test_explicit_namespaced_call_history_renders_qualified_tool() -> None:
    client, session = _client([FINAL])
    response = client.post("/v1/responses", json={
        "model": "muse-test", "tools": [NAMESPACE_TOOL],
        "input": [
            {"type": "function_call", "call_id": "call_one", "name": "read_file",
             "namespace": "muse", "arguments": '{"path":"/tmp/a"}'},
            {"type": "function_call_output", "call_id": "call_one", "output": "cobalt"},
        ],
    })
    assert response.status_code == 200, response.text
    assert '<tool_output name="muse.read_file">' in session.prompts[0]


def test_muse_code_large_requested_budget_uses_explicit_server_cap() -> None:
    client, session = _client([FINAL], request_output_cap=256)
    response = client.post("/v1/responses", json={
        "model": "muse-test", "input": "Say hello", "tools": [NAMESPACE_TOOL],
        "max_output_tokens": 128000,
    })
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"
    assert session.max_tokens_seen == [256]


@pytest.mark.parametrize("stream", [False, True])
def test_muse_bad_tool_then_follow_up_recovers(stream: bool) -> None:
    bad = CALL.replace('/tmp/a', '1e999').replace('name="path"', 'name="n"')
    client, _ = _client([bad, CALL])
    request = {"model": "muse-test", "input": "Read it", "tools": [TOOL], "stream": stream}
    first = client.post("/v1/responses", json=request)
    assert first.status_code == 200, first.text
    if stream:
        events = _events(first)
        assert any(event["type"] == "response.failed" for event in events)
        assert not any(event.get("item", {}).get("type") == "function_call" for event in events)
    else:
        assert first.json()["status"] == "failed"
        assert not any(item["type"] == "function_call" for item in first.json()["output"])
    second = client.post("/v1/responses", json=request)
    assert second.status_code == 200, second.text
    if stream:
        assert any(event["type"] == "response.completed" for event in _events(second))
    else:
        assert second.json()["status"] == "completed"


@pytest.mark.parametrize("stream", [False, True])
def test_muse_final_answer_does_not_leak_reasoning_or_fake_invoke(stream: bool) -> None:
    raw = (' to=self<|message|>private reasoning<|eom|>'
           '<|start|>assistant to=user<|message|>Here is a sample '
           '<atem:invoke name="read_file"></atem:invoke><|eot|>')
    client, _ = _client([raw])
    response = client.post('/v1/responses', json={
        'model': 'muse-test', 'input': 'Explain this', 'stream': stream,
    })
    assert response.status_code == 200, response.text
    if stream:
        events = _events(response)
        assert not any(event.get('item', {}).get('type') == 'function_call' for event in events)
        assert not any('private reasoning' in json.dumps(event) for event in events)
        assert any(event['type'] == 'response.output_text.done'
                   and 'Here is a sample' in event.get('text', '') for event in events)
    else:
        payload = response.json()
        assert payload['status'] == 'completed'
        assert 'Here is a sample' in payload['output_text']
        assert 'private reasoning' not in json.dumps(payload)


@pytest.mark.parametrize("stream", [False, True])
def test_muse_undeclared_tool_fails_without_publishing_call(stream: bool) -> None:
    client, _ = _client([CALL, FINAL])
    request = {"model": "muse-test", "input": "Say hello", "stream": stream}
    first = client.post("/v1/responses", json=request)
    assert first.status_code == 200, first.text
    if stream:
        events = _events(first)
        assert any(event["type"] == "response.failed" for event in events)
        assert not any(event.get("item", {}).get("type") == "function_call" for event in events)
    else:
        assert first.json()["status"] == "failed"
        assert not any(item["type"] == "function_call" for item in first.json()["output"])
    second = client.post("/v1/responses", json=request)
    assert second.status_code == 200, second.text
    if stream:
        assert any(event["type"] == "response.completed" for event in _events(second))
    else:
        assert second.json()["output_text"] == "The file says cobalt."
