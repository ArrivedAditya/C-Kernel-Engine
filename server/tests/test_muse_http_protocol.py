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


class _Session:
    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)
        self.prompts: list[str] = []
        self.stops: list[tuple[str, ...]] = []

    def generate(self, system, user, *, max_tokens, temperature, top_p, on_token,
                 flags=0, stop_on_text=(), stop_at_eos=False):
        self.prompts.append(user)
        self.stops.append(tuple(stop_on_text))
        output = next(self.outputs)
        for index, part in enumerate((output[:7], output[7:29], output[29:])):
            on_token(index, part)
        return {"prompt_tokens": 10, "generated_tokens": 3, "stop_reason": 1}

    def cancel(self):
        pass


def _client(outputs: list[str]) -> tuple[TestClient, _Session]:
    session = _Session(outputs)
    return TestClient(create_app(
        session, model="muse-test", chat_template=TEMPLATE,
        tool_protocol="muse_atem_v1", renderer_tokens=TOKENS,
        stop_on_text=["<|eot|>", "<|end_of_text|>"],
    )), session


def _events(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ")]


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
