"""Nemotron's publisher-declared TOOLCALL JSON envelope through CKE HTTP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.live import _extract_tool_calls_from_text, create_app

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "version" / "v8" / "scripts"))
from ck_serve_v8 import create_app as create_chat_app


TEMPLATE = (Path(__file__).parent / "fixtures/nemotron_nano_9b_v2_publisher_chat_template.jinja").read_text()
TOOL = {"type": "function", "name": "read_file", "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}
CALL = '<TOOLCALL>[{"name":"read_file","arguments":{"path":"x"}}]</TOOLCALL>'
REASONING_CALL = "I should use cat 'note.txt' to read the file.\n</think>\n\n" + CALL
EXPLANATORY_CALL = (
    "The `add` function returns `a - b`. For example, `add(2, 3)` is -1.\n\n"
    "Let me execute the corrected command:\n\n" + CALL
)


@pytest.mark.parametrize("text,expected", [
    (CALL, 1),
    ("</think>\n" + CALL, 1),
    (REASONING_CALL, 1),
    (EXPLANATORY_CALL, 1),
    ("I will read the file.\n\n" + CALL, 1),
    ('<TOOLCALL>[{"name":"read_file","arguments":{"path":"x"}},'
     '{"name":"read_file","arguments":"{\\"path\\":\\"y\\"}"}]</TOOLCALL>', 2),
])
def test_publisher_envelope_preserves_calls(text: str, expected: int) -> None:
    calls, code, _ = _extract_tool_calls_from_text(
        text, {"read_file"}, tool_syntax="nemotron_toolcall_v1",
        tool_parameters={"read_file": TOOL["parameters"]},
    )
    assert code is None
    assert len(calls) == expected
    assert len({call["call_id"] for call in calls}) == expected
    assert json.loads(calls[0]["arguments"])["path"] == "x"


@pytest.mark.parametrize("text,code", [
    ('<TOOLCALL>[{"name":"read_file","arguments":{"path":"x"}}]', "malformed"),
    ('<TOOLCALL>{"name":"read_file","arguments":{"path":"x"}}</TOOLCALL>', "malformed"),
    ('<TOOLCALL>[{"name":"read_file","arguments":{"path":"x"}}]'
     '</TOOLCALL><TOOLCALL>[]</TOOLCALL>', "malformed"),
    ("</TOOLCALL><TOOLCALL>[]", "malformed"),
    ("Example only:\n```xml\n" + CALL + "\n```", "malformed"),
    ("Example only:\n\n" + CALL, "malformed"),
    ("Here is a closed fence:\n```xml\n<TOOLCALL>[]</TOOLCALL>\n```\n\n" + CALL, "malformed"),
    ("    " + CALL, "malformed"),
    ("> " + CALL, "malformed"),
    ("The example is \"" + CALL + "\"", "malformed"),
    ("<think>Example: " + CALL + "</think>", "malformed"),
    ("I should read the file.\n</think>\n</think>\n" + CALL, "malformed"),
    ('<TOOLCALL>[{"name":"other","arguments":{}}]</TOOLCALL>', "unknown"),
    ('<TOOLCALL>[{"name":"read_file","arguments":{"path":"x","n":1e999}}]'
     '</TOOLCALL>', "malformed"),
    ('<TOOLCALL>[{"name":"read_file","arguments":{"path":"x","nested":{"n":1e999}}}]'
     '</TOOLCALL>', "malformed"),
    ('<TOOLCALL>[{"name":"read_file","arguments":{"path":"x","n":'
     + "9" * 4400 + '}}]</TOOLCALL>', "malformed"),
])
def test_malformed_or_undeclared_calls_fail_closed(text: str, code: str) -> None:
    calls, actual, _ = _extract_tool_calls_from_text(
        text, {"read_file"}, tool_syntax="nemotron_toolcall_v1",
        tool_parameters={"read_file": TOOL["parameters"]},
    )
    assert calls == []
    assert actual == code


class _Session:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.prompts: list[str] = []

    def generate(self, system, user, *, max_tokens, temperature, top_p, on_token,
                 flags=0, stop_on_text=(), stop_at_eos=False):
        self.prompts.append(user)
        output = next(self.outputs)
        for index, part in enumerate((output[:7], output[7:18], output[18:])):
            if part:
                on_token(index, part)
        return {"prompt_tokens": 10, "generated_tokens": 3, "stop_reason": 1}

    def cancel(self):
        pass

    def count_tokens(self, prompt):
        return len(prompt.split())

    def close(self):
        pass


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("first_output", [CALL, REASONING_CALL, EXPLANATORY_CALL])
def test_http_call_result_and_continuation(stream: bool, first_output: str) -> None:
    session = _Session([first_output, "The file says cobalt."])
    client = TestClient(create_app(
        session, model="nemotron-test", chat_template=TEMPLATE,
        tool_protocol="nemotron_toolcall_v1",
    ))
    first = client.post("/v1/responses", json={
        "model": "nemotron-test", "input": "Read x", "tools": [TOOL],
        "stream": stream,
    })
    assert first.status_code == 200, first.text
    if stream:
        events = [json.loads(line[6:]) for line in first.text.splitlines()
                  if line.startswith("data: ")]
        calls = [event["item"] for event in events
                 if event["type"] == "response.output_item.done"
                 and event.get("item", {}).get("type") == "function_call"]
        assert not any("<TOOLCALL>" in json.dumps(event) for event in events)
        response_id = next(event["response"]["id"] for event in events
                           if event["type"] == "response.completed")
    else:
        payload = first.json()
        calls = [item for item in payload["output"] if item["type"] == "function_call"]
        response_id = payload["id"]
    assert len(calls) == 1
    assert json.loads(calls[0]["arguments"]) == {"path": "x"}
    second = client.post("/v1/responses", json={
        "model": "nemotron-test", "previous_response_id": response_id,
        "input": [{"type": "function_call_output", "call_id": calls[0]["call_id"],
                   "output": "cobalt"}], "tools": [TOOL],
    })
    assert second.status_code == 200, second.text
    assert second.json()["output_text"] == "The file says cobalt."
    assert '<TOOL_RESPONSE>[cobalt]</TOOL_RESPONSE>' in session.prompts[1]


@pytest.mark.parametrize("first_output", [REASONING_CALL, EXPLANATORY_CALL])
def test_chat_completions_publisher_reasoning_call_and_result(first_output: str) -> None:
    session = _Session([first_output, "The file says cobalt."])
    client = TestClient(create_chat_app(
        session, model="nemotron-test", chat_template=TEMPLATE,
        tool_protocol="nemotron_toolcall_v1", viz=False,
    ))
    tool = {"type": "function", "function": {
        "name": TOOL["name"], "description": TOOL["description"],
        "parameters": TOOL["parameters"],
    }}
    first = client.post("/v1/chat/completions", json={
        "model": "nemotron-test", "messages": [{"role": "user", "content": "Read x"}],
        "tools": [tool],
    })
    assert first.status_code == 200, first.text
    call = first.json()["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"path": "x"}
    second = client.post("/v1/chat/completions", json={
        "model": "nemotron-test", "messages": [
            {"role": "user", "content": "Read x"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "cobalt"},
        ], "tools": [tool],
    })
    assert second.status_code == 200, second.text
    assert second.json()["choices"][0]["message"]["content"] == "The file says cobalt."
    assert '<TOOL_RESPONSE>[cobalt]</TOOL_RESPONSE>' in session.prompts[1]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("bad_output", [
    CALL.replace("</TOOLCALL>", ""),
    "</TOOLCALL><TOOLCALL>[]",
    "Example only:\n```xml\n" + CALL + "\n```",
    "Example only:\n\n" + CALL,
    "    " + CALL,
    "The example is \"" + CALL + "\"",
])
def test_http_malformed_call_then_valid_request(stream: bool, bad_output: str) -> None:
    session = _Session([bad_output, CALL])
    client = TestClient(create_app(
        session, model="nemotron-test", chat_template=TEMPLATE,
        tool_protocol="nemotron_toolcall_v1",
    ))
    request = {"model": "nemotron-test", "input": "Read x", "tools": [TOOL],
               "stream": stream}
    bad = client.post("/v1/responses", json=request)
    assert bad.status_code == 200
    if stream:
        events = [json.loads(line[6:]) for line in bad.text.splitlines()
                  if line.startswith("data: ")]
        assert any(event["type"] == "response.failed" for event in events)
        assert not any(event.get("item", {}).get("type") == "function_call"
                       for event in events)
    else:
        assert bad.json()["status"] == "failed"
        assert not any(item["type"] == "function_call" for item in bad.json()["output"])
    good = client.post("/v1/responses", json=request)
    assert good.status_code == 200
    if stream:
        events = [json.loads(line[6:]) for line in good.text.splitlines()
                  if line.startswith("data: ")]
        assert any(event.get("item", {}).get("type") == "function_call"
                   for event in events)
    else:
        assert good.json()["status"] == "completed"
        assert any(item["type"] == "function_call" for item in good.json()["output"])
