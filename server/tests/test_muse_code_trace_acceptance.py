"""Muse Code transport evidence must preserve tool and reasoning history."""

import copy
import json

import pytest

from server.muse_code_trace_acceptance import evaluate, parse_trace


MODEL = "muse-test"
ANSWER = "The directory is /tmp/muse-work"
IDENTITY = {
    "schema": "cke.loaded_serving_identity.v1",
    "model": MODEL,
    "serving_identity": "a" * 64,
    "session_library_sha256": "b" * 64,
    "server_instance_id": "instance-1",
}
REASON = {"type": "reasoning", "id": "reason-1",
          "content": [{"type": "reasoning_text", "text": "Run pwd."}]}
CALL = {"type": "function_call", "id": "fc-1", "call_id": "call-1",
        "namespace": "muse", "name": "bash", "arguments": '{"command":"pwd"}'}
RESULT = {"type": "function_call_output", "call_id": "call-1",
          "output": "/tmp/muse-work\n"}


def _trace():
    groups = [
        {
            "request": {"model": MODEL, "input": [{"type": "message", "role": "user",
                                                 "content": "Run pwd once."}]},
            "events": [
                {"type": "response.output_item.done", "item": REASON},
                {"type": "response.output_item.done", "item": CALL},
                {"type": "response.completed", "response": {
                    "status": "completed", "output": [REASON, CALL],
                    "output_text": "", "usage": {"input_tokens": 10}}},
            ],
        },
        {
            "request": {"model": MODEL, "input": [
                {"type": "message", "role": "user", "content": "Run pwd once."},
                REASON, {**CALL, "namespace": None, "name": "muse.bash"}, RESULT,
            ]},
            "events": [
                {"type": "response.output_item.done", "item": {
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": ANSWER}]}},
                {"type": "response.completed", "response": {
                "status": "completed", "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": ANSWER}]}],
                "output_text": ANSWER, "usage": {"input_tokens": 20}}}],
        },
    ]
    # Real transport JSON is deserialized per line; it never shares item objects.
    return json.loads(json.dumps(groups))


def _raw(groups):
    lines = []
    for group in groups:
        lines.append("REQUEST: " + json.dumps(group["request"]))
        for event in group["events"]:
            lines.append("SSE RESPONSE: event: " + event["type"]
                         + "\\ndata: " + json.dumps(event))
    return "\n".join(lines).encode()


def _evaluate(groups, **kwargs):
    return evaluate(_raw(groups), ANSWER, model=MODEL, tool_name="muse.bash",
                    expected_arguments={"command": "pwd"}, expected_answer=ANSWER,
                    expected_tool_calls=1, client_exit_code=0,
                    identity_before=IDENTITY, identity_after=IDENTITY,
                    expected_serving="a" * 64, expected_session="b" * 64,
                    **kwargs)


def test_correlated_tool_result_replay_and_identity_pass():
    report = _evaluate(_trace())
    assert [report[key]["status"] for key in ("protocol", "task", "certification")] == [
        "pass", "pass", "pass"]
    assert report["max_observed_prompt_tokens"] == 20


@pytest.mark.parametrize("mutation,expected", [
    (lambda g: g[1]["request"]["input"].pop(), "request_1_tool_result_missing"),
    (lambda g: g[1]["request"]["input"][3].update(call_id="wrong"),
     "request_1_tool_result_missing"),
    (lambda g: g[1]["request"]["input"][1]["content"][0].update(text="changed"),
     "request_1_reasoning_replay_mismatch"),
    (lambda g: g[1]["request"]["input"][2].update(arguments='{"command":"ls"}'),
     "request_1_call_replay_mismatch"),
    (lambda g: g[0]["events"][2]["response"]["output"][1].update(
        arguments='{"command":"pwd","n":1e999}'), "request_0_malformed_call"),
    (lambda g: g[0]["events"].append({"type": "response.output_item.done", "item": CALL}),
     "request_0_terminal_not_completed_once"),
])
def test_protocol_negative_controls(mutation, expected):
    groups = copy.deepcopy(_trace())
    mutation(groups)
    report = _evaluate(groups)
    assert report["protocol"]["status"] == "fail"
    assert expected in report["protocol"]["errors"]


def test_extra_tool_call_is_task_failure_even_when_answer_is_correct():
    groups = copy.deepcopy(_trace())
    second = copy.deepcopy(groups[0])
    second["request"] = copy.deepcopy(groups[1]["request"])
    second["events"][0]["item"]["id"] = "reason-2"
    second["events"][2]["response"]["output"][0]["id"] = "reason-2"
    second["events"][1]["item"]["call_id"] = "call-2"
    second["events"][2]["response"]["output"][1]["call_id"] = "call-2"
    groups[1] = second
    final = copy.deepcopy(_trace()[1])
    final["request"]["input"].extend([
        {**REASON, "id": "reason-2"},
        {**CALL, "id": "fc-2", "call_id": "call-2", "namespace": None,
         "name": "muse.bash"},
        {**RESULT, "call_id": "call-2"},
    ])
    groups.append(final)
    report = _evaluate(groups)
    assert report["protocol"]["status"] == "pass"
    assert report["task"] == {"status": "fail", "reason": "unexpected_tool_call_count"}


def test_earlier_tool_result_cannot_change_in_later_request():
    groups = copy.deepcopy(_trace())
    second = copy.deepcopy(groups[0])
    second["request"] = copy.deepcopy(groups[1]["request"])
    second["events"][0]["item"]["id"] = "reason-2"
    second["events"][2]["response"]["output"][0]["id"] = "reason-2"
    second["events"][1]["item"]["call_id"] = "call-2"
    second["events"][2]["response"]["output"][1]["call_id"] = "call-2"
    groups[1] = second
    final = copy.deepcopy(_trace()[1])
    final["request"]["input"].extend([
        {**REASON, "id": "reason-2"},
        {**CALL, "id": "fc-2", "call_id": "call-2", "namespace": None,
         "name": "muse.bash"},
        {**RESULT, "call_id": "call-2"},
    ])
    final["request"]["input"][3]["output"] = "tampered earlier output"
    groups.append(final)
    report = _evaluate(groups)
    assert report["protocol"]["status"] == "fail"
    assert "request_2_tool_result_changed" in report["protocol"]["errors"]


def test_identity_mismatch_fails_and_missing_after_is_incomplete():
    groups = _trace()
    changed = dict(IDENTITY, session_library_sha256="c" * 64)
    base = dict(model=MODEL, tool_name="muse.bash", expected_arguments={"command": "pwd"},
                expected_answer=ANSWER, expected_tool_calls=1, client_exit_code=0,
                identity_before=IDENTITY, expected_serving="a" * 64,
                expected_session="b" * 64)
    mismatch = evaluate(_raw(groups), ANSWER, identity_after=changed, **base)
    assert mismatch["certification"]["status"] == "fail"
    missing = evaluate(_raw(groups), ANSWER, **base)
    assert missing["certification"] == {
        "status": "incomplete", "reason": "before_and_after_identity_required"}


def test_malformed_trace_rejected():
    with pytest.raises(ValueError, match="event before request"):
        parse_trace(b'SSE RESPONSE: event: response.completed\\ndata: {}')


@pytest.mark.parametrize("mutation,expected", [
    (lambda g: g[1]["request"]["input"].reverse(), "request_1_history_order_mismatch"),
    (lambda g: g[1]["request"]["input"].pop(0), "request_1_history_order_mismatch"),
    (lambda g: g[1]["request"]["input"].__setitem__(slice(1, 3),
        list(reversed(g[1]["request"]["input"][1:3]))),
     "request_1_history_order_mismatch"),
    (lambda g: g[0]["request"]["input"].append(RESULT),
     "request_0_unknown_tool_result"),
    (lambda g: g[1]["events"][1]["response"]["output"][0]["content"][0].update(
        text="WRONG ANSWER"), "request_1_message_text_mismatch"),
    (lambda g: g[0]["events"][0]["item"]["content"][0].update(
        text="WRONG REASONING"), "request_0_stream_output_mismatch"),
    (lambda g: g[1]["events"].pop(0), "request_1_stream_output_mismatch"),
])
def test_reviewed_false_passes_now_fail(mutation, expected):
    groups = _trace()
    mutation(groups)
    report = _evaluate(groups)
    assert report["protocol"]["status"] == "fail"
    assert expected in report["protocol"]["errors"]
    assert report["task"]["status"] == "fail"
