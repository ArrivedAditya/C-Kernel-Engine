"""Negative controls for the optional LangGraph tool-result assessor."""
import argparse
import json
from types import SimpleNamespace

from server import langgraph_read_acceptance as lane

assess = lane.assess


def _trace():
    return [
        {"type": "AIMessage", "content": "", "tool_calls": [{
            "name": "exec_bash_command", "args": {"cmd": "cat note.txt"}, "id": "call-1",
        }]},
        {"type": "ToolMessage", "content": '{"stdout":"cke-123"}',
         "tool_call_id": "call-1"},
        {"type": "AIMessage", "content": "cke-123", "tool_calls": [],
         "finish_reason": "stop"},
    ]


def test_correlated_tool_execution_and_answer_pass():
    assert assess(_trace(), [{"cmd": "cat note.txt", "stdout": "cke-123"}], "cke-123") == (True, "pass")


def test_answer_without_executed_tool_fails():
    assert assess(_trace(), [], "cke-123") == (False, "tool_execution")


def test_wrong_result_call_id_fails():
    trace = _trace()
    trace[1]["tool_call_id"] = "call-2"
    assert assess(trace, [{"cmd": "cat note.txt", "stdout": "cke-123"}], "cke-123") == (False, "result_correlation")


def test_wrong_tool_result_fails_even_when_answer_is_right():
    trace = _trace()
    trace[1]["content"] = '{"stdout":"wrong"}'
    assert assess(trace, [{"cmd": "cat note.txt", "stdout": "cke-123"}], "cke-123") == (False, "result_payload")


def test_output_limit_is_not_a_complete_answer():
    trace = _trace()
    trace[-1]["finish_reason"] = "length"
    assert assess(trace, [{"cmd": "cat note.txt", "stdout": "cke-123"}], "cke-123") == (False, "terminal_state")


def test_tool_result_must_follow_its_call():
    trace = _trace()
    trace[0], trace[1] = trace[1], trace[0]
    assert assess(trace, [{"cmd": "cat note.txt", "stdout": "cke-123"}], "cke-123") == (False, "event_order")


def _args(tmp_path):
    return argparse.Namespace(run_dir=tmp_path / "run", base_url="http://localhost:19099/v1",
                              model="nemotron-test", expected_serving_identity="a" * 64,
                              expected_session_sha256="b" * 64, attempts=1,
                              output_tokens=256, case_timeout=1)


def _identity(instance="instance"):
    return {"schema": "cke.loaded_serving_identity.v1", "model": "nemotron-test",
            "serving_identity": "a" * 64, "session_library_sha256": "b" * 64,
            "server_instance_id": instance}


def test_missing_instance_rejected_before_child(tmp_path, monkeypatch):
    monkeypatch.setattr(lane.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("child must not run")))
    for number, instance in enumerate(("", "   ")):
        monkeypatch.setattr(lane, "identity", lambda _, value=instance: _identity(value))
        args = _args(tmp_path)
        args.run_dir = tmp_path / f"run-{number}"
        report = lane.run(args)
        assert report["status"] == "FAIL" and report["gate"] == "missing_server_instance"
        assert report["attempts"] == []


def test_initial_identity_failure_is_structured(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "identity", lambda _: (_ for _ in ()).throw(OSError("offline")))
    report = lane.run(_args(tmp_path))
    assert report["status"] == "FAIL" and report["gate"] == "identity_unavailable"
    assert report["error_type"] == "OSError" and report["attempts"] == []


def test_malformed_child_output_retains_failed_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "identity", lambda _: _identity())
    monkeypatch.setattr(lane.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="not-json", stderr="bad output", returncode=0))
    report = lane.run(_args(tmp_path))
    assert report["status"] == "FAIL"
    assert report["attempts"][0]["gate"] == "malformed_child_output"
    assert report["attempts"][0]["same_server_instance"]
    assert (tmp_path / "run/attempt-1.json").read_text() == "not-json"
    assert (tmp_path / "run/attempt-1.stderr").read_text() == "bad output"


def test_nested_malformed_child_output_is_structured(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "identity", lambda _: _identity())
    monkeypatch.setattr(lane.secrets, "token_hex", lambda _: "123456789abc")
    monkeypatch.setattr(lane.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=json.dumps({"messages": [{"type": "AIMessage", "tool_calls": ["invalid"]}],
                           "calls": [{"cmd": "cat note.txt", "stdout": "cke-123456789abc"}]}),
        stderr="", returncode=0))
    report = lane.run(_args(tmp_path))
    assert report["status"] == "FAIL"
    assert report["attempts"][0]["gate"] == "malformed_child_output"
