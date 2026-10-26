"""Negative controls for the optional LangGraph tool-result assessor."""
from server.langgraph_read_acceptance import assess


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
