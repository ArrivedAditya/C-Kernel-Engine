from __future__ import annotations

import sys
from pathlib import Path

# Calculate the parent directory path and add it to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from schemas.annotations import FileCitation, URLCitation, FilePath
from schemas.common import (
    ErrorCode,
    Includable,
    ItemStatus,
    Phase,
    ResponseError,
    ResponseIncompleteDetails,
    ResponseStatus,
    Role,
    Usage,
)
from schemas.computer_actions import Click, DoubleClick, Drag, Keypress, Move, Scroll, TypeAction, Wait
from schemas.content import (
    ResponseInputFile,
    ResponseInputImage,
    ResponseInputText,
    ResponseOutputRefusal,
    ResponseOutputText,
)
from schemas.conversation import Conversation, CreateConversationRequest
from schemas.filters import ComparisonFilter, CompoundFilter
from schemas.input_items import EasyInputMessage, Message
from schemas.output_items import (
    ComputerCall,
    FileSearchCall,
    FunctionCall,
    FunctionCallOutput,
    ReasoningItem,
    ResponseOutputMessage,
    WebSearchCall,
)
from schemas.response import CreateResponseRequest, Response
from schemas.streaming import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseTextDeltaEvent,
)
from schemas.tool_definitions import (
    CodeInterpreterTool,
    FileSearchTool,
    FunctionTool,
    WebSearchTool,
)


def test_role_enum():
    assert Role.user == "user"
    assert Role.assistant == "assistant"
    assert Role.system == "system"
    assert Role.developer == "developer"


def test_item_status_enum():
    assert ItemStatus.in_progress == "in_progress"
    assert ItemStatus.completed == "completed"
    assert ItemStatus.incomplete == "incomplete"


def test_response_status():
    assert ResponseStatus.completed == "completed"


def test_phase_enum():
    assert Phase.commentary == "commentary"
    assert Phase.final_answer == "final_answer"


def test_error_code_enum():
    assert ErrorCode.server_error == "server_error"
    assert ErrorCode.rate_limit_exceeded == "rate_limit_exceeded"


def test_response_error():
    err = ResponseError(code=ErrorCode.server_error, message="Server error")
    assert err.code == "server_error"
    assert err.message == "Server error"
    d = err.model_dump()
    assert d["code"] == "server_error"


def test_response_incomplete_details():
    d = ResponseIncompleteDetails(reason="max_output_tokens")
    assert d.reason == "max_output_tokens"
    d2 = ResponseIncompleteDetails()
    assert d2.reason is None


def test_usage():
    u = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
    assert u.total_tokens == 30


def test_text_input():
    t = ResponseInputText(text="Hello", type="input_text")
    assert t.text == "Hello"
    assert t.type == "input_text"
    assert t.prompt_cache_breakpoint is None


def test_image_input():
    img = ResponseInputImage(file_id="file_abc", type="input_image")
    assert img.type == "input_image"
    assert img.detail == "auto"


def test_file_input():
    f = ResponseInputFile(file_data="base64data", filename="test.txt")
    assert f.type == "input_file"
    assert f.filename == "test.txt"


def test_output_text():
    ot = ResponseOutputText(text="Hello world", type="output_text")
    assert ot.text == "Hello world"
    assert ot.annotations == []


def test_output_refusal():
    ref = ResponseOutputRefusal(refusal="I cannot answer that", type="refusal")
    assert ref.refusal == "I cannot answer that"


def test_file_citation():
    fc = FileCitation(file_id="f1", filename="doc.pdf", index=0)
    assert fc.type == "file_citation"
    d = fc.model_dump()
    assert d["type"] == "file_citation"


def test_url_citation():
    uc = URLCitation(end_index=10, start_index=0, title="OpenAI", url="https://openai.com")
    assert uc.type == "url_citation"


def test_file_path():
    fp = FilePath(file_id="f1", index=0)
    assert fp.type == "file_path"


def test_easy_input_message():
    msg = EasyInputMessage(content="Hello", role=Role.user, type="message")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.type == "message"


def test_easy_input_message_with_content_list():
    content = [ResponseInputText(text="Hello", type="input_text")]
    msg = EasyInputMessage(content=content, role=Role.user)
    assert isinstance(msg.content, list)
    assert msg.content[0].text == "Hello"


def test_message():
    content = [ResponseInputText(text="Hello", type="input_text")]
    msg = Message(content=content, role=Role.user)
    assert msg.role == "user"
    assert msg.status is None


def test_response_output_message():
    content = [ResponseOutputText(text="Hi", type="output_text")]
    om = ResponseOutputMessage(
        id="msg_1",
        content=content,
        status=ItemStatus.completed,
    )
    assert om.role == "assistant"
    assert om.type == "message"


def test_file_search_call():
    fsc = FileSearchCall(
        id="fs_1",
        queries=["query1"],
        status="completed",
    )
    assert fsc.type == "file_search_call"
    assert fsc.status == "completed"


def test_computer_call():
    cc = ComputerCall(
        id="cc_1",
        call_id="call_1",
        status=ItemStatus.in_progress,
    )
    assert cc.type == "computer_call"


def test_web_search_call():
    from schemas.output_items import WebSearchActionSearch

    action = WebSearchActionSearch(queries=["test query"])
    wsc = WebSearchCall(
        id="ws_1",
        action=action,
        status="completed",
    )
    assert wsc.type == "web_search_call"
    assert wsc.action.type == "search"


def test_function_call():
    fc = FunctionCall(
        arguments='{"x": 1}',
        call_id="call_1",
        name="my_func",
    )
    assert fc.type == "function_call"
    assert fc.arguments == '{"x": 1}'


def test_function_call_output():
    fco = FunctionCallOutput(
        call_id="call_1",
        output='{"result": 42}',
    )
    assert fco.type == "function_call_output"


def test_reasoning_item():
    ri = ReasoningItem()
    assert ri.type == "reasoning"


def test_computer_actions():
    click = Click(button="left", x=100, y=200)
    assert click.type == "click"

    dc = DoubleClick(x=100, y=200)
    assert dc.type == "double_click"

    drag = Drag(path=[{"x": 0, "y": 0}, {"x": 100, "y": 100}])
    assert drag.type == "drag"

    kp = Keypress(keys=["ctrl", "c"])
    assert kp.type == "keypress"

    move = Move(x=500, y=500)
    assert move.type == "move"

    scroll = Scroll(scroll_x=0, scroll_y=100, x=0, y=0)
    assert scroll.type == "scroll"

    ta = TypeAction(text="hello")
    assert ta.type == "type"

    w = Wait()
    assert w.type == "wait"


def test_filters():
    cf = ComparisonFilter(key="age", type="gt", value=18)
    assert cf.type == "gt"

    compound = CompoundFilter(
        filters=[cf, cf],
        type="and",
    )
    assert compound.type == "and"


def test_function_tool():
    ft = FunctionTool(
        name="get_weather",
        parameters={"type": "object", "properties": {}},
        strict=True,
    )
    assert ft.type == "function"
    assert ft.name == "get_weather"
    assert ft.strict is True


def test_file_search_tool():
    fst = FileSearchTool(vector_store_ids=["vs_1"])
    assert fst.type == "file_search"


def test_web_search_tool():
    wst = WebSearchTool(type="web_search")
    assert wst.type == "web_search"


def test_code_interpreter_tool():
    cit = CodeInterpreterTool(container="auto")
    assert cit.type == "code_interpreter"


def test_includable_enum():
    assert Includable.file_search_call_results == "file_search_call.results"


def test_create_response_request():
    req = CreateResponseRequest(model="gpt-4o")
    assert req.model == "gpt-4o"
    assert req.input is None
    assert req.tools is None

    req2 = CreateResponseRequest(
        model="gpt-4o",
        input="Hello",
        temperature=0.7,
        max_output_tokens=100,
    )
    assert req2.input == "Hello"
    assert req2.temperature == 0.7


def test_response_model():
    resp = Response(
        id="resp_abc",
        created_at=1234567890,
        status=ResponseStatus.completed,
        model="gpt-4o",
    )
    assert resp.object == "response"
    # JSON roundtrip
    d = resp.model_dump(mode="json")
    assert d["id"] == "resp_abc"
    assert d["status"] == "completed"
    assert d["object"] == "response"

    r2 = Response.model_validate(d)
    assert r2.id == resp.id
    assert r2.status == resp.status


def test_conversation_model():
    conv = Conversation(id="conv_abc", created_at=1234567890)
    assert conv.object == "conversation"
    assert conv.items == []


def test_create_conversation_request():
    req = CreateConversationRequest()
    assert req.items is None

    req2 = CreateConversationRequest(items=["item1"])
    assert req2.items == ["item1"]


def test_streaming_events():
    event = ResponseCreatedEvent(response={"id": "resp_1"})
    assert event.type == "response.created"

    completed = ResponseCompletedEvent(response={"id": "resp_1"})
    assert completed.type == "response.completed"

    delta = ResponseTextDeltaEvent(
        response_id="resp_1",
        output_index=0,
        content_index=0,
        delta="Hello",
    )
    assert delta.type == "response.output_text.delta"
    assert delta.delta == "Hello"
