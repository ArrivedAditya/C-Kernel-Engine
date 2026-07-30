from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from schemas.common import (
    Includable,
    ResponseError,
    ResponseIncompleteDetails,
    ResponseStatus,
    Usage,
)
from schemas.input_items import EasyInputMessage, Message
from schemas.output_items import (
    ComputerCall,
    ComputerCallOutput,
    FileSearchCall,
    FunctionCall,
    FunctionCallOutput,
    ReasoningItem,
    ResponseOutputMessage,
    ToolSearchCall,
    ToolSearchOutput,
    WebSearchCall,
)
from schemas.tool_definitions import ToolDefinition


class ContextManagementEntry(BaseModel):
    type: str = "compaction"
    compact_threshold: int | None = None


class ResponseConversationParam(BaseModel):
    id: str


CreateResponseInput = str | list[
    EasyInputMessage
    | Message
    | ResponseOutputMessage
    | FileSearchCall
    | ComputerCall
    | ComputerCallOutput
    | WebSearchCall
    | FunctionCall
    | FunctionCallOutput
    | ToolSearchCall
    | ToolSearchOutput
    | ReasoningItem
]


class CreateResponseRequest(BaseModel):
    model: str
    input: CreateResponseInput | None = None
    instructions: str | list[Any] | None = None
    conversation: str | ResponseConversationParam | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    include: list[Includable] | None = None
    metadata: dict[str, str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    max_output_tokens: int | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    truncation: str | None = None
    text: dict[str, Any] | None = None
    user: str | None = None
    background: bool | None = None
    context_management: list[ContextManagementEntry] | None = None
    prompt_cache_options: dict[str, Any] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None


class Response(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    status: ResponseStatus
    error: ResponseError | None = None
    incomplete_details: ResponseIncompleteDetails | None = None
    instructions: str | list[Any] | None = None
    input_items: list[Any] = []
    output_items: list[Any] = []
    model: str
    tools: list[ToolDefinition] = []
    tool_choice: str | dict[str, Any] | None = None
    include: list[Includable] = []
    metadata: dict[str, str] = {}
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    max_output_tokens: int | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    reasoning: dict[str, Any] | None = None
    truncation: str | None = None
    text: dict[str, Any] | None = None
    usage: Usage | None = None
    user: str | None = None
    conversation_id: str | None = None
    context_management: list[ContextManagementEntry] | None = None
    prompt_cache_options: dict[str, Any] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    token_usage: dict[str, Any] | None = None


class ResponseList(BaseModel):
    object: str = "list"
    data: list[Response]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
