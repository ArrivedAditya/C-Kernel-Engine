"""Request and response models for the experimental schema subset."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .common import (
    Includable,
    ResponseError,
    ResponseIncompleteDetails,
    ResponseStatus,
    Usage,
)
from .input_items import EasyInputMessage, Message
from .output_items import (
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
from .tool_definitions import ToolDefinition


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
    input_items: list[Any] = Field(default_factory=list)
    output: list[Any] = Field(default_factory=list)
    model: str
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    include: list[Includable] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
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
