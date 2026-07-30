"""Streaming event shapes; generation remains mocked in this scaffold."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from .common import ResponseIncompleteDetails
from .output_items import ResponseOutputItem


class ResponseCreatedEvent(BaseModel):
    type: Literal["response.created"] = "response.created"
    response: dict[str, Any]


class ResponseInProgressEvent(BaseModel):
    type: Literal["response.in_progress"] = "response.in_progress"
    response: dict[str, Any]


class ResponseCompletedEvent(BaseModel):
    type: Literal["response.completed"] = "response.completed"
    response: dict[str, Any]


class ResponseFailedEvent(BaseModel):
    type: Literal["response.failed"] = "response.failed"
    response: dict[str, Any]


class ResponseIncompleteEvent(BaseModel):
    type: Literal["response.incomplete"] = "response.incomplete"
    response: dict[str, Any]


class ResponseOutputItemAddedEvent(BaseModel):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    response_id: str
    output_index: int
    item: ResponseOutputItem


class ResponseOutputItemDoneEvent(BaseModel):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    response_id: str
    output_index: int
    item: ResponseOutputItem


class ResponseContentPartAddedEvent(BaseModel):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    response_id: str
    output_index: int
    content_index: int
    part: dict[str, Any]


class ResponseContentPartDoneEvent(BaseModel):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    response_id: str
    output_index: int
    content_index: int
    part: dict[str, Any]


class ResponseTextDeltaEvent(BaseModel):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    response_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseTextDoneEvent(BaseModel):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    response_id: str
    output_index: int
    content_index: int
    text: str


class ResponseRefusalDeltaEvent(BaseModel):
    type: Literal["response.refusal.delta"] = "response.refusal.delta"
    response_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseRefusalDoneEvent(BaseModel):
    type: Literal["response.refusal.done"] = "response.refusal.done"
    response_id: str
    output_index: int
    content_index: int
    refusal: str


class ResponseIncompleteDoneEvent(BaseModel):
    type: Literal["response.incomplete.done"] = "response.incomplete.done"
    response_id: str
    output_index: int
    content_index: int
    incomplete_details: ResponseIncompleteDetails | None = None


class ResponseFunctionCallArgumentsDeltaEvent(BaseModel):
    type: Literal[
        "response.function_call_arguments.delta"
    ] = "response.function_call_arguments.delta"
    response_id: str
    output_index: int
    item_id: str
    delta: str


class ResponseFunctionCallArgumentsDoneEvent(BaseModel):
    type: Literal[
        "response.function_call_arguments.done"
    ] = "response.function_call_arguments.done"
    response_id: str
    output_index: int
    item_id: str
    arguments: str


ResponseStreamEvent = (
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseFailedEvent
    | ResponseIncompleteEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseRefusalDeltaEvent
    | ResponseRefusalDoneEvent
    | ResponseIncompleteDoneEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
)
