from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from schemas.annotations import Annotation
from schemas.common import FileDetail, ImageDetail


class PromptCacheBreakpoint(BaseModel):
    mode: Literal["explicit"] = "explicit"


class ResponseInputText(BaseModel):
    text: str
    type: Literal["input_text"] = "input_text"
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = None


class ResponseInputImage(BaseModel):
    detail: ImageDetail = ImageDetail.auto
    type: Literal["input_image"] = "input_image"
    file_id: str | None = None
    image_url: str | None = None
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = None


class ResponseInputFile(BaseModel):
    type: Literal["input_file"] = "input_file"
    detail: FileDetail = FileDetail.auto
    file_data: str | None = None
    file_id: str | None = None
    file_url: str | None = None
    filename: str | None = None
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = None


ResponseInputContent = Annotated[
    ResponseInputText | ResponseInputImage | ResponseInputFile,
    Field(discriminator="type"),
]


class TopLogprob(BaseModel):
    token: str
    bytes: list[int] | None = None
    logprob: float


class Logprob(BaseModel):
    token: str
    bytes: list[int] | None = None
    logprob: float
    top_logprobs: list[TopLogprob] = []


class ResponseOutputText(BaseModel):
    annotations: list[Annotation] = []
    logprobs: list[Logprob] | None = None
    text: str
    type: Literal["output_text"] = "output_text"


class ResponseOutputRefusal(BaseModel):
    refusal: str
    type: Literal["refusal"] = "refusal"


ResponseOutputContent = Annotated[
    ResponseOutputText | ResponseOutputRefusal,
    Field(discriminator="type"),
]


ResponseFunctionOutput = Annotated[
    ResponseInputText | ResponseInputImage | ResponseInputFile,
    Field(discriminator="type"),
]
