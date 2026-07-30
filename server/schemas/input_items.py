from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from schemas.common import ItemStatus, Phase, Role
from schemas.content import ResponseInputContent


class EasyInputMessage(BaseModel):
    content: str | list[ResponseInputContent]
    role: Role
    phase: Phase | None = None
    type: Literal["message"] | None = "message"


class Message(BaseModel):
    content: list[ResponseInputContent]
    role: Literal[Role.user, Role.system, Role.developer]
    status: ItemStatus | None = None
    type: Literal["message"] | None = "message"
