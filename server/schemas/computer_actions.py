from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Click(BaseModel):
    button: Literal["left", "right", "wheel", "back", "forward"]
    type: Literal["click"] = "click"
    x: int
    y: int
    keys: list[str] | None = None


class DoubleClick(BaseModel):
    keys: list[str] = []
    type: Literal["double_click"] = "double_click"
    x: int
    y: int


class DragPoint(BaseModel):
    x: int
    y: int


class Drag(BaseModel):
    path: list[DragPoint]
    type: Literal["drag"] = "drag"
    keys: list[str] | None = None


class Keypress(BaseModel):
    keys: list[str]
    type: Literal["keypress"] = "keypress"


class Move(BaseModel):
    type: Literal["move"] = "move"
    x: int
    y: int
    keys: list[str] | None = None


class Screenshot(BaseModel):
    type: Literal["screenshot"] = "screenshot"


class Scroll(BaseModel):
    scroll_x: int
    scroll_y: int
    type: Literal["scroll"] = "scroll"
    x: int
    y: int
    keys: list[str] | None = None


class TypeAction(BaseModel):
    text: str
    type: Literal["type"] = "type"


class Wait(BaseModel):
    type: Literal["wait"] = "wait"


ComputerAction = Annotated[
    Click | DoubleClick | Drag | Keypress | Move | Screenshot | Scroll | TypeAction | Wait,
    Field(discriminator="type"),
]
