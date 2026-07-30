"""Citation annotation shapes accepted by the schema scaffold."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class FileCitation(BaseModel):
    file_id: str
    filename: str
    index: int
    type: Literal["file_citation"] = "file_citation"


class URLCitation(BaseModel):
    end_index: int
    start_index: int
    title: str
    type: Literal["url_citation"] = "url_citation"
    url: str


class ContainerFileCitation(BaseModel):
    container_id: str
    end_index: int
    file_id: str
    filename: str
    start_index: int
    type: Literal["container_file_citation"] = "container_file_citation"


class FilePath(BaseModel):
    file_id: str
    index: int
    type: Literal["file_path"] = "file_path"


Annotation = Annotated[
    FileCitation | URLCitation | ContainerFileCitation | FilePath,
    Field(discriminator="type"),
]
