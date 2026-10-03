"""Strict parser for the Gemma 4 publisher's delimited tool-call grammar."""
from __future__ import annotations

import math
import re
from typing import Any


class GemmaToolSyntaxError(ValueError):
    pass


_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_STRING = '<|"|>'
_MAX_NUMBER_CHARS = 128


class _Reader:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0

    def skip_space(self) -> None:
        while self.pos < len(self.source) and self.source[self.pos].isspace():
            self.pos += 1

    def take(self, marker: str) -> bool:
        if self.source.startswith(marker, self.pos):
            self.pos += len(marker)
            return True
        return False

    def expect(self, marker: str) -> None:
        if not self.take(marker):
            raise GemmaToolSyntaxError(f"expected {marker!r} at offset {self.pos}")

    def name(self) -> str:
        match = _NAME.match(self.source, self.pos)
        if match is None:
            raise GemmaToolSyntaxError(f"expected name at offset {self.pos}")
        self.pos = match.end()
        return match.group()

    def value(self, depth: int = 0) -> Any:
        if depth > 16:
            raise GemmaToolSyntaxError("tool argument nesting exceeds 16")
        self.skip_space()
        if self.take(_STRING):
            end = self.source.find(_STRING, self.pos)
            if end < 0:
                raise GemmaToolSyntaxError("unterminated delimited string")
            value = self.source[self.pos:end]
            self.pos = end + len(_STRING)
            return value
        if self.take("{"):
            result: dict[str, Any] = {}
            self.skip_space()
            if self.take("}"):
                return result
            while True:
                key = self.name()
                if key in result:
                    raise GemmaToolSyntaxError(f"duplicate argument {key!r}")
                self.skip_space()
                self.expect(":")
                result[key] = self.value(depth + 1)
                self.skip_space()
                if self.take("}"):
                    return result
                self.expect(",")
                self.skip_space()
        if self.take("["):
            result: list[Any] = []
            self.skip_space()
            if self.take("]"):
                return result
            while True:
                result.append(self.value(depth + 1))
                self.skip_space()
                if self.take("]"):
                    return result
                self.expect(",")
        for token, value in (("true", True), ("false", False), ("null", None)):
            if self.source.startswith(token, self.pos):
                end = self.pos + len(token)
                if end == len(self.source) or self.source[end] in ",}] \t\r\n":
                    self.pos = end
                    return value
        match = _NUMBER.match(self.source, self.pos)
        if match is None:
            raise GemmaToolSyntaxError(f"expected delimited value at offset {self.pos}")
        self.pos = match.end()
        number = match.group()
        if len(number) > _MAX_NUMBER_CHARS:
            raise GemmaToolSyntaxError("tool numeric literal exceeds parser limit")
        try:
            value = float(number) if any(c in number for c in ".eE") else int(number)
        except (OverflowError, ValueError) as exc:
            raise GemmaToolSyntaxError("invalid tool numeric literal") from exc
        if isinstance(value, float) and not math.isfinite(value):
            raise GemmaToolSyntaxError("nonfinite tool numeric literal")
        return value


def parse_gemma_tool_calls(text: str) -> list[dict[str, Any]]:
    """Return complete calls; ordinary text is never searched for executable data.

    A thought channel may precede calls. Calls must then occupy the terminal
    response, optionally followed by the publisher's tool-response stop token.
    """
    if len(text) > 1_000_000:
        raise GemmaToolSyntaxError("tool response exceeds parser limit")
    source = text.strip()
    marker = "<|tool_call>"
    if marker not in source:
        if "<tool_call|>" in source or "<|tool_response>" in source:
            raise GemmaToolSyntaxError("tool delimiter without a complete call")
        return []
    if source.startswith("<|channel>thought\n"):
        end = source.find("<channel|>")
        if end < 0:
            raise GemmaToolSyntaxError("unterminated thought channel")
        source = source[end + len("<channel|>"):].strip()
    reader = _Reader(source)
    calls: list[dict[str, Any]] = []
    while reader.take(marker):
        reader.expect("call:")
        name = reader.name()
        arguments = reader.value()
        if not isinstance(arguments, dict):
            raise GemmaToolSyntaxError("tool arguments must be an object")
        reader.expect("<tool_call|>")
        calls.append({"name": name, "arguments": arguments})
        reader.skip_space()
    if not calls:
        raise GemmaToolSyntaxError("tool call must be a terminal envelope")
    reader.take("<|tool_response>")
    reader.skip_space()
    if reader.pos != len(source):
        raise GemmaToolSyntaxError("text outside terminal tool-call envelope")
    return calls
