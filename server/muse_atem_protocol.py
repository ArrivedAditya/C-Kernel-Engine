"""Parse Muse's recipient-scoped assistant turn and ATEM tool blocks.

Publisher messages may continue across <|eom|>. Only <|eot|> and
<|end_of_text|> terminate a turn; callers must not stop on <|eom|>.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any


class MuseSyntaxError(ValueError):
    """Generated Muse output does not satisfy the declared protocol."""


@dataclass(frozen=True)
class MuseTurn:
    reasoning: str
    final_text: str
    tool_calls: list[dict[str, Any]]


_HEADER = re.compile(r"<\|start\|>assistant(?: to=([^<\s]+))?<\|message\|>")
_INVOKE = re.compile(r'<atem:invoke name="([A-Za-z_][\w.-]*)">')
_PARAM = re.compile(r'<atem:parameter name="([A-Za-z_][\w.-]*)">')
_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")
_ENDS = ("<|eom|>", "<|eot|>", "<|end_of_text|>")


def _json_value(raw: str) -> Any:
    stripped = raw.strip()

    def bounded_integer(value: str) -> int:
        if len(value) > 1000:
            raise ValueError("numeric tool parameter is too long")
        return int(value)

    def bounded_float(value: str) -> float:
        if len(value) > 1000:
            raise ValueError("numeric tool parameter is too long")
        return float(value)

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite numeric value {value}")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(stripped, parse_constant=reject_constant,
                           parse_int=bounded_integer, parse_float=bounded_float,
                           object_pairs_hook=unique_keys)
    except json.JSONDecodeError as exc:
        if _NUMBER.fullmatch(stripped) or stripped in {"NaN", "Infinity", "-Infinity"}:
            raise MuseSyntaxError("invalid numeric tool parameter") from exc
        return raw
    except ValueError as exc:
        raise MuseSyntaxError("invalid JSON tool parameter") from exc

    def finite(item: Any) -> bool:
        if isinstance(item, float):
            return math.isfinite(item)
        if isinstance(item, list):
            return all(finite(child) for child in item)
        if isinstance(item, dict):
            return all(finite(child) for child in item.values())
        return True

    if not finite(value):
        raise MuseSyntaxError("nonfinite numeric tool parameter")
    return value


def _parse_atem(body: str, recipient: str) -> list[dict[str, Any]]:
    opening, closing = "<atem:function_calls>", "</atem:function_calls>"
    body = body.strip()
    if not body.startswith(opening) or not body.endswith(closing):
        raise MuseSyntaxError("tool channel requires one complete ATEM envelope")
    source = body[len(opening):-len(closing)]
    pos = 0
    calls: list[dict[str, Any]] = []
    while pos < len(source):
        pos += len(source[pos:]) - len(source[pos:].lstrip())
        if pos == len(source):
            break
        invoke = _INVOKE.match(source, pos)
        if invoke is None:
            raise MuseSyntaxError("malformed ATEM invoke")
        name = invoke.group(1)
        if name != recipient:
            raise MuseSyntaxError("ATEM invocation does not match recipient")
        pos = invoke.end()
        arguments: dict[str, Any] = {}
        while True:
            pos += len(source[pos:]) - len(source[pos:].lstrip())
            if source.startswith("</atem:invoke>", pos):
                pos += len("</atem:invoke>")
                break
            param = _PARAM.match(source, pos)
            if param is None:
                raise MuseSyntaxError("malformed ATEM parameter")
            key = param.group(1)
            if key in arguments:
                raise MuseSyntaxError(f"duplicate ATEM parameter {key!r}")
            pos = param.end()
            end = source.find("</atem:parameter>", pos)
            if end < 0:
                raise MuseSyntaxError("unterminated ATEM parameter")
            arguments[key] = _json_value(source[pos:end])
            pos = end + len("</atem:parameter>")
        calls.append({"name": name, "arguments": arguments})
    if not calls:
        raise MuseSyntaxError("empty ATEM envelope")
    return calls


def parse_muse_turn(text: str) -> MuseTurn:
    """Parse one generated assistant turn; plain unframed output is final text."""
    if not text:
        return MuseTurn("", "", [])
    raw = text
    if raw.lstrip().startswith("to="):
        raw = "<|start|>assistant " + raw.lstrip()
    if not raw.startswith("<|start|>"):
        return MuseTurn("", text.strip(), [])

    reasoning: list[str] = []
    final: list[str] = []
    calls: list[dict[str, Any]] = []
    pos = 0
    finished = False
    while pos < len(raw):
        header = _HEADER.match(raw, pos)
        if header is None:
            raise MuseSyntaxError("invalid Muse assistant channel header")
        recipient = header.group(1) or "user"
        start = header.end()
        candidates = [(raw.find(marker, start), marker) for marker in (*_ENDS, "<|start|>")]
        end, marker = min(((idx, marker) for idx, marker in candidates if idx >= 0),
                          default=(len(raw), ""))
        if marker == "<|start|>":
            raise MuseSyntaxError("missing message boundary before next channel")
        body = raw[start:end]
        if recipient == "self":
            if marker != "<|eom|>":
                raise MuseSyntaxError("reasoning channel must end with <|eom|>")
            reasoning.append(body)
        elif recipient == "user":
            if marker == "<|eom|>":
                raise MuseSyntaxError("final channel cannot end with <|eom|>")
            final.append(body)
        else:
            calls.extend(_parse_atem(body, recipient))
        pos = end + len(marker)
        if marker in {"<|eot|>", "<|end_of_text|>"}:
            finished = True
            if raw[pos:].strip():
                raise MuseSyntaxError("output follows terminal Muse turn")
            break
        if not marker:
            break
    if not finished and not final and not calls:
        raise MuseSyntaxError("Muse turn has no final answer or tool call")
    if len(final) > 1 or (final and calls):
        raise MuseSyntaxError("Muse turn mixes final answer and tool calls")
    return MuseTurn("".join(reasoning).strip(), "".join(final).strip(), calls)
