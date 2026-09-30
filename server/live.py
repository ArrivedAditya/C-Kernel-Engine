"""Live generic OpenAI Responses + tools factory for ``server/``.

Ports the Responses lifecycle from ``version/v8/scripts/ck_serve_v8.py`` into
the server boundary so a generic harness can use ``POST /v1/responses`` with
``function``/``mcp`` tools without importing the scripts tree:

* single-flight native session (HTTP 429 when busy)
* ``previous_response_id`` + ``function_call_output`` multi-turn history
* tool parsing driven by an explicit protocol (tagged JSON, Qwen XML, or bare
  JSON); ``malformed``/``unknown``
  → ``failed``, ``parallel_tool_calls:false`` + >1 → ``incomplete``
* non-stream + SSE streaming with OpenAI event order and ``sequence_number``
* real ``usage`` + ``performance``, capacity preflight, cancel (404/409/500/504)

The server never executes tools server-side. Only function-like tools
(``function``/``mcp``) are parsed back into ``function_call`` items for the
client to execute. Unsupported tool types are rejected before generation.
"""

from __future__ import annotations

import hashlib
import copy
import json
import math
import queue
import re
import threading
import time
import uuid
from datetime import datetime
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import anyio
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

try:
    import jinja2
    import jinja2.sandbox

    _JINJA_AVAILABLE = True
except ImportError:
    jinja2 = None
    _JINJA_AVAILABLE = False

from .routes.conversations import router as conversations_router
from .runtime import load_manifest_templates
from .schemas.common import ResponseStatus
from .schemas.content import ResponseOutputText
from .schemas.output_items import (
    FunctionCall,
    ReasoningItem,
    ReasoningTextContent,
    ResponseOutputMessage,
)
from .schemas.response import CreateResponseRequest
from .session_v8 import (
    CK_SESSION_REQUEST_RAW_PROMPT,
    SessionBusyError,
    stop_reason_name,
    truncate_stop_markers,
)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

C_GRAY = "\033[38;5;242m"
C_ORANGE = "\033[38;5;208m"
C_RESET = "\033[0m"

# How long a second request waits for the single-flight lock before the
# server answers 429 (harness retry storm mitigation).
_FLIGHT_WAIT_SECONDS = 30.0

_IGNORED_TOOL_TYPES = frozenset(
    {
        "file_search",
        "computer",
        "computer_use_preview",
        "web_search",
        "code_interpreter",
        "image_generation",
    }
)
_FUNCTION_LIKE_TYPES = frozenset({"function", "mcp"})

# Legacy name (mirrors the ck_serve_v8.py helper); canonical is
# server.session_v8.truncate_stop_markers.
_truncate_stop_markers = truncate_stop_markers


def _harness_error(
    status_code: int,
    message: str,
    *,
    err_type: str,
    code: str,
    retry_after: float | None = None,
) -> HTTPException:
    """HTTP error carrying an OpenAI-shaped ``error`` object.

    The app-level exception handler renders this as
    ``{"error": {...}, "detail": message}`` so harnesses can parse ``error``
    while existing ``detail`` readers keep working.
    """
    # Fractional admission waits must not become Retry-After: 0.  A harness
    # treating that as an immediate retry can recreate the busy-request storm.
    headers = (
        {"Retry-After": str(max(1, math.ceil(retry_after)))}
        if retry_after is not None else None
    )
    return HTTPException(
        status_code=status_code,
        detail={"error": {"message": message, "type": err_type, "code": code}},
        headers=headers,
    )


def _log_rejection(server_model: str, body: Any, exc: HTTPException) -> None:
    """One warn line per rejected request (rejections otherwise stay silent)."""
    try:
        tools_n: Any = len(getattr(body, "tools", None) or [])
    except Exception:
        tools_n = "?"
    if isinstance(exc.detail, dict) and isinstance(exc.detail.get("error"), dict):
        err = exc.detail["error"]
        code: Any = err.get("code", "?")
        message: Any = err.get("message", "")
    else:
        code, message = "?", exc.detail
    print(
        f"{C_ORANGE}{server_model} - rejected {exc.status_code} "
        f"({code}): {message} "
        f"[model={getattr(body, 'model', '?')} "
        f"max_output_tokens={getattr(body, 'max_output_tokens', '?')} "
        f"tools={tools_n} "
        f"previous_response_id={getattr(body, 'previous_response_id', None)}]{C_RESET}",
        flush=True,
    )


# --- prompt helpers -----------------------------------------------------------


def _image_identifier(part: Any) -> str:
    for attr in ("image_url", "file_id", "file_url", "url"):
        try:
            value = getattr(part, attr, None)
        except (AttributeError, ValueError, TypeError):
            value = None
        if isinstance(value, str) and value.strip():
            return value.strip()
    try:
        raw = getattr(part, "model_dump", None)
        if callable(raw):
            data = raw()
            for key in ("image_url", "file_id", "file_url", "url"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, dict):
                    nested = value.get("url")
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()
    except (AttributeError, ValueError, TypeError):
        pass
    return ""


_MAX_VISION_IMAGES = 8
_MAX_IMAGE_REF_CHARS = 5_000_000


def _check_image_ref(ident: str) -> str:
    ident = (ident or "").strip()
    if not ident:
        raise ValueError("input_image is missing image_url or file_id")
    if len(ident) > _MAX_IMAGE_REF_CHARS:
        raise ValueError(
            f"input_image exceeds {_MAX_IMAGE_REF_CHARS} characters; "
            "upload a smaller image"
        )
    lowered = ident.lower()
    if lowered.startswith(("http://", "https://", "data:image/")):
        return ident
    raise ValueError(
        "input_image must be an http(s) URL or data:image/... URL, "
        "not raw base64 or a file path"
    )


def _ordered_content_parts(content: Any) -> Any:
    """Render-only normalization preserving interleaved text and image order."""
    if content is None or isinstance(content, str):
        return content or ""
    if not isinstance(content, list):
        raise ValueError("message content must be text or ordered parts")
    parts = []
    for part in content:
        data = part if isinstance(part, dict) else part.model_dump()
        kind = data.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": data.get("text", "")})
        elif kind in ("input_image", "image", "image_url"):
            ident = data.get("image_url", data.get("image", ""))
            if isinstance(ident, dict):
                ident = ident.get("url", "")
            ident = _check_image_ref(str(ident))
            parts.append({"type": "image", "image_url": ident, "image": ident})
        else:
            raise ValueError(f"unsupported content part {kind!r}")
    return parts


def _content_parts(content: Any) -> tuple[str, list[str]]:
    parts = _ordered_content_parts(content)
    if isinstance(parts, str):
        return parts, []
    return ("\n".join(p["text"] for p in parts if p["type"] == "text"),
            [p["image"] for p in parts if p["type"] == "image"])


def _reject_live_media(value: Any) -> None:
    """Inspect original typed input, including tool results, before extraction."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, list):
        for item in value:
            _reject_live_media(item)
    elif isinstance(value, dict):
        kind = value.get("type")
        if kind in {"input_image", "image", "image_url", "input_file", "file",
                    "input_audio", "audio", "video", "input_video"}:
            raise _harness_error(
                422, f"typed {kind} input is unsupported by this text-only native session; "
                "no connected generated media pipeline is available",
                err_type="invalid_request_error", code="unsupported_media",
            )
        for item in value.values():
            _reject_live_media(item)


def _extract_prompt(body: Any) -> str:
    if body.input is None:
        return ""
    if isinstance(body.input, str):
        return body.input
    parts: list[str] = []
    for item in body.input:
        if isinstance(item, str):
            parts.append(item)
            continue
        content = getattr(item, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            try:
                text, images = _content_parts(content)
            except ValueError:
                continue
            if text:
                parts.append(text)
            for ident in images:
                parts.append(f"[image: {ident}]" if ident else "[image]")
    return "\n".join(parts)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    try:
        text, images = _content_parts(content)
    except ValueError:
        raise
    if images:
        markers = ["[image: " + i + "]" if i else "[image]" for i in images]
        return "\n".join([text, *markers]) if text else "\n".join(markers)
    return text


def _select_template(
    chat_template: str | None,
    chat_templates: dict[str, str] | None,
    *,
    has_tools: bool,
) -> str | None:
    """Return the template the renderer will use (single source of truth).

    Select the tool variant when tools are attached, otherwise the base.
    """
    if has_tools and isinstance(chat_templates, dict):
        for key in ("tool_use", "tools", "default"):
            candidate = chat_templates.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    if isinstance(chat_template, str) and chat_template.strip():
        return chat_template
    return None


def _input_chat_messages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    messages: list[dict[str, Any]] = []
    for item in value:
        item_type = getattr(item, "type", None)
        if item_type == "message" or (
            item_type is None and getattr(item, "role", None) is not None
        ):
            role = getattr(item, "role", "user")
            role = getattr(role, "value", role)
            role_str = str(role)
            content = _ordered_content_parts(item.content)
            if isinstance(content, list) and all(p["type"] == "text" for p in content):
                content = "\n".join(p["text"] for p in content)
            message: dict[str, Any] = {
                "role": role_str,
                "content": content,
            }
            if role_str == "assistant":
                # Native templates (e.g. Qwen3 line 48
                # `{%- if message.tool_calls %}`) read this key unguarded;
                # under StrictUndefined a missing key aborts the whole
                # render, surfacing as 422 downstream.
                message["tool_calls"] = []
            messages.append(message)
        elif item_type == "function_call":
            try:
                arguments = json.loads(item.arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"function_call {item.call_id!r} has invalid JSON arguments"
                ) from exc
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {"name": item.name, "arguments": arguments},
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "content": _content_text(item.output),
                    "tool_call_id": item.call_id,
                }
            )
    return messages


def _usage(
    input_tokens: int, output_tokens: int, reasoning_tokens: int = 0
) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


def _sse(event_type: str, data: Any) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


def _performance_profile(result: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = int(result.get("prompt_tokens") or 0)
    generated_tokens = int(result.get("generated_tokens") or 0)
    prefill_ms = float(result.get("prefill_time_ms") or 0.0)
    decode_ms = float(result.get("decode_time_ms") or 0.0)
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prefill_ms": prefill_ms,
        "prefill_ms_per_token": round(prefill_ms / prompt_tokens, 2)
        if prompt_tokens > 0
        else 0.0,
        "prefill_tokens_per_sec": round(1000 * prompt_tokens / prefill_ms, 2)
        if prefill_ms > 0
        else 0.0,
        "decode_ms": decode_ms,
        "decode_ms_per_token": round(decode_ms / generated_tokens, 2)
        if generated_tokens > 0
        else 0.0,
        "decode_tokens_per_sec": round(1000 * generated_tokens / decode_ms, 2)
        if decode_ms > 0
        else 0.0,
        "total_ms": round(prefill_ms + decode_ms, 2),
        "stop_reason": stop_reason_name(result.get("stop_reason")),
    }


def _log_performance(model: str, perf: dict[str, Any] | None) -> None:
    if not perf:
        return
    line = (
        f"eval time = {perf['prefill_ms']:.1f} ms prompt, "
        f"{perf['decode_ms']:.1f} ms decode "
        f"({perf['decode_ms_per_token']:.2f} ms/token, "
        f"{perf['decode_tokens_per_sec']:.1f} tokens/s), "
        f"total {perf['total_ms']:.1f} ms, stop: {perf['stop_reason']}"
    )
    print(f"{C_GRAY}{model} - {line}{C_RESET}", flush=True)


def _marker_index(lowered: str, marker: str) -> int:
    anchored = lowered.find("\n" + marker)
    if anchored != -1:
        return anchored + 1
    if lowered.startswith(marker):
        return 0
    return lowered.find(marker)


def split_thinking(text: str) -> tuple[str, str]:
    if not text:
        return "", text
    lowered = text.lower()
    open_idx = _marker_index(lowered, _THINK_OPEN)
    close_idx = _marker_index(lowered, _THINK_CLOSE)
    if open_idx != -1:
        if close_idx == -1:
            return text[open_idx + len(_THINK_OPEN) :].strip(), ""
        if open_idx + len(_THINK_OPEN) <= close_idx:
            return (
                text[open_idx + len(_THINK_OPEN) : close_idx].strip(),
                text[close_idx + len(_THINK_CLOSE) :].lstrip(),
            )
    if close_idx != -1:
        return text[:close_idx].strip(), text[close_idx + len(_THINK_CLOSE) :].lstrip()
    return "", text


def _prompt_opens_thinking(generation_prefix: str) -> bool:
    """Inspect only the isolated generation suffix, never arbitrary prompt text."""
    return generation_prefix.rstrip().lower().endswith(_THINK_OPEN)


class _StreamThinkSplitter:
    _KEEP = len(_THINK_CLOSE) + 2

    def __init__(self, *, start_thinking: bool = False) -> None:
        self._look = ""
        self._mode = "thinking" if start_thinking else "undetermined"
        self._thinking_lstrip = True
        self._answer_lstrip = True
        self._strip_leading_open = start_thinking

    def feed(self, chunk: str):
        if self._mode == "answer":
            if self._answer_lstrip:
                chunk = chunk.lstrip()
                if not chunk:
                    return
                self._answer_lstrip = False
            yield ("answer", chunk)
            return
        buf = self._look + chunk
        if self._mode == "thinking" and self._strip_leading_open:
            head = buf.lstrip()
            if head[: len(_THINK_OPEN)].lower() == _THINK_OPEN:
                buf = head[len(_THINK_OPEN) :]
                self._look = ""
                self._strip_leading_open = False
            elif len(head) < len(_THINK_OPEN) and _THINK_OPEN.startswith(
                head.lower()
            ):
                self._look = buf
                return
            else:
                self._strip_leading_open = False
        if self._mode == "undetermined":
            open_idx = _marker_index(buf.lower(), _THINK_OPEN)
            close_idx = _marker_index(buf.lower(), _THINK_CLOSE)
            if open_idx != -1 and (close_idx == -1 or open_idx < close_idx):
                buf = buf[open_idx + len(_THINK_OPEN) :]
                self._mode = "thinking"
            elif close_idx != -1:
                pre = buf[:close_idx].strip()
                if pre:
                    yield ("thinking", pre)
                self._mode = "answer"
                rest = buf[close_idx + len(_THINK_CLOSE) :].lstrip()
                self._look = ""
                if rest:
                    self._answer_lstrip = False
                    yield ("answer", rest)
                return
            else:
                self._look = buf
                return
        close_idx = _marker_index(buf.lower(), _THINK_CLOSE)
        if close_idx != -1:
            pre = buf[:close_idx]
            if pre:
                if self._thinking_lstrip:
                    pre = pre.lstrip()
                    self._thinking_lstrip = False
                pre = pre.rstrip()
                if pre:
                    yield ("thinking", pre)
            self._mode = "answer"
            rest = buf[close_idx + len(_THINK_CLOSE) :]
            self._look = ""
            if rest:
                if self._answer_lstrip:
                    rest = rest.lstrip()
                    if not rest:
                        return
                    self._answer_lstrip = False
                yield ("answer", rest)
            return
        emit_len = len(buf) - self._KEEP
        if emit_len > 0:
            head, buf = buf[:emit_len], buf[emit_len:]
            if self._thinking_lstrip:
                head = head.lstrip()
                if head:
                    self._thinking_lstrip = False
            if head:
                yield ("thinking", head)
        self._look = buf

    def flush(self):
        if self._mode == "answer" or not self._look:
            return
        text = self._look
        self._look = ""
        if self._mode == "undetermined":
            yield ("answer", text)
            return
        if self._thinking_lstrip:
            text = text.lstrip()
            if not text:
                return
            self._thinking_lstrip = False
        if self._mode == "thinking":
            text = text.rstrip()
            if not text:
                return
        yield ("thinking", text)


# --- chat contract ------------------------------------------------------------


def _resolve_contract_thinking_overrides(
    contract: dict[str, Any], thinking_mode: str | None
) -> tuple[str, str]:
    assistant_generation_prefix = str(contract.get("assistant_generation_prefix") or "")
    last_user_prefix = str(contract.get("last_user_prefix") or "")
    requested_mode = str(thinking_mode or "auto").strip().lower()
    default_mode = str(contract.get("thinking_mode_default") or "").strip().lower()
    resolved_mode = default_mode if requested_mode in {"", "auto"} else requested_mode
    assistant_by_mode = contract.get("assistant_generation_prefix_by_thinking_mode")
    if isinstance(assistant_by_mode, dict):
        override = assistant_by_mode.get(resolved_mode)
        if isinstance(override, str):
            assistant_generation_prefix = override
    last_user_prefix_by_mode = contract.get("last_user_prefix_by_thinking_mode")
    if isinstance(last_user_prefix_by_mode, dict):
        override = last_user_prefix_by_mode.get(resolved_mode)
        if isinstance(override, str):
            last_user_prefix = override
    return assistant_generation_prefix, last_user_prefix


def _format_prompt_with_chat_contract(
    prompt: str,
    contract: dict[str, Any] | None,
    *,
    thinking_mode: str = "auto",
    system_prompt: str | None = None,
) -> str:
    if not isinstance(contract, dict):
        return str(prompt or "")
    role_labels = (
        contract.get("role_labels")
        if isinstance(contract.get("role_labels"), dict)
        else {}
    )
    turn_prefix = str(contract.get("turn_prefix") or "")
    turn_suffix = str(contract.get("turn_suffix") or "")
    system_prompt_mode = (
        str(contract.get("system_prompt_mode") or "disabled").strip().lower()
    )
    system_prompt_separator = str(contract.get("system_prompt_separator") or "\n\n")
    default_system_prompt = str(contract.get("default_system_prompt") or "")
    inject_default_system_prompt = bool(contract.get("inject_default_system_prompt"))
    bos_prefix = str(contract.get("force_bos_text_if_tokenizer_add_bos_false") or "")
    suppression_markers = [
        str(m).lower()
        for m in list(contract.get("last_user_prefix_suppression_markers") or [])
        if str(m or "").strip()
    ]
    assistant_generation_prefix, last_user_prefix = (
        _resolve_contract_thinking_overrides(contract, thinking_mode)
    )
    user_text = str(prompt or "")
    if last_user_prefix:
        lowered = user_text.lower()
        if last_user_prefix.lower() not in lowered and not any(
            m in lowered for m in suppression_markers
        ):
            user_text = f"{last_user_prefix}{user_text}"
    system_text = str(system_prompt or "")
    if not system_text and inject_default_system_prompt:
        system_text = default_system_prompt
    if system_text and system_prompt_mode == "prepend_first_user":
        user_text = (
            f"{system_text}{system_prompt_separator}{user_text}"
            if user_text
            else system_text
        )
        system_text = ""

    def _render_turn(role: str, content: str) -> str:
        label = str(role_labels.get(role) or role)
        return f"{turn_prefix.replace('{role}', label)}{content}{turn_suffix}"

    formatted = ""
    if bos_prefix:
        formatted += bos_prefix
    if system_text and system_prompt_mode == "dedicated_turn":
        formatted += _render_turn("system", system_text)
    formatted += _render_turn("user", user_text)
    formatted += assistant_generation_prefix
    return formatted if formatted else user_text


def _load_runtime_chat_contract(run_dir: str | Path) -> dict[str, Any] | None:
    """No chat contract is loaded from disk (pure-Jinja runtime)."""
    return None


def _load_runtime_templates(
    run_dir: str | Path,
) -> tuple[str | None, dict[str, str] | None, dict[str, Any] | None]:
    """Return (chat_template, chat_templates, None) from the sidecar files."""
    return load_manifest_templates(Path(run_dir))


def _load_builtin_chat_contract(
    template_name: str | None,
    *,
    _seen: set[str] | None = None,
) -> dict[str, Any] | None:
    """Backward-compat stub: the manifest is the single source of truth."""
    return None


# --- tools --------------------------------------------------------------------


def _effective_tools(body: Any) -> list[Any]:
    tools = getattr(body, "tools", None) if body is not None else None
    if not tools:
        return []
    return [t for t in tools if getattr(t, "type", None) in _FUNCTION_LIKE_TYPES]


def _has_function_tools(body: Any) -> bool:
    return bool(_effective_tools(body))


def _resolve_thinking_mode(body: Any) -> str:
    """Single thinking on/off authority: ``"visible"`` or ``"suppressed"``.

    ``"none"`` means reasoning off even when a reasoning object is present;
    ``"default"`` (or any graded effort, or an absent effort) means on. The
    engine has no effort gradations — on/off is the only distinction. Both
    prompt formatting and response handling key off this value.
    """
    reasoning = getattr(body, "reasoning", None) if body is not None else None
    if reasoning is None:
        return "suppressed"
    if getattr(reasoning, "effort", None) == "none":
        return "suppressed"
    return "visible"


def _has_tool_support(
    chat_template: str | None, chat_templates: dict[str, str] | None,
    *, tool_protocol: str | None = None,
) -> bool:
    """A template's spelling never certifies tool support by itself."""
    has_template = bool(isinstance(chat_template, str) and chat_template.strip()) or bool(
        isinstance(chat_templates, dict)
        and any(isinstance(value, str) and value.strip() for value in chat_templates.values())
    )
    return has_template and tool_protocol in {
        "tagged_json", "bare_json", "qwen_xml", "qwen_code_xml", "qwen_code_xml_raw_v2",
    }


#: Tool-call syntax emitted by the model-native Jinja template: Qwen3-style
#: ``<tool_call>{"name": ..., "arguments": ...}</tool_call>`` blocks.
_TOOL_SYNTAX_TOOL_CALL_JSON = "tool_call_json"
#: Plain JSON tool calls with no template-declared tag wrapper.
_TOOL_SYNTAX_JSON = "json"
_TOOL_SYNTAX_QWEN_XML = "qwen_xml"
_TOOL_SYNTAX_QWEN_CODE_XML = "qwen_code_xml"
_TOOL_SYNTAX_QWEN_CODE_XML_RAW_V2 = "qwen_code_xml_raw_v2"
_TOOL_SYNTAX_NONE = "none"


class TemplateRenderError(RuntimeError):
    """The selected template failed; callers must not substitute another format."""


def _tool_syntax_for_protocol(protocol: str | None) -> str:
    if protocol in (None, "none"):
        return _TOOL_SYNTAX_NONE
    if protocol == "tagged_json":
        return _TOOL_SYNTAX_TOOL_CALL_JSON
    if protocol == "bare_json":
        return _TOOL_SYNTAX_JSON
    if protocol == "qwen_xml":
        return _TOOL_SYNTAX_QWEN_XML
    if protocol == "qwen_code_xml":
        return _TOOL_SYNTAX_QWEN_CODE_XML
    if protocol == "qwen_code_xml_raw_v2":
        return _TOOL_SYNTAX_QWEN_CODE_XML_RAW_V2
    raise ValueError(f"unsupported tool protocol {protocol!r}")


def _detect_template_tool_syntax(
    chat_template: str | None, chat_templates: dict[str, str] | None
) -> str:
    """Legacy diagnostic helper; serving requires an explicit protocol.

    A template containing ``<tool_call>`` does not establish whether its
    payload is tagged JSON or Qwen's XML parameter form.
    """
    texts: list[str] = []
    if isinstance(chat_template, str) and chat_template.strip():
        texts.append(chat_template)
    if isinstance(chat_templates, dict):
        for value in chat_templates.values():
            if isinstance(value, str) and value.strip():
                texts.append(value)
    for text in texts:
        if "<tool_call" in text.lower():
            return _TOOL_SYNTAX_TOOL_CALL_JSON
    return _TOOL_SYNTAX_JSON


def _split_generated_thinking(text: str, generation_prefix: str) -> tuple[str, str]:
    splitter = _StreamThinkSplitter(start_thinking=_prompt_opens_thinking(generation_prefix))
    thinking, answer = [], []
    for state, delta in [*splitter.feed(text), *splitter.flush()]:
        (thinking if state == "thinking" else answer).append(delta)
    return "".join(thinking).strip(), "".join(answer).strip()


def _render_with_chat_templates(
    chat_template: str | None,
    chat_templates: dict[str, str] | None,
    messages: list[dict[str, Any]],
    body: Any,
    chat_contract: dict[str, Any] | None = None,
    effective_thinking: str = "suppressed",
    *, add_generation_prompt: bool = True,
    renderer_tokens: dict[str, str] | None = None,
    render_time: datetime | None = None,
) -> str | None:
    tmpl_str = _select_template(
        chat_template,
        chat_templates,
        has_tools=bool(body is not None and getattr(body, "tools", None)),
    )
    if not tmpl_str:
        return None
    digest = hashlib.sha256(tmpl_str.encode("utf-8")).hexdigest()
    if not _JINJA_AVAILABLE:
        raise TemplateRenderError(
            f"selected chat template sha256:{digest} cannot render: Jinja2 is unavailable"
        )
    try:
        instructions = getattr(body, "instructions", None) if body is not None else None
        if isinstance(instructions, str) and instructions.strip():
            messages = [{"role": "system", "content": instructions}, *messages]
        tools = None
        if body is not None and getattr(body, "tools", None):
            tools = [t.model_dump() for t in body.tools]

        def _raise_exception(message: str = "") -> None:
            raise ValueError(str(message))

        env = jinja2.sandbox.SandboxedEnvironment(
            undefined=jinja2.StrictUndefined, autoescape=False
        )
        captured_time = render_time if render_time is not None else datetime.now()
        env.globals["strftime_now"] = lambda format: captured_time.strftime(format)
        tmpl = env.from_string(tmpl_str)
        return str(
            tmpl.render(
                messages=messages,
                tools=tools,
                tool_choice=getattr(body, "tool_choice", None)
                if body is not None
                else None,
                enable_thinking=(effective_thinking == "visible"),
                add_generation_prompt=add_generation_prompt,
                add_vision_id=False,
                raise_exception=_raise_exception,
                **(renderer_tokens or {}),
            )
        )
    except Exception as exc:
        raise TemplateRenderError(
            f"selected chat template sha256:{digest} cannot render: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


_TOOL_SCHEMA_TYPES = {"string", "integer", "number", "boolean", "object", "array", "null"}
_TOOL_SCHEMA_KEYS = {
    "type", "properties", "required", "additionalProperties", "items",
    "enum", "description", "title", "default", "minimum", "maximum",
    "minLength", "maxLength",
}


def _validate_tool_schema_subset(schema: Any, path: str = "parameters") -> None:
    """Reject schemas the server cannot validate before a model sees them."""
    if not isinstance(schema, dict):
        raise ValueError(f"{path} must be an object schema")
    unsupported = set(schema) - _TOOL_SCHEMA_KEYS
    if unsupported:
        raise ValueError(f"{path} has unsupported schema keywords: {', '.join(sorted(unsupported))}")
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if kind is not None and (
        not kinds or any(not isinstance(item, str) or item not in _TOOL_SCHEMA_TYPES for item in kinds)
        or len(set(kinds)) != len(kinds)
    ):
        raise ValueError(f"{path} has unsupported type {kind!r}")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise ValueError(f"{path}.enum must be an array")
    for bound in ("minimum", "maximum"):
        if bound in schema:
            value = schema[bound]
            if kind is None or not ({"integer", "number"} & set(kinds)):
                raise ValueError(f"{path}.{bound} requires a numeric type")
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{path}.{bound} must be a finite number")
    if "minimum" in schema and "maximum" in schema and schema["minimum"] > schema["maximum"]:
        raise ValueError(f"{path}.minimum exceeds maximum")
    for bound in ("minLength", "maxLength"):
        if bound in schema:
            value = schema[bound]
            if kind is None or "string" not in kinds:
                raise ValueError(f"{path}.{bound} requires a string type")
            if type(value) is not int or value < 0:
                raise ValueError(f"{path}.{bound} must be a nonnegative integer")
    if "minLength" in schema and "maxLength" in schema and schema["minLength"] > schema["maxLength"]:
        raise ValueError(f"{path}.minLength exceeds maxLength")
    if "properties" in schema:
        if kind not in (None, "object") or not isinstance(schema["properties"], dict):
            raise ValueError(f"{path}.properties requires an object schema")
        for key, child in schema["properties"].items():
            if not isinstance(key, str):
                raise ValueError(f"{path}.properties keys must be strings")
            _validate_tool_schema_subset(child, f"{path}.properties.{key}")
    if "required" in schema:
        required = schema["required"]
        if kind not in (None, "object") or not isinstance(required, list) or not all(
            isinstance(key, str) for key in required
        ):
            raise ValueError(f"{path}.required must be an array of property names")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        raise ValueError(f"{path}.additionalProperties must be boolean")
    if "items" in schema:
        if kind not in (None, "array"):
            raise ValueError(f"{path}.items requires an array schema")
        _validate_tool_schema_subset(schema["items"], f"{path}.items")


def _tool_value_matches_schema(value: Any, schema: dict[str, Any]) -> bool:
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    matches = {
        "string": lambda: isinstance(value, str),
        "integer": lambda: type(value) is int,
        "number": lambda: type(value) is int or (type(value) is float and math.isfinite(value)),
        "boolean": lambda: type(value) is bool,
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "null": lambda: value is None,
    }
    if kind is not None and not any(matches[item]() for item in kinds):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if type(value) in (int, float):
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return False
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(key not in properties for key in value):
            return False
        if any(
            not _tool_value_matches_schema(item, properties[key])
            for key, item in value.items() if key in properties
        ):
            return False
    if isinstance(value, list) and "items" in schema:
        if any(not _tool_value_matches_schema(item, schema["items"]) for item in value):
            return False
    return True


def _extract_tool_calls_from_text(
    text: str,
    allowed_names: set[str] | None,
    *,
    tool_syntax: str = _TOOL_SYNTAX_TOOL_CALL_JSON,
    tool_parameters: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Parse model-native tool calls from generated text.

    The wire format is the explicitly selected protocol: tagged JSON or a
    whole-response bare JSON envelope. Ordinary prose containing an example
    JSON object is never mined for a tool call.
    """
    if not text or not text.strip():
        return [], None, None
    for name, schema in (tool_parameters or {}).items():
        try:
            _validate_tool_schema_subset(schema, f"tool {name!r} parameters")
        except ValueError as exc:
            return [], "malformed", str(exc)

    def strict_json_loads(source: str) -> Any:
        def reject_constant(value: str) -> Any:
            raise ValueError(f"nonfinite JSON value {value}")

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key!r}")
                result[key] = value
            return result

        return json.loads(
            source, parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )

    def declared_parameter_value(name: str, key: str, value: str) -> Any:
        declared = (tool_parameters or {}).get(name, {})
        properties = declared.get("properties", {}) if isinstance(declared, dict) else {}
        property_schema = properties.get(key, {}) if isinstance(properties, dict) else {}
        expected_type = property_schema.get("type") if isinstance(property_schema, dict) else None
        types = expected_type if isinstance(expected_type, list) else [expected_type]
        if expected_type is None or types == ["string"]:
            return value
        if "null" in types and value.strip() == "null":
            return None
        if "string" in types:
            return value
        try:
            parsed = strict_json_loads(value.strip())
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid {expected_type} tool parameter {key!r}") from exc
        if not _tool_value_matches_schema(parsed, {"type": expected_type}):
            raise ValueError(f"invalid {expected_type} tool parameter {key!r}")
        return parsed

    stripped = text.strip()
    if tool_syntax == _TOOL_SYNTAX_NONE:
        return [], None, None
    if tool_syntax in {
        _TOOL_SYNTAX_TOOL_CALL_JSON, _TOOL_SYNTAX_QWEN_XML,
        _TOOL_SYNTAX_QWEN_CODE_XML,
    }:
        tool_blocks: list[str] = []
        for m in re.finditer(
            r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL | re.IGNORECASE
        ):
            tool_blocks.append(m.group(1).strip())
        if text.lower().count("<tool_call>") != len(tool_blocks):
            return [], "malformed", "malformed tool call: missing closing </tool_call>"
        remainder = re.sub(
            r"<tool_call>.*?</tool_call>", "", text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if re.search(r"</?\s*(?:tool[_\s-]|function\s*=|parameter\s*=)", remainder, re.IGNORECASE):
            return [], "malformed", "malformed tool call delimiter outside a complete block"
    else:
        tool_blocks = []
    if tool_syntax == _TOOL_SYNTAX_QWEN_CODE_XML_RAW_V2 and stripped.startswith("<tool_call"):
        return [], "malformed", "qwen_code_xml_raw_v2 requires a function_calls envelope"
    candidates: list[str | dict[str, Any]] = []
    if tool_syntax == _TOOL_SYNTAX_QWEN_CODE_XML and stripped.startswith("<function_calls"):
        try:
            if "<!" in stripped:
                raise ValueError("XML declarations are not allowed in tool calls")
            root = ET.fromstring(stripped)
            if root.tag != "function_calls" or root.attrib or (root.text or "").strip():
                raise ValueError("expected a function_calls root")
            if not len(root):
                raise ValueError("empty function_calls envelope")
            for invoke in root:
                if invoke.tag != "invoke" or set(invoke.attrib) != {"name"} or (invoke.text or "").strip():
                    raise ValueError("invalid invoke element")
                name = invoke.attrib["name"]
                if not re.fullmatch(r"[A-Za-z_][\w.-]*", name):
                    raise ValueError("invalid function name")
                parameters: dict[str, Any] = {}
                for param in invoke:
                    if param.tag != "parameter" or set(param.attrib) != {"name"} or len(param):
                        raise ValueError("invalid parameter element")
                    key = param.attrib["name"]
                    if not re.fullmatch(r"[A-Za-z_][\w.-]*", key) or key in parameters:
                        raise ValueError(f"duplicate or invalid tool parameter {key!r}")
                    parameters[key] = declared_parameter_value(name, key, param.text or "")
                    if (param.tail or "").strip():
                        raise ValueError("text outside a parameter element")
                if (invoke.tail or "").strip():
                    raise ValueError("text outside an invoke element")
                candidates.append({"name": name, "arguments": parameters})
        except (ET.ParseError, ValueError) as exc:
            return [], "malformed", f"malformed function_calls XML: {exc}"
    elif tool_syntax == _TOOL_SYNTAX_QWEN_CODE_XML and "<function_calls" in stripped:
        return [], "malformed", "function_calls XML must be the entire response"
    xml_envelope = stripped
    if tool_syntax == _TOOL_SYNTAX_QWEN_CODE_XML_RAW_V2 and not stripped.startswith("<function_calls"):
        marker = "<function_calls"
        start = stripped.find(marker)
        if start >= 0:
            preamble = stripped[:start]
            # This protocol permits one short plain-text introduction before a
            # terminal XML envelope. It never mines inline examples or fenced
            # code for executable calls.
            if (
                not preamble.endswith("\n\n")
                or len(preamble) > 1024
                or any(char in preamble for char in "<>`")
                or stripped.count(marker) != 1
            ):
                return [], "malformed", "function_calls XML must be a terminal envelope"
            xml_envelope = stripped[start:]
    if tool_syntax == _TOOL_SYNTAX_QWEN_CODE_XML_RAW_V2 and xml_envelope.startswith("<function_calls"):
        try:
            opening, closing = "<function_calls>", "</function_calls>"
            if not xml_envelope.startswith(opening) or not xml_envelope.endswith(closing):
                raise ValueError("expected a complete function_calls envelope")
            # This is a delimited tool protocol, not general XML. Treat the
            # parameter body as raw text so edit arguments retain HTML, code,
            # ampersands, whitespace, and comments byte-for-byte. The closing
            # </parameter> delimiter itself cannot occur in a raw value.
            body = xml_envelope[len(opening):-len(closing)]
            pos = 0
            while pos < len(body):
                pos += len(body[pos:]) - len(body[pos:].lstrip())
                if pos == len(body):
                    break
                invoke = re.match(r'<invoke name="([A-Za-z_][\w.-]*)">', body[pos:])
                if invoke is None:
                    raise ValueError("invalid invoke element")
                name = invoke.group(1)
                pos += invoke.end()
                parameters: dict[str, Any] = {}
                while True:
                    pos += len(body[pos:]) - len(body[pos:].lstrip())
                    if body.startswith("</invoke>", pos):
                        pos += len("</invoke>")
                        break
                    param = re.match(r'<parameter name="([A-Za-z_][\w.-]*)">', body[pos:])
                    if param is None:
                        raise ValueError("invalid parameter element")
                    key = param.group(1)
                    if key in parameters:
                        raise ValueError(f"duplicate tool parameter {key!r}")
                    pos += param.end()
                    end = body.find("</parameter>", pos)
                    if end < 0:
                        raise ValueError(f"unterminated tool parameter {key!r}")
                    parameters[key] = declared_parameter_value(name, key, body[pos:end])
                    pos = end + len("</parameter>")
                candidates.append({"name": name, "arguments": parameters})
            if not candidates:
                raise ValueError("empty function_calls envelope")
        except ValueError as exc:
            return [], "malformed", f"malformed function_calls XML: {exc}"
    elif tool_syntax == _TOOL_SYNTAX_QWEN_CODE_XML_RAW_V2 and "<function_calls" in stripped:
        return [], "malformed", "function_calls XML must be a terminal envelope"
    if tool_blocks and tool_syntax in {
        _TOOL_SYNTAX_QWEN_XML, _TOOL_SYNTAX_QWEN_CODE_XML,
    }:
        for block in tool_blocks:
            match = re.fullmatch(
                r"\s*<function=([A-Za-z_][\w.-]*)>(.*?)</function>\s*",
                block, flags=re.DOTALL,
            )
            if match is None:
                return [], "malformed", "malformed Qwen XML tool call"
            name, body = match.groups()
            parameters: dict[str, Any] = {}
            parameter_pattern = re.compile(
                r"<parameter=([A-Za-z_][\w.-]*)>(.*?)</parameter>", re.DOTALL
            )
            for param in parameter_pattern.finditer(body):
                key, value = param.groups()
                if key in parameters:
                    return [], "malformed", f"duplicate tool parameter {key!r}"
                # Native Qwen XML commonly puts one framing newline on each
                # side of the value. Remove only that pair; spaces and any
                # further newlines remain part of string arguments.
                if value.startswith("\r\n") and value.endswith("\r\n"):
                    value = value[2:-2]
                elif value.startswith("\n") and value.endswith("\n"):
                    value = value[1:-1]
                try:
                    parameters[key] = declared_parameter_value(name, key, value)
                except ValueError as exc:
                    return [], "malformed", str(exc)
            if parameter_pattern.sub("", body).strip():
                return [], "malformed", "malformed Qwen XML tool parameters"
            candidates.append({"name": name, "arguments": parameters})
    elif tool_blocks:
        # Native template syntax: each block must hold a JSON tool-call
        # object (or list of them). Anything else is malformed output.
        candidates = list(tool_blocks)
    elif tool_syntax == _TOOL_SYNTAX_JSON and stripped.startswith(("{", "[")):
        # Bare JSON is meaningful only when the *entire* response is a tool
        # envelope. Never mine an example out of ordinary assistant prose.
        candidates = [stripped]
    tool_calls: list[dict[str, Any]] = []
    for snippet in candidates:
        if isinstance(snippet, dict):
            obj = snippet
        else:
            try:
                obj = strict_json_loads(snippet)
            except (json.JSONDecodeError, ValueError) as exc:
                return [], "malformed", f"malformed tool call: {exc}"
        items = obj if isinstance(obj, list) else [obj]
        for item in items:
            if not isinstance(item, dict):
                return [], "malformed", "malformed tool call: expected object"
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                func = item.get("function")
                if isinstance(func, dict):
                    name = func.get("name")
                if not isinstance(name, str) or not name.strip():
                    return [], "malformed", "malformed tool call: missing name"
            name = str(name).strip()
            if allowed_names is not None and name not in allowed_names:
                return [], "unknown", f"unknown tool {name!r}"
            args = item.get("arguments")
            if args is None and isinstance(item.get("function"), dict):
                args = item["function"].get("arguments")
            if isinstance(args, dict):
                args_str = json.dumps(args, separators=(",", ":"))
            elif isinstance(args, str):
                if args.strip():
                    try:
                        strict_json_loads(args)
                    except (json.JSONDecodeError, ValueError) as exc:
                        return [], "malformed", f"malformed tool call arguments: {exc}"
                args_str = args if args.strip() else "{}"
            elif args is None:
                args_str = "{}"
            else:
                return [], "malformed", "malformed tool call: invalid arguments"
            contract = (tool_parameters or {}).get(name)
            if isinstance(contract, dict):
                parsed_args = strict_json_loads(args_str)
                if not isinstance(parsed_args, dict):
                    return [], "malformed", f"tool {name!r} arguments must be an object"
                required = contract.get("required", [])
                if isinstance(required, list):
                    missing = [key for key in required if isinstance(key, str) and key not in parsed_args]
                    if missing:
                        return [], "malformed", f"tool {name!r} missing required arguments: {', '.join(missing)}"
                properties = contract.get("properties", {})
                if contract.get("additionalProperties") is False and isinstance(properties, dict):
                    unknown = sorted(set(parsed_args) - set(properties))
                    if unknown:
                        return [], "malformed", f"tool {name!r} has unknown arguments: {', '.join(unknown)}"
                if not _tool_value_matches_schema(parsed_args, contract):
                    return [], "malformed", f"tool {name!r} arguments violate the declared schema"
            tool_calls.append(
                {
                    "name": name,
                    "arguments": args_str,
                    "call_id": f"call_{uuid.uuid4().hex[:24]}",
                }
            )
    return tool_calls, None, None


def _strip_tool_json_from_text(
    text: str,
    tool_calls: list[dict[str, Any]] | None,
    *,
    tool_syntax: str = _TOOL_SYNTAX_TOOL_CALL_JSON,
) -> str:
    if not text or not tool_calls:
        return text
    stripped = text.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            json.loads(stripped)
            return ""
        except json.JSONDecodeError:
            pass
    remaining = text
    if tool_syntax == _TOOL_SYNTAX_QWEN_CODE_XML and stripped.startswith("<function_calls"):
        return ""
    if tool_syntax == _TOOL_SYNTAX_QWEN_CODE_XML_RAW_V2:
        start = stripped.find("<function_calls")
        if start >= 0:
            return stripped[:start].strip()
    if tool_syntax in {_TOOL_SYNTAX_TOOL_CALL_JSON, _TOOL_SYNTAX_QWEN_XML, _TOOL_SYNTAX_QWEN_CODE_XML}:
        remaining = re.sub(
            r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
    return remaining.strip()


def _tool_call_limit(parallel: Any, max_calls: Any) -> int | None:
    """Effective cap on emitted tool calls, or None for unlimited."""
    if parallel is False:
        return 1
    if max_calls is None:
        return None
    try:
        n = int(max_calls)
    except (TypeError, ValueError):
        return None
    return n if n >= 1 else None


def _required_tool_name(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, dict):
        func = tool_choice.get("function")
        if isinstance(func, dict):
            name = func.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return None


def _enforce_tool_choice(
    tool_calls: list[dict[str, Any]] | None, *, tool_choice: Any
) -> tuple[str | None, bool]:
    """Enforce the request's tool_choice. Returns (error, drop_calls).

    ``auto`` (or unset/unknown shapes) enforces nothing. ``required`` (or
    a ``{"function": {"name"}}`` dict) fails when no call was parsed.
    ``none`` drops parsed calls so nothing executes; the model text is
    returned as-is so the misbehavior stays visible.
    """
    if tool_choice is None or tool_choice == "auto":
        return None, False
    if tool_choice == "none":
        return (None, True) if tool_calls else (None, False)
    if tool_choice == "required":
        if not tool_calls:
            return "tool_choice 'required' but the model made no tool call", False
        return None, False
    name = _required_tool_name(tool_choice)
    if name is not None:
        if not tool_calls or any(tc.get("name") != name for tc in tool_calls):
            return (
                f"tool_choice requires tool {name!r} but the model did not call it",
                False,
            )
        return None, False
    return None, False


def _strict_tool_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Tighten a tool schema for ``strict: true`` (pure tightening).

    All declared properties become required and unknown arguments are
    rejected. Explicit author settings are preserved where they already
    constrain at least as much.
    """
    tightened = dict(params)
    properties = tightened.get("properties")
    if isinstance(properties, dict):
        required = tightened.get("required")
        required_set = set(required) if isinstance(required, list) else set()
        required_set.update(k for k in properties if isinstance(k, str))
        tightened["required"] = sorted(required_set)
    if tightened.get("additionalProperties") is not False:
        tightened["additionalProperties"] = False
    return tightened


def _classify_stream_mode(
    buffer: str, *, tool_syntax: str = _TOOL_SYNTAX_TOOL_CALL_JSON
) -> str | None:
    """Legacy diagnostic classifier; serving buffers tool-bearing output."""
    stripped = buffer.lstrip()
    if not stripped:
        return None
    if stripped.startswith("{") or stripped.startswith("["):
        return "tool" if tool_syntax == _TOOL_SYNTAX_JSON else "text"
    if stripped.startswith("<"):
        if stripped.lower().startswith("<function_calls"):
            return "tool" if tool_syntax in {
                _TOOL_SYNTAX_QWEN_CODE_XML, _TOOL_SYNTAX_QWEN_CODE_XML_RAW_V2,
            } else "text"
        if stripped.lower().startswith("<tool_call"):
            return "tool" if tool_syntax in {_TOOL_SYNTAX_TOOL_CALL_JSON, _TOOL_SYNTAX_QWEN_XML, _TOOL_SYNTAX_QWEN_CODE_XML} else "text"
        return "text"
    return "text"


class _FlightLease:
    """One request's execution ownership, independent of HTTP consumption."""
    def __init__(self, lock, session):
        self.lock = lock
        self.session = session
        self.guard = threading.Lock()
        self.active = True
        self.worker_started = False
        self.cancelled = None

    def start_worker(self, cancelled):
        with self.guard:
            if not self.active:
                return False
            self.worker_started = True
            self.cancelled = cancelled
            return True

    def release(self):
        with self.guard:
            if self.active:
                self.active = False
                self.lock.release()

    def cancel(self):
        with self.guard:
            if not self.active:
                return
            if self.cancelled is not None:
                self.cancelled.set()
            if self.worker_started:
                self.session.cancel()

    def disconnect(self):
        with self.guard:
            if not self.active:
                return
            if self.worker_started:
                self.cancelled.set()
                self.session.cancel()
                # Native completion, not a timeout or disconnected consumer,
                # releases state still in use.
            else:
                self.active = False
                self.lock.release()


class _OwnedStreamingResponse(StreamingResponse):
    def __init__(self, *args, lease, **kwargs):
        super().__init__(*args, **kwargs)
        self.lease = lease

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.lease.disconnect()


# --- app factory --------------------------------------------------------------


def create_app(
    session,
    *,
    model: str = "ck-v8",
    context_length: int | None = None,
    stats: bool = True,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 512,
    stop_on_text: Sequence[str] = (),
    stop_at_eos: bool = False,
    flags: int = 0,
    chat_contract: dict[str, Any] | None = None,
    chat_template: str | None = None,
    chat_templates: dict[str, str] | None = None,
    tool_protocol: str | None = None,
    loaded_identity: dict[str, Any] | None = None,
    renderer_tokens: dict[str, str] | None = None,
    allow_untemplated: bool = False,
    cancel_wait_seconds: float = 10.0,
    viz_html: str | None = None,
    extra_route_registrar: Callable[[APIRouter, Callable[..., Any]], None] | None = None,
):
    """Build the live Responses FastAPI app around a session (real or fake).

    ``viz_html`` serves a ``GET /viz`` page when provided (scripts-side file
    loading stays in the caller). ``extra_route_registrar`` is called with
    ``(router, create_response)`` before the app is built so hosts can attach
    compatibility routes (e.g. Chat Completions) against the canonical handler.
    """

    if not isinstance(chat_template, str) or not chat_template.strip():
        if not allow_untemplated:
            raise ValueError(
                "normal chat serving requires a nonempty native Jinja template; "
                "enable untemplated raw serving explicitly"
            )

    router = APIRouter()
    if loaded_identity is not None and loaded_identity.get("schema") != "cke.loaded_serving_identity.v1":
        raise ValueError("invalid loaded serving identity")
    if loaded_identity is not None and (
        not isinstance(context_length, int) or isinstance(context_length, bool)
        or context_length <= 0 or not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool) or max_tokens <= 0
    ):
        raise ValueError("loaded serving identity requires positive effective context and output limits")
    attestation = copy.deepcopy(loaded_identity)
    server_instance_id = uuid.uuid4().hex
    response_store: OrderedDict[str, dict[str, Any]] = OrderedDict()
    response_history_store: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    response_store_lock = threading.Lock()
    response_store_limit = 256
    _flight_lock = threading.Lock()
    active_streams: dict[str, dict[str, Any]] = {}
    active_streams_lock = threading.Lock()

    stop_markers = [str(m) for m in (stop_on_text or ()) if str(m)]
    all_stop_markers = list(stop_markers)
    if stop_at_eos:
        all_stop_markers.append("<eos>")

    # Tool-call wire format is declared by the loaded chat template, not by
    # hardcoded tags or template-text heuristics.
    declared_tool_protocol = tool_protocol or (chat_contract or {}).get("tool_protocol")
    tool_syntax = _tool_syntax_for_protocol(declared_tool_protocol)
    effective_serving = {
        "schema": "cke.effective_serving.v1",
        "configured_mode": "raw" if allow_untemplated and not chat_template else "templated",
        "output_protocol": declared_tool_protocol if chat_template else None,
        "active_context_limit": context_length,
        "default_max_output_tokens": max_tokens,
        "stop_on_text": stop_markers,
        "stop_at_eos": stop_at_eos,
    }

    def _conversation_echo(body) -> dict[str, Any] | None:
        conv = body.conversation
        if isinstance(conv, str):
            return {"id": conv}
        if conv is not None:
            return {"id": conv.id}
        return None

    def _validate_request(body) -> None:
        _reject_live_media(body.input)
        if body.model != model:
            raise HTTPException(
                status_code=404,
                detail=f"Model {body.model!r} is not loaded; available model: {model!r}",
            )
        unsupported_types = sorted({
            str(getattr(tool, "type", "unknown"))
            for tool in (body.tools or [])
            if getattr(tool, "type", None) not in _FUNCTION_LIKE_TYPES
        })
        if unsupported_types:
            raise _harness_error(
                501, f"unsupported tool types: {', '.join(unsupported_types)}",
                err_type="invalid_request_error", code="unsupported_tool_type",
            )
        for tool in _effective_tools(body):
            if getattr(tool, "name", None):
                try:
                    _validate_tool_schema_subset(tool.parameters, f"tool {tool.name!r} parameters")
                except ValueError as exc:
                    raise _harness_error(
                        400, str(exc), err_type="invalid_request_error",
                        code="unsupported_tool_schema",
                    ) from exc
        if (
            body.tools
            and _effective_tools(body)
            and not _has_tool_support(
                chat_template, chat_templates, tool_protocol=declared_tool_protocol
            )
        ):
            raise _harness_error(
                501, "tool protocol is undeclared for the selected chat template",
                err_type="invalid_request_error", code="tool_protocol_undeclared",
            )
        max_calls = getattr(body, "max_tool_calls", None)
        if max_calls is not None:
            try:
                valid_max = int(max_calls) >= 1
            except (TypeError, ValueError):
                valid_max = False
            if not valid_max:
                raise _harness_error(
                    400, "max_tool_calls must be a positive integer",
                    err_type="invalid_request_error", code="invalid_request",
                )

    def _store_response(
        response_id: str, response: dict[str, Any], history: list[dict[str, Any]]
    ) -> None:
        with response_store_lock:
            response_store[response_id] = response
            response_history_store[response_id] = history
            response_store.move_to_end(response_id)
            response_history_store.move_to_end(response_id)
            while len(response_store) > response_store_limit:
                evicted_id, _ = response_store.popitem(last=False)
                response_history_store.pop(evicted_id, None)

    def _request_messages(body) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if body.previous_response_id:
            with response_store_lock:
                previous = response_history_store.get(body.previous_response_id)
            if previous is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Previous response {body.previous_response_id!r} not found; "
                    "responses created with store:false are ephemeral and cannot be chained",
                )
            messages.extend(dict(m) for m in previous)
        try:
            current = _input_chat_messages(body.input)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        known_call_ids = {
            call.get("id")
            for message in [*messages, *current]
            for call in message.get("tool_calls", [])
            if isinstance(call, dict) and call.get("id")
        }
        for message in current:
            if message.get("role") != "tool":
                continue
            if message.get("tool_call_id") not in known_call_ids:
                raise _harness_error(
                    400,
                    f"Function output references unknown call_id {message.get('tool_call_id')!r}",
                    err_type="invalid_request_error",
                    code="unknown_call_id",
                )
        messages.extend(current)
        return messages

    def _prepare_request(body):
        _validate_request(body)
        messages = _request_messages(body)
        prompt = _extract_prompt(body)
        tok_limit = (
            body.max_output_tokens if body.max_output_tokens is not None else max_tokens
        )
        temperature_eff = (
            body.temperature if body.temperature is not None else temperature
        )
        top_p_eff = body.top_p if body.top_p is not None else top_p
        effective_flags = flags
        effective_thinking = _resolve_thinking_mode(body)
        jinja_rendered: str | None = None
        generation_prefix = ""
        if chat_template is not None or chat_templates is not None:
            try:
                render_time = datetime.now()
                jinja_rendered = _render_with_chat_templates(
                    chat_template,
                    chat_templates,
                    messages,
                    body,
                    chat_contract,
                    effective_thinking,
                    renderer_tokens=renderer_tokens,
                    render_time=render_time,
                )
                without_prefix = _render_with_chat_templates(
                    chat_template, chat_templates, messages, body, chat_contract,
                    effective_thinking, add_generation_prompt=False,
                    renderer_tokens=renderer_tokens,
                    render_time=render_time,
                )
                if jinja_rendered is not None and without_prefix is not None:
                    if not jinja_rendered.startswith(without_prefix):
                        raise TemplateRenderError("cannot isolate selected template generation prefix")
                    generation_prefix = jinja_rendered[len(without_prefix):]
            except TemplateRenderError as exc:
                raise _harness_error(
                    422, str(exc), err_type="invalid_request_error",
                    code="template_render_failed",
                ) from exc
            if jinja_rendered is not None and not jinja_rendered.strip():
                raise _harness_error(
                    422, "selected chat template rendered an empty prompt",
                    err_type="invalid_request_error", code="template_render_failed",
                )
        requires_role_rendering = any(
            m.get("role") != "user" or m.get("tool_calls") for m in messages
        )
        if requires_role_rendering and jinja_rendered is None:
            raise HTTPException(
                status_code=422,
                detail="The selected model template cannot render role-aware tool history",
            )
        if jinja_rendered is not None and jinja_rendered.strip():
            prompt = jinja_rendered
            effective_flags |= CK_SESSION_REQUEST_RAW_PROMPT
        elif chat_contract is not None:
            prompt = _format_prompt_with_chat_contract(
                prompt,
                chat_contract,
                thinking_mode=effective_thinking,
                system_prompt=body.instructions,
            )
            effective_flags |= CK_SESSION_REQUEST_RAW_PROMPT
        elif isinstance(body.instructions, str):
            prompt = f"{body.instructions}\n{prompt}".strip()
        if not prompt or not prompt.strip():
            prompt = _extract_prompt(body) or "Hello"
            if isinstance(body.instructions, str) and body.instructions.strip():
                prompt = f"{body.instructions}\n{prompt}".strip()
            if not prompt.strip():
                prompt = "Hello"
        if context_length is not None:
            if tok_limit >= context_length:
                raise _harness_error(
                    400,
                    f"max_output_tokens {tok_limit} leaves no room in the loaded "
                    f"context capacity {context_length}; request a smaller output "
                    "budget or load a larger generated runtime",
                    err_type="invalid_request_error",
                    code="context_length_exceeded",
                )
            count_tokens = getattr(session, "count_tokens", None)
            if callable(count_tokens):
                prompt_tokens = count_tokens(prompt)
                if prompt_tokens + tok_limit > context_length:
                    available = max(0, context_length - prompt_tokens)
                    raise _harness_error(
                        400,
                        f"rendered prompt has {prompt_tokens} tokens and the request "
                        f"reserves {tok_limit} output tokens, exceeding loaded context "
                        f"capacity {context_length}; at most {available} output tokens remain",
                        err_type="invalid_request_error",
                        code="context_length_exceeded",
                    )
        return prompt, tok_limit, temperature_eff, top_p_eff, effective_flags, generation_prefix

    def build_response(
        body,
        *,
        response_id: str,
        message_id: str,
        created_at: int,
        status,
        text: str,
        input_tokens: int,
        output_tokens: int,
        thinking: str | None = None,
        reasoning_tokens: int = 0,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        incomplete_details: dict[str, Any] | None = None,
        completed_at: int | None = None,
        item_status: str = "completed",
        reasoning_item_id: str | None = None,
        include_empty_reasoning: bool = False,
        tool_calls: list[dict[str, Any]] | None = None,
    ):
        item_status_value = (
            "in_progress" if status == ResponseStatus.in_progress else item_status
        )
        output: list[dict[str, Any]] = []
        if status != ResponseStatus.in_progress:
            if thinking is not None or include_empty_reasoning:
                output.append(
                    ReasoningItem(
                        id=reasoning_item_id or f"rsn_{uuid.uuid4().hex[:24]}",
                        status=item_status_value,
                        content=(
                            [ReasoningTextContent(text=thinking)]
                            if thinking is not None
                            else []
                        ),
                        summary=[],
                    ).model_dump()
                )
            if tool_calls:
                for tc in tool_calls:
                    output.append(
                        FunctionCall(
                            id=tc.get("id") or f"fc_{uuid.uuid4().hex[:24]}",
                            call_id=tc.get("call_id")
                            or f"call_{uuid.uuid4().hex[:24]}",
                            name=tc.get("name") or "",
                            arguments=tc.get("arguments") or "{}",
                            status=item_status_value,  # type: ignore
                        ).model_dump()
                    )
                if text and text.strip():
                    output.append(
                        ResponseOutputMessage(
                            id=message_id,
                            content=[ResponseOutputText(text=text)],
                            role="assistant",
                            status=item_status_value,
                        ).model_dump()
                    )
            else:
                output.append(
                    ResponseOutputMessage(
                        id=message_id,
                        content=[ResponseOutputText(text=text)],
                        role="assistant",
                        status=item_status_value,
                    ).model_dump()
                )
        resp: dict[str, Any] = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "completed_at": completed_at,
            "status": status,
            "error": error,
            "incomplete_details": incomplete_details,
            "instructions": body.instructions,
            "metadata": body.metadata or {},
            "model": body.model or model,
            "output": output,
            "output_text": text,
            "parallel_tool_calls": body.parallel_tool_calls
            if body.parallel_tool_calls is not None
            else True,
            "temperature": body.temperature
            if body.temperature is not None
            else temperature,
            "top_p": body.top_p if body.top_p is not None else top_p,
            "top_logprobs": body.top_logprobs,
            "tool_choice": body.tool_choice,
            "tools": [t.model_dump() for t in body.tools] if body.tools else [],
            "truncation": body.truncation,
            "text": body.text.model_dump() if body.text is not None else None,
            "user": body.user,
            "background": body.background,
            "conversation": _conversation_echo(body),
            "max_output_tokens": body.max_output_tokens
            if body.max_output_tokens is not None
            else max_tokens,
            "max_tool_calls": body.max_tool_calls,
            "moderation": body.moderation.model_dump()
            if body.moderation is not None
            else None,
            "previous_response_id": body.previous_response_id,
            "prompt": body.prompt.model_dump() if body.prompt is not None else None,
            "prompt_cache_key": body.prompt_cache_key,
            "prompt_cache_options": (
                body.prompt_cache_options.model_dump()
                if body.prompt_cache_options is not None
                else None
            ),
            "prompt_cache_retention": body.prompt_cache_retention,
            "reasoning": body.reasoning.model_dump()
            if body.reasoning is not None
            else None,
            "safety_identifier": body.safety_identifier,
            "service_tier": body.service_tier,
            "usage": _usage(input_tokens, output_tokens, reasoning_tokens),
        }
        if result is not None:
            resp["performance"] = _performance_profile(result)
        should_store = getattr(body, "store", None) is not False
        if should_store:
            history = _request_messages(body)
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": text,
                # Always present: the next turn renders this history entry
                # through the native template, which reads
                # `message.tool_calls` unguarded (StrictUndefined raises on
                # a missing key).
                "tool_calls": [],
            }
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": call.get("name"),
                            "arguments": json.loads(call.get("arguments") or "{}"),
                        },
                    }
                    for call in tool_calls
                ]
            _store_response(response_id, resp, [*history, assistant])
        return resp

    def _parse_tool_result(
        text: str, body
    ) -> tuple[list[dict[str, Any]] | None, str | None, str | None]:
        if not _has_function_tools(body):
            return None, None, None
        allowed = {t.name for t in _effective_tools(body) if getattr(t, "name", None)}
        parameter_schemas = {}
        for t in _effective_tools(body):
            name = getattr(t, "name", None)
            params = getattr(t, "parameters", None)
            if not name or not isinstance(params, dict):
                continue
            if getattr(t, "strict", False):
                params = _strict_tool_parameters(params)
            parameter_schemas[name] = params
        tool_calls, code, msg = _extract_tool_calls_from_text(
            text, allowed, tool_syntax=tool_syntax, tool_parameters=parameter_schemas
        )
        if (code is None and tool_calls
                and getattr(body, "parallel_tool_calls", True) is False
                and len(tool_calls) > 1):
            return [], "parallel_tool_calls_disallowed", (
                "parallel_tool_calls=false forbids multiple tool calls in one response"
            )
        cap = _tool_call_limit(getattr(body, "parallel_tool_calls", True),
                               getattr(body, "max_tool_calls", None))
        if code is None and tool_calls and cap is not None and len(tool_calls) > cap:
            return [], "max_tool_calls_exceeded", "generated response exceeds max_tool_calls; no calls emitted"
        if code is None and not tool_calls:
            return None, None, None
        if tool_calls == [] and code is None:
            return None, None, None
        return tool_calls, code, msg

    def stream_events(
        body,
        prompt,
        *,
        max_tokens,
        temperature,
        top_p,
        effective_flags=None,
        request=None,
        generation_prefix="",
        lease=None,
    ):
        think_enabled = _resolve_thinking_mode(body) == "visible"
        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        reasoning_item_id = f"rsn_{uuid.uuid4().hex[:24]}" if think_enabled else None
        created_at = int(time.time())
        pending = build_response(
            body,
            response_id=response_id,
            message_id=message_id,
            created_at=created_at,
            status=ResponseStatus.in_progress,
            text="",
            input_tokens=0,
            output_tokens=0,
        )
        seq = 0
        yield _sse(
            "response.created",
            {"type": "response.created", "response": pending, "sequence_number": seq},
        )
        seq += 1
        yield _sse(
            "response.in_progress",
            {
                "type": "response.in_progress",
                "response": pending,
                "sequence_number": seq,
            },
        )
        seq += 1

        complete: list[str] = []
        events: queue.Queue = queue.Queue()
        cancelled = threading.Event()
        worker_finished = threading.Event()
        splitter = (
            _StreamThinkSplitter(start_thinking=_prompt_opens_thinking(generation_prefix))
            if think_enabled
            else None
        )
        # Tool-bearing output is buffered until the complete protocol
        # envelope can be parsed and validated.

        def on_token(_tid, text):
            if text:
                complete.append(text)
                if splitter is not None:
                    for state, delta in splitter.feed(text):
                        if state == "thinking":
                            events.put(("reasoning_text", delta))
                        elif not _has_function_tools(body):
                            events.put(("text", delta))
                    # Tool-bearing answer text stays buffered until the
                    # full response is parsed and validated at terminal.
                elif not _has_function_tools(body):
                    events.put(("text", text))
                # A tool-bearing response can switch from prose to a tool
                # envelope at any token boundary. Buffer it until the full
                # response is parsed, then emit only validated content.
            return -1 if cancelled.is_set() else 0

        def worker():
            try:
                result = session.generate(
                    None,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    on_token=on_token,
                    flags=effective_flags if effective_flags is not None else flags,
                    stop_on_text=stop_markers,
                    stop_at_eos=stop_at_eos,
                )
                if splitter is not None:
                    for state, delta in splitter.flush():
                        if state == "thinking":
                            events.put(("reasoning_text", delta))
                        elif not _has_function_tools(body):
                            events.put(("text", delta))
                terminal_event = ("done", result)
            except SessionBusyError:
                terminal_event = (
                    "busy",
                    "Session busy: another request is in progress.",
                )
            except Exception as e:
                terminal_event = ("error", str(e))
            finally:
                with active_streams_lock:
                    active_streams.pop(response_id, None)
                lease.release()
                worker_finished.set()
            events.put(terminal_event)

        worker_thread = threading.Thread(target=worker, daemon=True)
        with active_streams_lock:
            active_streams[response_id] = {
                "cancelled": cancelled,
                "finished": worker_finished,
                "thread": worker_thread,
                "lease": lease,
            }
        if not lease.start_worker(cancelled):
            with active_streams_lock:
                active_streams.pop(response_id, None)
            worker_finished.set()
            return
        try:
            worker_thread.start()
        except BaseException:
            with active_streams_lock:
                active_streams.pop(response_id, None)
            lease.release()
            raise

        reasoning_started = False
        message_started = False
        message_content_part_added = False

        def emit(kind, data):
            nonlocal seq
            data["sequence_number"] = seq
            seq += 1
            yield _sse(kind, data)

        def emit_reasoning_lifecycle(thinking, is_cancelled):
            nonlocal reasoning_started
            status_value = "incomplete" if is_cancelled else "completed"
            if reasoning_started:
                if thinking is not None:
                    yield from emit(
                        "response.reasoning_text.done",
                        {
                            "type": "response.reasoning_text.done",
                            "item_id": reasoning_item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "text": thinking,
                        },
                    )
                yield from emit(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": ReasoningItem(
                            id=reasoning_item_id,
                            status=status_value,
                            content=(
                                [ReasoningTextContent(text=thinking)]
                                if thinking is not None
                                else []
                            ),
                            summary=[],
                        ).model_dump(),
                    },
                )
            else:
                yield from emit(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": ReasoningItem(
                            id=reasoning_item_id,
                            status="in_progress",
                            content=[],
                            summary=[],
                        ).model_dump(),
                    },
                )
                reasoning_started = True
                yield from emit(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": ReasoningItem(
                            id=reasoning_item_id,
                            status=status_value,
                            content=[],
                            summary=[],
                        ).model_dump(),
                    },
                )

        try:
            last_heartbeat = time.monotonic()
            while True:
                try:
                    kind, payload = events.get(timeout=0.2)
                except queue.Empty:
                    if _has_function_tools(body) and time.monotonic() - last_heartbeat >= 15:
                        yield ": keep-alive\n\n"
                        last_heartbeat = time.monotonic()
                    if cancelled.is_set() and worker_finished.is_set():
                        continue
                    if worker_finished.is_set():
                        continue
                    continue
                if kind == "reasoning_text":
                    if not reasoning_started:
                        yield from emit(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": 0,
                                "item": ReasoningItem(
                                    id=reasoning_item_id,
                                    status="in_progress",
                                    content=[],
                                    summary=[],
                                ).model_dump(),
                            },
                        )
                        reasoning_started = True
                    yield from emit(
                        "response.reasoning_text.delta",
                        {
                            "type": "response.reasoning_text.delta",
                            "item_id": reasoning_item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": payload,
                        },
                    )
                elif kind == "text":
                    msg_idx = 1 if think_enabled else 0
                    if not message_started:
                        yield from emit(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": msg_idx,
                                "item": ResponseOutputMessage(
                                    id=message_id,
                                    content=[],
                                    role="assistant",
                                    status="in_progress",
                                ).model_dump(),
                            },
                        )
                        message_started = True
                        yield from emit(
                            "response.content_part.added",
                            {
                                "type": "response.content_part.added",
                                "item_id": message_id,
                                "output_index": msg_idx,
                                "content_index": 0,
                                "part": {
                                    "type": "output_text",
                                    "text": "",
                                    "annotations": [],
                                },
                            },
                        )
                        message_content_part_added = True
                    yield from emit(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": message_id,
                            "output_index": msg_idx,
                            "content_index": 0,
                            "delta": payload,
                            "logprobs": None,
                        },
                    )
                elif kind == "done":
                    result = payload or {}
                    stop_reason_val = int(result.get("stop_reason") or 0)
                    is_cancelled = cancelled.is_set() or stop_reason_val == 3
                    text = truncate_stop_markers("".join(complete), all_stop_markers)
                    thinking = None
                    if think_enabled:
                        thinking, text = _split_generated_thinking(text, generation_prefix)
                        thinking = thinking or None
                    input_tokens = int(result.get("prompt_tokens") or 0)
                    output_tokens = int(result.get("generated_tokens") or len(complete))
                    message_index = 1 if think_enabled else 0
                    tool_calls, tool_error_code, tool_error_msg = _parse_tool_result(
                        text, body
                    )
                    if tool_error_code is None:
                        choice_error, drop = _enforce_tool_choice(
                            tool_calls,
                            tool_choice=getattr(body, "tool_choice", None),
                        )
                        if choice_error is not None:
                            tool_error_code, tool_error_msg = "choice", choice_error
                        elif drop:
                            tool_calls = None
                    if tool_error_code is not None and not is_cancelled:
                        if think_enabled:
                            yield from emit_reasoning_lifecycle(thinking, is_cancelled)
                        final = build_response(
                            body,
                            response_id=response_id,
                            message_id=message_id,
                            created_at=created_at,
                            completed_at=int(time.time()),
                            status=ResponseStatus.failed,
                            text=text,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            thinking=thinking,
                            reasoning_tokens=max(0, len(thinking or "") // 4)
                            if thinking
                            else 0,
                            result=result,
                            error={
                                "code": "server_error",
                                "message": tool_error_msg or "tool call failed",
                            },
                            reasoning_item_id=reasoning_item_id,
                            include_empty_reasoning=think_enabled,
                            item_status="incomplete",
                            tool_calls=None,
                        )
                        yield from emit(
                            "error",
                            {
                                "type": "error",
                                "code": "server_error",
                                "message": tool_error_msg or "tool call failed",
                                "param": None,
                            },
                        )
                        yield from emit(
                            "response.failed",
                            {
                                "type": "response.failed",
                                "response": final,
                                "error": {
                                    "code": "server_error",
                                    "message": tool_error_msg or "tool call failed",
                                },
                            },
                        )
                        return
                    incomplete_details = None
                    remaining_text = (
                        _strip_tool_json_from_text(text, tool_calls, tool_syntax=tool_syntax)
                        if tool_calls
                        else text
                    )
                    if is_cancelled:
                        final_status = ResponseStatus.cancelled
                        msg_status = "incomplete"
                    elif stop_reason_val == 2:
                        final_status = ResponseStatus.incomplete
                        msg_status = "incomplete"
                        incomplete_details = {"reason": "max_output_tokens"}
                    else:
                        final_status = ResponseStatus.completed
                        msg_status = "completed"
                    if think_enabled:
                        yield from emit_reasoning_lifecycle(thinking, is_cancelled)
                    if tool_calls:
                        base_idx = message_index
                        for idx, tc in enumerate(tool_calls):
                            out_idx = base_idx + idx
                            func_id = tc.get("id") or f"fc_{uuid.uuid4().hex[:24]}"
                            tc["id"] = func_id
                            yield from emit(
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "output_index": out_idx,
                                    "item": FunctionCall(
                                        id=func_id,
                                        call_id=tc.get("call_id")
                                        or f"call_{uuid.uuid4().hex[:24]}",
                                        name=tc["name"],
                                        arguments="",
                                        status="in_progress",  # type: ignore
                                    ).model_dump(),
                                },
                            )
                            args_str = tc.get("arguments") or "{}"
                            if args_str:
                                yield from emit(
                                    "response.function_call_arguments.delta",
                                    {
                                        "type": "response.function_call_arguments.delta",
                                        "item_id": func_id,
                                        "output_index": out_idx,
                                        "delta": args_str,
                                    },
                                )
                            yield from emit(
                                "response.function_call_arguments.done",
                                {
                                    "type": "response.function_call_arguments.done",
                                    "item_id": func_id,
                                    "output_index": out_idx,
                                    "arguments": args_str,
                                },
                            )
                            yield from emit(
                                "response.output_item.done",
                                {
                                    "type": "response.output_item.done",
                                    "output_index": out_idx,
                                    "item": FunctionCall(
                                        id=func_id,
                                        call_id=tc.get("call_id") or func_id,
                                        name=tc["name"],
                                        arguments=args_str,
                                        status=msg_status,  # type: ignore
                                    ).model_dump(),
                                },
                            )
                        if remaining_text and remaining_text.strip():
                            msg_out_idx = base_idx + len(tool_calls)
                            yield from emit(
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "output_index": msg_out_idx,
                                    "item": ResponseOutputMessage(
                                        id=message_id,
                                        content=[],
                                        role="assistant",
                                        status="in_progress",
                                    ).model_dump(),
                                },
                            )
                            message_started = True
                            yield from emit(
                                "response.content_part.added",
                                {
                                    "type": "response.content_part.added",
                                    "item_id": message_id,
                                    "output_index": msg_out_idx,
                                    "content_index": 0,
                                    "part": {
                                        "type": "output_text",
                                        "text": "",
                                        "annotations": [],
                                    },
                                },
                            )
                            yield from emit(
                                "response.output_text.delta",
                                {
                                    "type": "response.output_text.delta",
                                    "item_id": message_id,
                                    "output_index": msg_out_idx,
                                    "content_index": 0,
                                    "delta": remaining_text,
                                    "logprobs": None,
                                },
                            )
                            yield from emit(
                                "response.content_part.done",
                                {
                                    "type": "response.content_part.done",
                                    "item_id": message_id,
                                    "output_index": msg_out_idx,
                                    "content_index": 0,
                                    "part": {
                                        "type": "output_text",
                                        "text": remaining_text,
                                        "annotations": [],
                                    },
                                },
                            )
                            yield from emit(
                                "response.output_text.done",
                                {
                                    "type": "response.output_text.done",
                                    "item_id": message_id,
                                    "output_index": msg_out_idx,
                                    "content_index": 0,
                                    "text": remaining_text,
                                },
                            )
                            yield from emit(
                                "response.output_item.done",
                                {
                                    "type": "response.output_item.done",
                                    "output_index": msg_out_idx,
                                    "item": ResponseOutputMessage(
                                        id=message_id,
                                        content=[
                                            ResponseOutputText(text=remaining_text)
                                        ],
                                        role="assistant",
                                        status=msg_status,
                                    ).model_dump(),
                                },
                            )
                    else:
                        if not message_started:
                            yield from emit(
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "output_index": message_index,
                                    "item": ResponseOutputMessage(
                                        id=message_id,
                                        content=[],
                                        role="assistant",
                                        status="in_progress",
                                    ).model_dump(),
                                },
                            )
                            message_started = True
                            yield from emit(
                                "response.content_part.added",
                                {
                                    "type": "response.content_part.added",
                                    "item_id": message_id,
                                    "output_index": message_index,
                                    "content_index": 0,
                                    "part": {
                                        "type": "output_text",
                                        "text": "",
                                        "annotations": [],
                                    },
                                },
                            )
                            message_content_part_added = True
                        # Tool-bearing responses buffer model text until the
                        # whole response is validated.
                        if _has_function_tools(body) and remaining_text:
                            yield from emit(
                                "response.output_text.delta",
                                {
                                    "type": "response.output_text.delta",
                                    "item_id": message_id,
                                    "output_index": message_index,
                                    "content_index": 0,
                                    "delta": remaining_text,
                                    "logprobs": None,
                                },
                            )
                        if message_content_part_added:
                            yield from emit(
                                "response.content_part.done",
                                {
                                    "type": "response.content_part.done",
                                    "item_id": message_id,
                                    "output_index": message_index,
                                    "content_index": 0,
                                    "part": {
                                        "type": "output_text",
                                        "text": remaining_text,
                                        "annotations": [],
                                    },
                                },
                            )
                        yield from emit(
                            "response.output_text.done",
                            {
                                "type": "response.output_text.done",
                                "item_id": message_id,
                                "output_index": message_index,
                                "content_index": 0,
                                "text": remaining_text,
                            },
                        )
                        yield from emit(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": message_index,
                                "item": ResponseOutputMessage(
                                    id=message_id,
                                    content=[ResponseOutputText(text=remaining_text)],
                                    role="assistant",
                                    status=msg_status,
                                ).model_dump(),
                            },
                        )
                    reasoning_tokens = (
                        max(0, len(thinking) // 4) if thinking is not None else 0
                    )
                    final = build_response(
                        body,
                        response_id=response_id,
                        message_id=message_id,
                        created_at=created_at,
                        completed_at=int(time.time()),
                        status=final_status,
                        text=remaining_text,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        thinking=thinking,
                        reasoning_tokens=reasoning_tokens,
                        result=result,
                        incomplete_details=incomplete_details,
                        reasoning_item_id=reasoning_item_id,
                        include_empty_reasoning=think_enabled,
                        item_status=msg_status,
                        tool_calls=tool_calls,
                    )
                    if stats:
                        _log_performance(model, final.get("performance"))
                    if is_cancelled:
                        yield from emit(
                            "response.cancelled",
                            {"type": "response.cancelled", "response": final},
                        )
                    elif final_status == ResponseStatus.incomplete:
                        yield from emit(
                            "response.incomplete",
                            {"type": "response.incomplete", "response": final},
                        )
                    else:
                        yield from emit(
                            "response.completed",
                            {"type": "response.completed", "response": final},
                        )
                    return
                elif kind == "busy":
                    final = build_response(
                        body,
                        response_id=response_id,
                        message_id=message_id,
                        created_at=created_at,
                        completed_at=int(time.time()),
                        status=ResponseStatus.failed,
                        text="".join(complete),
                        input_tokens=0,
                        output_tokens=0,
                        error={"code": "session_busy", "message": str(payload)},
                    )
                    yield from emit(
                        "error",
                        {
                            "type": "error",
                            "code": "rate_limit_exceeded",
                            "message": str(payload),
                            "param": None,
                        },
                    )
                    yield from emit(
                        "response.failed",
                        {
                            "type": "response.failed",
                            "response": final,
                            "error": {"code": "session_busy", "message": str(payload)},
                        },
                    )
                    return
                elif kind == "error":
                    final = build_response(
                        body,
                        response_id=response_id,
                        message_id=message_id,
                        created_at=created_at,
                        completed_at=int(time.time()),
                        status=ResponseStatus.failed,
                        text="".join(complete),
                        input_tokens=0,
                        output_tokens=len(complete),
                        error={"code": "server_error", "message": str(payload)},
                    )
                    yield from emit(
                        "error",
                        {
                            "type": "error",
                            "code": "server_error",
                            "message": str(payload),
                            "param": None,
                        },
                    )
                    yield from emit(
                        "response.failed",
                        {
                            "type": "response.failed",
                            "response": final,
                            "error": {"code": "server_error", "message": str(payload)},
                        },
                    )
                    return
        finally:
            if not worker_finished.is_set():
                lease.cancel()


    def _acquire_flight_or_429(timeout: float | None = None) -> None:
        """Take the single-flight lock, waiting briefly for harness bursts.

        Concurrent harness requests (e.g. title + main, client retries) wait
        up to ``_FLIGHT_WAIT_SECONDS`` instead of failing instantly. A healthy
        long generation may still answer 429 after that wait; retries are not
        a substitute for a bounded scheduler or confirmed worker completion.
        """
        if timeout is None:
            timeout = _FLIGHT_WAIT_SECONDS
        deadline = time.monotonic() + timeout
        while True:
            if _flight_lock.acquire(blocking=False):
                return
            if time.monotonic() >= deadline:
                raise _harness_error(
                    429,
                    "Session busy: another request is in progress. Retry later.",
                    err_type="rate_limit_error",
                    code="rate_limit_exceeded",
                    retry_after=timeout,
                )
            time.sleep(0.05)

    @router.post("/responses", response_model=None)
    def create_response(body: CreateResponseRequest, request: Request):
        acquired = False
        lease = None
        try:
            _validate_request(body)
            _acquire_flight_or_429()
            acquired = True
            lease = _FlightLease(_flight_lock, session)
            prompt, tok_limit, temperature_eff, top_p_eff, effective_flags, generation_prefix = (
                _prepare_request(body)
            )
        except HTTPException as exc:
            _log_rejection(model, body, exc)
            if acquired:
                lease.release()
            raise
        except Exception:
            if acquired:
                lease.release()
            raise
        if body.stream:
            return _OwnedStreamingResponse(
                stream_events(
                    body,
                    prompt,
                    max_tokens=tok_limit,
                    temperature=temperature_eff,
                    top_p=top_p_eff,
                    effective_flags=effective_flags,
                    request=request,
                    generation_prefix=generation_prefix,
                    lease=lease,
                ),
                lease=lease,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        think_enabled = _resolve_thinking_mode(body) == "visible"
        chunks: list[str] = []
        result: dict[str, Any] | None = None
        non_stream_reasoning_id = (
            f"rsn_{uuid.uuid4().hex[:24]}" if think_enabled else None
        )
        disconnect_cancelled = threading.Event()
        lease.start_worker(disconnect_cancelled)
        monitor_stop = threading.Event()

        async def _event_loop_token():
            return anyio.lowlevel.current_token()

        try:
            loop_token = anyio.from_thread.run(_event_loop_token)
        except RuntimeError:
            loop_token = None

        def _watch_disconnect():
            while not monitor_stop.wait(0.1):
                try:
                    disconnected = anyio.from_thread.run(
                        request.is_disconnected, token=loop_token
                    )
                except Exception:
                    return
                if disconnected and not monitor_stop.is_set():
                    try:
                        lease.cancel()
                    except Exception:
                        pass
                    return

        monitor_thread = None
        if loop_token is not None:
            monitor_thread = threading.Thread(
                target=_watch_disconnect, name="cke-request-disconnect", daemon=True
            )
            monitor_thread.start()
        try:
            try:

                def _collect(_tid, text):
                    if text:
                        chunks.append(text)
                    return -1 if disconnect_cancelled.is_set() else 0

                result = session.generate(
                    None,
                    prompt,
                    max_tokens=tok_limit,
                    temperature=temperature_eff,
                    top_p=top_p_eff,
                    on_token=_collect,
                    flags=effective_flags,
                    stop_on_text=stop_markers,
                    stop_at_eos=stop_at_eos,
                )
            except SessionBusyError:
                busy_exc = _harness_error(
                    429,
                    "Session busy: another request is in progress. Retry later.",
                    err_type="rate_limit_error",
                    code="rate_limit_exceeded",
                    retry_after=_FLIGHT_WAIT_SECONDS,
                )
                _log_rejection(model, body, busy_exc)
                raise busy_exc
            except Exception as e:
                err_text = truncate_stop_markers("".join(chunks), all_stop_markers)
                err_thinking: str | None = None
                err_reasoning_tokens = 0
                if think_enabled:
                    err_thinking, err_text = split_thinking(err_text)
                    err_thinking = err_thinking or None
                    if err_thinking is not None:
                        err_reasoning_tokens = max(0, len(err_thinking) // 4)
                err_input = int(result.get("prompt_tokens") or 0) if result else 0
                err_output = (
                    int(result.get("generated_tokens") or len(chunks))
                    if result
                    else len(chunks)
                )
                err_resp = build_response(
                    body,
                    response_id=f"resp_{uuid.uuid4().hex[:24]}",
                    message_id=f"msg_{uuid.uuid4().hex[:24]}",
                    created_at=int(time.time()),
                    completed_at=int(time.time()),
                    status=ResponseStatus.failed,
                    text=err_text,
                    input_tokens=err_input,
                    output_tokens=err_output,
                    thinking=err_thinking,
                    reasoning_tokens=err_reasoning_tokens,
                    result=result,
                    error={"code": "server_error", "message": str(e)},
                    reasoning_item_id=non_stream_reasoning_id,
                    include_empty_reasoning=think_enabled,
                )
                if stats:
                    _log_performance(model, err_resp.get("performance"))
                return err_resp
        finally:
            monitor_stop.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=1.0)
            lease.release()
        text = truncate_stop_markers("".join(chunks), all_stop_markers)
        thinking = None
        reasoning_tokens = 0
        if think_enabled:
            thinking, text = _split_generated_thinking(text, generation_prefix)
            thinking = thinking or None
            if thinking is not None:
                reasoning_tokens = max(0, len(thinking) // 4)
        tool_calls, tool_error_code, tool_error_msg = _parse_tool_result(text, body)
        if tool_error_code is None:
            choice_error, drop = _enforce_tool_choice(
                tool_calls, tool_choice=getattr(body, "tool_choice", None)
            )
            if choice_error is not None:
                tool_error_code, tool_error_msg = "choice", choice_error
            elif drop:
                tool_calls = None
        remaining_text = (
            _strip_tool_json_from_text(text, tool_calls, tool_syntax=tool_syntax)
            if tool_calls
            else text
        )
        input_tokens = int(result.get("prompt_tokens") or 0) if result else 0
        output_tokens = (
            int(result.get("generated_tokens") or len(chunks))
            if result
            else len(chunks)
        )
        stop_reason_val = int(result.get("stop_reason") or 0) if result else 0
        incomplete_details = None
        final_status = ResponseStatus.completed
        item_status = "completed"
        error: dict[str, Any] | None = None
        if tool_error_code is not None:
            final_status = ResponseStatus.failed
            item_status = "incomplete"
            error = {
                "code": "server_error",
                "message": tool_error_msg or "tool call failed",
            }
        elif stop_reason_val == 3:
            final_status = ResponseStatus.cancelled
            item_status = "incomplete"
        elif stop_reason_val == 2:
            final_status = ResponseStatus.incomplete
            incomplete_details = {"reason": "max_output_tokens"}
            item_status = "incomplete"
        final_resp = build_response(
            body,
            response_id=f"resp_{uuid.uuid4().hex[:24]}",
            message_id=f"msg_{uuid.uuid4().hex[:24]}",
            created_at=int(time.time()),
            completed_at=int(time.time()),
            status=final_status,
            text=remaining_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking=thinking,
            reasoning_tokens=reasoning_tokens,
            result=result,
            incomplete_details=incomplete_details,
            item_status=item_status,
            reasoning_item_id=non_stream_reasoning_id,
            include_empty_reasoning=think_enabled,
            tool_calls=tool_calls,
            error=error,
        )
        if stats:
            _log_performance(model, final_resp.get("performance"))
        return final_resp

    @router.get("/responses/{response_id}")
    def get_response(response_id: str):
        with response_store_lock:
            resp = response_store.get(response_id)
        if not resp:
            raise HTTPException(status_code=404, detail="Response not found")
        return resp

    @router.post("/responses/{response_id}/cancel")
    def cancel_response(response_id: str):
        with response_store_lock:
            response = response_store.get(response_id)
            if response is None:
                raise HTTPException(status_code=404, detail="Response not found")
            if response["status"] != ResponseStatus.in_progress:
                raise HTTPException(
                    status_code=409, detail="Response is not in progress"
                )
        with active_streams_lock:
            entry = active_streams.get(response_id)
        if entry is not None:
            try:
                entry["lease"].cancel()
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Native session cancellation failed: {exc}"
                ) from exc
            finished: threading.Event = entry["finished"]
            if not finished.wait(timeout=max(0.0, cancel_wait_seconds)):
                raise HTTPException(
                    status_code=504,
                    detail="Generation did not stop before the cancellation deadline",
                )
        else:
            raise HTTPException(status_code=409, detail="Response is not in progress")
        with response_store_lock:
            cur = response_store.get(response_id)
            if cur is not None and cur["status"] == ResponseStatus.in_progress:
                cur["status"] = ResponseStatus.cancelled
                return cur
            if cur is not None:
                return cur
            return response

    model_created_at = int(time.time())

    def _model_obj(mid: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": mid,
            "object": "model",
            "created": model_created_at,
            "owned_by": "cke",
            # Reasoning capability block (OpenRouter-shaped): harnesses gate
            # their Effort picker on its presence ("none" = off,
            # "default" = on). No custom capability fields — strict
            # discovery parsers must see only standard shapes here.
            "reasoning": {
                "supported_efforts": ["none", "default"],
                "default_effort": "default",
                "default_enabled": False,
            },
            "supported_parameters": ["reasoning", "reasoning_effort"],
        }
        if context_length is not None:
            result["cke_context_length"] = context_length
            result["cke_default_max_output_tokens"] = max_tokens
        return result

    @router.get("/models")
    def list_models():
        return {"object": "list", "data": [_model_obj(model)]}

    @router.get("/models/{model_id}")
    def retrieve_model(model_id: str):
        if model_id != model:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id!r} not found; available model: {model!r}",
            )
        return _model_obj(model_id)

    @router.get("/health")
    def health():
        return {"status": "ok", "mode": "live", "inference": True}

    @router.get("/cke/loaded-identity")
    def get_loaded_identity():
        if attestation is None:
            raise HTTPException(status_code=404, detail="loaded bundle identity unavailable")
        return {**attestation, "server_instance_id": server_instance_id,
                "effective_serving": effective_serving}

    if extra_route_registrar is not None:
        extra_route_registrar(router, create_response)

    app = FastAPI(title="CKE v8 live Responses server", version="0.2.0")
    app.include_router(router, prefix="/v1")
    app.include_router(conversations_router, prefix="/v1")

    @app.exception_handler(HTTPException)
    async def _harness_error_handler(request: Request, exc: HTTPException):
        """Render HTTP errors as ``{"error": {...}, "detail": message}``.

        Harness SDKs parse ``error`` (OpenAI shape); existing ``detail``
        readers keep working unchanged.
        """
        detail = exc.detail
        if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
            error = dict(detail["error"])
            message = str(error.get("message", ""))
        else:
            message = str(detail)
            error = {
                "message": message,
                "type": "invalid_request_error",
                "code": "invalid_request",
            }
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": error, "detail": message},
            headers=dict(exc.headers or {}),
        )

    if viz_html is not None:

        @app.get("/viz", response_class=HTMLResponse)
        def viz_page():
            return HTMLResponse(viz_html)

    app.state.response_store = response_store
    app.state.response_history_store = response_history_store
    app.state.response_store_lock = response_store_lock
    app.state.active_streams = active_streams
    app.state.active_streams_lock = active_streams_lock
    app.state.flight_lock = _flight_lock
    app.state.session = session

    return app


def open_live_session(
    run_dir: str | Path,
    *,
    context_length: int | None = None,
    num_threads: int | None = None,
):
    """Open a native session for ``run_dir`` (imports native binding lazily)."""
    from .runtime import load_manifest_templates, resolve_runtime_context_length
    from .session_v8 import SessionV8

    run_dir = Path(run_dir).expanduser().resolve()
    capacity = resolve_runtime_context_length(run_dir, context_length)
    session = SessionV8.open(run_dir, context_length=capacity, num_threads=num_threads)
    _, _, contract = load_manifest_templates(run_dir)
    return session, capacity, contract


def create_live_app_from_run_dir(
    run_dir: str | Path,
    *,
    model: str = "ck-v8",
    context_length: int | None = None,
    num_threads: int | None = None,
    **kwargs: Any,
):
    """Build a live app directly from a compiled runtime directory."""
    from .runtime import load_manifest_templates, load_tool_protocol
    from .serving_bundle import (
        load_resolved_serving, loaded_serving_identity, resolved_renderer_tokens,
        verify_loaded_libraries,
    )

    run_dir = Path(run_dir).expanduser().resolve()
    chat_template, chat_templates, contract = load_manifest_templates(run_dir)
    resolved = load_resolved_serving(run_dir)
    renderer_tokens = resolved_renderer_tokens(run_dir, resolved) if resolved is not None else None
    if not chat_template and not kwargs.get("allow_untemplated", False):
        raise ValueError(
            f"normal chat serving requires {run_dir / 'chat_template.jinja'} "
            "before opening the native session"
        )
    if not chat_template:
        kwargs["flags"] = int(kwargs.get("flags", 0)) | CK_SESSION_REQUEST_RAW_PROMPT
    sidecar_protocol = load_tool_protocol(run_dir, chat_template, chat_templates)
    explicit_protocol = kwargs.pop("tool_protocol", None)
    if explicit_protocol is not None and sidecar_protocol is not None and explicit_protocol != sidecar_protocol:
        raise ValueError(
            f"explicit tool protocol {explicit_protocol!r} conflicts with "
            f"tool_protocol.json declaration {sidecar_protocol!r}"
        )
    tool_protocol = explicit_protocol or sidecar_protocol
    session, capacity, _ = open_live_session(
        run_dir, context_length=context_length, num_threads=num_threads
    )
    try:
        resolved_serving = load_resolved_serving(run_dir)
        loaded_identity = None
        if resolved_serving is not None:
            verify_loaded_libraries(resolved_serving)
            loaded_identity = loaded_serving_identity(
                resolved_serving,
                model=model,
                session_library=session.lib,
            )
    except Exception:
        session.close()
        raise
    return create_app(
        session,
        model=model,
        context_length=capacity,
        chat_contract=contract,
        chat_template=chat_template,
        chat_templates=chat_templates,
        tool_protocol=tool_protocol,
        loaded_identity=loaded_identity,
        renderer_tokens=renderer_tokens,
        **kwargs,
    )
