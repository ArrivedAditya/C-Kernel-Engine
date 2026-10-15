"""Pinned publisher-template bytes through CKE's generic Jinja renderer.

The fixture is Muse-Glimmer-30B's imported chat_template.jinja. Expected
prompt hashes and token counts come from Transformers 5.13.0 with the
publisher tokenizer on 2026-10-15; the tokenizer itself is not vendored.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from server.live import _render_with_chat_templates


FIXTURE = Path(__file__).parent / "fixtures/muse_glimmer_publisher_chat_template.jinja"
FIXTURE_SHA256 = "cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678"
TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


class _Tool:
    def model_dump(self):
        return TOOL


@pytest.mark.parametrize(
    ("case", "expected_sha256"),
    [
        ("plain", "5700386d33ee9aa1c8a70885abb9c90f3189755473e42e4576f0b962e9f9e3b9"),
        ("tools", "a9cae4d505d46f241ca19bb5f3e369fb3210fb8d59d2dda30bfc49e0ad674165"),
        ("continuation", "70d3a4831a4e6e432c2c0ec485904d8866a1eb871979451c54a78dfbfa57a313"),
    ],
)
def test_muse_publisher_prompt_bytes(case: str, expected_sha256: str) -> None:
    template_bytes = FIXTURE.read_bytes()
    assert hashlib.sha256(template_bytes).hexdigest() == FIXTURE_SHA256
    messages = [{"role": "user", "content": "Hello, Muse." if case == "plain" else "Read /tmp/a"}]
    body = None
    if case != "plain":
        body = SimpleNamespace(tools=[_Tool()], instructions=None, tool_choice="auto")
    if case == "continuation":
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_1", "type": "function", "function": {
                    "name": "read_file", "arguments": {"path": "/tmp/a"},
                },
            }]},
            {"role": "tool", "name": "read_file", "tool_call_id": "call_1", "content": "hello"},
        ])
    rendered = _render_with_chat_templates(
        template_bytes.decode(), None, messages, body,
        renderer_tokens={"bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>"},
        render_time=datetime(2026, 10, 15), tool_protocol="none",
    )
    assert rendered is not None
    assert hashlib.sha256(rendered.encode()).hexdigest() == expected_sha256
    assert "&#34;" not in rendered
    assert rendered.endswith("<|start|>assistant")
