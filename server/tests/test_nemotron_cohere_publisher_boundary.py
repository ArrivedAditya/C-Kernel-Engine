"""Publisher prompt fixtures for the pinned Nemotron and Cohere chat artifacts.

These check the common renderer and circuit linkage. Real tool execution and
model quality need separate artifact-backed acceptance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from server.live import _render_with_chat_templates


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
CASES = {
    "nemotron_h": {
        "fixture": "nemotron_nano_9b_v2_publisher_chat_template.jinja",
        "template_sha": "b7a3a520a4bc1beae6a25d615f9215580df8eea926f488256c2bef74039433c0",
        "hello_sha": "6e71ebeff3a1faf1133804545234c2f54235396f39b0120f70b23a37c83b175a",
        "multi_sha": "2ce674abce017008484fd0f088f55d494f528924a88ccc5af2f3c31c91630b6b",
    },
    "cohere2": {
        "fixture": "cohere_command_r7b_publisher_chat_template.jinja",
        "template_sha": "3732db8c378e5817a7b03183b0d80baaebce13a59b3a947336fd70572ce94ef3",
        "hello_sha": "9c65050030b33e663997425c409418802c4a75e467a221e8c7427a89901c6e9a",
        "multi_sha": "bd96304656f22a9ffadefe63af9602769efbfb1547f6faf4f7e8bd8c796981ba",
    },
}


@pytest.mark.parametrize("family", CASES)
def test_circuit_linked_chat_only_profile(family: str) -> None:
    circuit = json.loads((ROOT / f"version/v8/circuits/{family}.json").read_text())
    declaration = circuit["serving"]
    profile = json.loads((ROOT / "version/v8" / declaration["profile_ref"]).read_text())
    variant = profile["variants"][declaration["default_variant"]]
    assert variant["chat"] == {"source": "publisher", "variant": "default"}
    assert variant["output_protocol"] == "none"
    assert circuit["contract"]["tokenizer_contract"]["bos_policy"] == "publisher_template_no_implicit_bos"


@pytest.mark.parametrize("family", CASES)
def test_publisher_chat_prompt_bytes(family: str) -> None:
    case = CASES[family]
    source = (FIXTURES / case["fixture"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == case["template_sha"]
    body = SimpleNamespace(tools=None, instructions=None, tool_choice="auto")
    for label, messages in (
        ("hello", [{"role": "user", "content": "Hello."}]),
        ("multi", [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]),
    ):
        prompt = _render_with_chat_templates(
            source.decode(), None, messages, body,
            renderer_tokens={}, tool_protocol="none",
        )
        assert prompt is not None
        assert hashlib.sha256(prompt.encode()).hexdigest() == case[f"{label}_sha"]
