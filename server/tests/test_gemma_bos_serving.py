"""Gemma GGUF-native chat-template serving regression, without a model download.

The fixture is the exact chat_template.jinja shipped in
unsloth/gemma-3-270m-it-GGUF (sha256 af95fbef33b7…), the template that
failed serving with `jinja UndefinedError: 'bos_token' is undefined` while
the CLI rendered it fine. These tests pin the serving-side contract:

- the publisher template renders with special-token variables resolved from
  runtime artifacts (never hardcoded),
- the rendered prompt matches the HF/transformers reference semantics
  (trim_blocks/lstrip_blocks; no stray comment-line newline), and
- the literal leading BOS is stripped before the native encoder, which
  auto-adds BOS itself — so encoded IDs match the CLI path exactly
  ([2, 105, 2364, 107, ...] for "hello" under gemma's add_bos contract).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.live import _render_with_chat_templates, create_app
from server.serving_bundle import runtime_auto_bos_token, runtime_renderer_tokens

FIXTURE = Path(__file__).parent / "fixtures" / "gemma3_gguf_chat_template.jinja"
FIXTURE_SHA256 = "af95fbef33b76a50e5f463ff9766f85ce84f5849d915c4bd6c1619d852ac3231"

# Resolved from the runtime's own tokenizer for gemma-3-270m-it (verified on
# x86 and the DGX Spark): bos 2 -> "<bos>", eos 106 -> "<end_of_turn>".
GEMMA_TOKENS = {"bos_token": "<bos>", "eos_token": "<end_of_turn>"}
CLI_REFERENCE_RENDER = "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n"


class RecordingSession:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.prompts: list[str] = []

    def generate(self, system, user, *, max_tokens, temperature, top_p, on_token,
                 flags=0, stop_on_text=(), stop_at_eos=False):
        self.prompts.append(user)
        on_token(1, next(self.outputs))
        return {"prompt_tokens": 1, "generated_tokens": 1, "stop_reason": 1}

    def count_tokens(self, text: str) -> int:
        return 1

    def close(self):
        pass


def test_fixture_is_the_publisher_template() -> None:
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256


def test_gemma_template_renders_with_resolved_bos_token() -> None:
    from server.live import TemplateRenderError

    template = FIXTURE.read_text(encoding="utf-8")
    messages = [{"role": "user", "content": "hello"}]
    with pytest.raises(TemplateRenderError, match="bos_token"):
        _render_with_chat_templates(template, None, messages, None)
    rendered = _render_with_chat_templates(
        template, None, messages, None, renderer_tokens=GEMMA_TOKENS)
    # Reference semantics: literal <bos> then the CLI/contract body, no stray
    # newline from the template's leading comment block.
    assert rendered == "<bos>" + CLI_REFERENCE_RENDER


def test_auto_bos_strip_keeps_serving_ids_aligned_with_cli() -> None:
    template = FIXTURE.read_text(encoding="utf-8")
    session = RecordingSession(["Hello!"])
    app = create_app(
        session,
        model="test",
        chat_template=template,
        renderer_tokens=dict(GEMMA_TOKENS),
        auto_bos_token_text="<bos>",
    )
    client = TestClient(app)
    response = client.post("/v1/responses", json={"model": "test", "input": "hello"})
    assert response.status_code == 200
    # The session receives the prompt without the literal BOS; the native
    # encoder adds BOS itself, yielding the CLI-identical token stream.
    assert session.prompts == [CLI_REFERENCE_RENDER]


def test_auto_bos_not_stripped_without_flag() -> None:
    template = FIXTURE.read_text(encoding="utf-8")
    session = RecordingSession(["Hello!"])
    app = create_app(
        session,
        model="test",
        chat_template=template,
        renderer_tokens=dict(GEMMA_TOKENS),
    )
    client = TestClient(app)
    response = client.post("/v1/responses", json={"model": "test", "input": "hello"})
    assert response.status_code == 200
    assert session.prompts == ["<bos>" + CLI_REFERENCE_RENDER]


def test_runtime_auto_bos_token_from_manifest(tmp_path: Path) -> None:
    (tmp_path / "weights_manifest.json").write_text(json.dumps({
        "special_tokens": {"bos_token_id": 2, "add_bos_token": True},
    }))
    assert runtime_auto_bos_token(tmp_path, decode_ids=lambda ids: "<bos>") == "<bos>"


def test_runtime_auto_bos_token_none_when_no_auto_add(tmp_path: Path) -> None:
    (tmp_path / "weights_manifest.json").write_text(json.dumps({
        "special_tokens": {"bos_token_id": 2, "add_bos_token": False},
    }))
    assert runtime_auto_bos_token(tmp_path, decode_ids=lambda ids: "<bos>") is None
    (tmp_path / "weights_manifest.json").unlink()
    assert runtime_auto_bos_token(tmp_path, decode_ids=lambda ids: "<bos>") is None
