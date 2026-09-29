"""Unit tests for the text-generation thinking splitter.

Covers both marker styles (open+close and Qwen3-style close-only), open-only
output, no markers, case-insensitivity, and the streaming splitter.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "version" / "v8" / "scripts"))

from ck_serve_v8 import _prompt_opens_thinking, _StreamThinkSplitter, split_thinking


def test_split_thinking_open_and_close_markers():
    text = "<think>\nLet me think\n</think>\n42"
    assert split_thinking(text) == ("Let me think", "42")


def test_split_thinking_open_only():
    text = "<think>\nstill reasoning"
    assert split_thinking(text) == ("still reasoning", "")


def test_split_thinking_no_markers():
    assert split_thinking("plain answer text") == ("", "plain answer text")


def test_split_thinking_empty_input():
    assert split_thinking("") == ("", "")


def test_split_thinking_line_anchored_markers_only():
    text = "before\n<think>\nmid\n</think>\nafter"
    assert split_thinking(text) == ("mid", "after")


def test_stream_splitter_reconstructs_thinking_and_answer():
    splitter = _StreamThinkSplitter()
    thinking = []
    answer = []
    for chunk in ("<think>", "\nLet me think", "\n</think>", "\n42"):
        for state, delta in splitter.feed(chunk):
            (thinking if state == "thinking" else answer).append(delta)
    for state, delta in splitter.flush():
        (thinking if state == "thinking" else answer).append(delta)
    assert "".join(thinking) == "Let me think"
    assert answer == ["42"]


def test_stream_splitter_close_only_marks_prefix_as_thinking():
    splitter = _StreamThinkSplitter()
    states = []
    for chunk in ("a", " b", "</think>", " c"):
        states.extend(splitter.feed(chunk))
    states.extend(splitter.flush())
    assert states == [("thinking", "a b"), ("answer", "c")]


def test_stream_splitter_no_markers_emits_only_answer_on_flush():
    splitter = _StreamThinkSplitter()
    states = []
    for chunk in ("plain", " ", "answer"):
        states.extend(splitter.feed(chunk))
    states.extend(splitter.flush())
    assert states == [("answer", "plain answer")]


def test_prompt_opens_thinking_detects_unclosed_marker():
    assert _prompt_opens_thinking("<|im_start|>assistant\n<think>\n")
    assert not _prompt_opens_thinking("quoted <think> user text")
    assert not _prompt_opens_thinking("<|im_start|>assistant\n")
    assert not _prompt_opens_thinking("a<think>b</think>c")
    assert not _prompt_opens_thinking("")


def test_stream_splitter_prompt_opened_thinking_streams_token_wise():
    splitter = _StreamThinkSplitter(start_thinking=True)
    thinking = []
    answer = []
    for chunk in ("The user", " is asking", " for this", "</think>", "done"):
        for state, delta in splitter.feed(chunk):
            (thinking if state == "thinking" else answer).append(delta)
    for state, delta in splitter.flush():
        (thinking if state == "thinking" else answer).append(delta)
    assert len(thinking) > 1
    assert "".join(thinking) == "The user is asking for this"
    assert "".join(answer) == "done"


def test_stream_splitter_prompt_opened_strips_reemitted_open_tag():
    splitter = _StreamThinkSplitter(start_thinking=True)
    thinking = []
    for chunk in ("<thi", "nk>\nhello", "</think>"):
        for state, delta in splitter.feed(chunk):
            assert state == "thinking"
            thinking.append(delta)
    assert "".join(thinking) == "hello"
