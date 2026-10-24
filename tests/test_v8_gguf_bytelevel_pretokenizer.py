"""GGUF-declared byte-level token boundaries used by generated serving."""

from __future__ import annotations

import ctypes
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))

import build_ir_v8  # type: ignore
import convert_gguf_to_bump_v8 as converter  # type: ignore


class Config(ctypes.Structure):
    _fields_ = [
        ("add_bos", ctypes.c_bool),
        ("add_eos", ctypes.c_bool),
        ("byte_fallback", ctypes.c_bool),
        ("space_prefix_style", ctypes.c_int),
        ("pretokenizer", ctypes.c_int),
    ]


def test_gguf_tag_selects_declared_generated_policy() -> None:
    assert converter._gguf_pretokenizer_profile({"tokenizer.ggml.pre": "command-r"}) == "bytelevel_simple"
    assert converter._gguf_pretokenizer_profile({"tokenizer.ggml.pre": "other"}) is None
    generated = build_ir_v8._generate_tokenizer_c_code(
        "bpe", vocab_size=4, num_merges=1,
        tokenizer_contract={"tokenizer_type": "bpe", "pretokenizer": "bytelevel_simple"},
    )
    assert "cfg.pretokenizer = CK_BPE_PRETOKENIZER_BYTELEVEL_SIMPLE;" in generated["init"]


def test_unknown_declared_profile_cannot_fall_back_to_gpt2() -> None:
    import pytest

    with pytest.raises(ValueError, match="unsupported BPE pretokenizer"):
        build_ir_v8._generate_tokenizer_c_code(
            "bpe", vocab_size=4, num_merges=1,
            tokenizer_contract={"tokenizer_type": "bpe", "pretokenizer": "unknown"},
        )


def test_native_newline_boundaries_differ_from_gpt2() -> None:
    subprocess.run(["make", "build/libckernel_tokenizer.so"], cwd=ROOT,
                   check=True, capture_output=True)
    lib = ctypes.CDLL(str(ROOT / "build/libckernel_tokenizer.so"))
    lib.ck_true_bpe_create.restype = ctypes.c_void_p
    lib.ck_true_bpe_free.argtypes = [ctypes.c_void_p]
    lib.ck_true_bpe_load_binary.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
        ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ck_true_bpe_set_config.argtypes = [ctypes.c_void_p, ctypes.POINTER(Config)]
    lib.ck_true_bpe_encode.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
    ]
    vocab = ["A", "B", "Ċ", "ĊĊ", "ĉ", "Ċĉ", "Ġ", "ĠĠ", "1"]
    blob = b"".join(token.encode() + b"\0" for token in vocab)
    offsets = []
    next_offset = 0
    for token in vocab:
        offsets.append(next_offset)
        next_offset += len(token.encode()) + 1

    def encode(profile: int, prompt: str) -> list[int]:
        handle = lib.ck_true_bpe_create()
        assert handle
        try:
            merges = (ctypes.c_int32 * 9)(2, 2, 3, 2, 4, 5, 6, 6, 7)
            assert lib.ck_true_bpe_load_binary(
                handle, len(vocab), (ctypes.c_int32 * len(vocab))(*offsets),
                blob, 3, merges,
            ) == 0
            lib.ck_true_bpe_set_config(handle, ctypes.byref(Config(False, False, True, 1, profile)))
            prompt_bytes = prompt.encode()
            ids = (ctypes.c_int32 * 32)()
            count = lib.ck_true_bpe_encode(handle, prompt_bytes, len(prompt_bytes), ids, len(ids))
            assert count > 0
            return list(ids[:count])
        finally:
            lib.ck_true_bpe_free(handle)

    assert encode(0, "A\n\nB") == [0, 3, 1]
    assert encode(2, "A\n\nB") == [0, 2, 2, 1]
    assert encode(2, "A\n\n\nB") == [0, 3, 2, 1]
    assert encode(2, "A\n\t B") == [0, 5, 6, 1]
    assert encode(2, "A  1") == [0, 7, 8]
