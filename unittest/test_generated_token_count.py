#!/usr/bin/env python3
"""Compile generated count entries against the native tokenizer, without weights."""

import ctypes
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
import build_ir_v8  # noqa: E402


class GeneratedTokenCountTests(unittest.TestCase):
    def _library(self, kind):
        generated = build_ir_v8._generate_tokenizer_c_code(kind, vocab_size=32, num_merges=0)
        code = generated["api_functions"]
        match = re.search(r"CK_EXPORT int ck_model_count_text_tokens\([^)]*\) \{.*?\n\}", code, re.S)
        self.assertIsNotNone(match)
        header = "tokenizer/true_bpe.h" if kind == "bpe" else "tokenizer/tokenizer.h"
        setup = """void enable_count_bpe_bos_eos(CKTrueBPE *tok) {
    CKBPEConfig config = {0};
    config.add_bos = true;
    config.add_eos = true;
    ck_true_bpe_set_config(tok, &config);
    ck_true_bpe_set_special_ids(tok, 0, 1, 2, -1);
}""" if kind == "bpe" else ""
        source = f"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <{header}>
#define CK_EXPORT __attribute__((visibility("default")))
typedef struct {{ void *tokenizer; unsigned char token_buffer[64]; }} CountModel;
static CountModel model;
static CountModel *g_model = &model;
void set_count_tokenizer(void *tokenizer) {{ model.tokenizer = tokenizer; }}
void fill_count_buffer(void) {{ memset(model.token_buffer, 0x5a, sizeof(model.token_buffer)); }}
int count_buffer_unchanged(void) {{
    for (int i = 0; i < 64; i++) if (model.token_buffer[i] != 0x5a) return 0;
    return 1;
}}
{match.group(0)}
{setup}
"""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        c_path = Path(temp.name) / "count.c"
        so_path = Path(temp.name) / "count.so"
        c_path.write_text(source)
        subprocess.run([
            "cc", "-shared", "-fPIC", "-I", str(ROOT / "include"),
            str(c_path), "-L", str(ROOT / "build"), "-lckernel_tokenizer",
            "-Wl,-rpath," + str(ROOT / "build"), "-o", str(so_path),
        ], check=True, capture_output=True)
        lib = ctypes.CDLL(str(so_path))
        lib.set_count_tokenizer.argtypes = [ctypes.c_void_p]
        lib.ck_model_count_text_tokens.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.ck_model_count_text_tokens.restype = ctypes.c_int
        return lib, ctypes.CDLL(str(ROOT / "build/libckernel_tokenizer.so"))

    def test_spm_count_matches_complete_encode(self):
        generated, native = self._library("sentencepiece")
        native.ck_tokenizer_create.argtypes = [ctypes.c_int]
        native.ck_tokenizer_create.restype = ctypes.c_void_p
        native.ck_tokenizer_add_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                   ctypes.c_int32, ctypes.c_float]
        native.ck_tokenizer_add_token.restype = ctypes.c_int
        native.ck_tokenizer_add_special_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                           ctypes.c_int32]
        native.ck_tokenizer_add_special_token.restype = ctypes.c_int
        native.ck_tokenizer_set_add_bos_eos.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool]
        native.ck_tokenizer_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        native.ck_tokenizer_encode.restype = ctypes.c_int
        native.ck_tokenizer_free.argtypes = [ctypes.c_void_p]
        tok = native.ck_tokenizer_create(2)
        self.assertTrue(tok)
        self.addCleanup(native.ck_tokenizer_free, tok)
        for index, piece in enumerate(("<unk>", "<s>", "</s>", "▁", "a")):
            self.assertEqual(native.ck_tokenizer_add_token(tok, piece.encode(), index, 0), 0)
        native.ck_tokenizer_set_add_bos_eos(tok, True, True)
        generated.set_count_tokenizer(tok)
        # Exact five-piece reproducer from the review: the old count buffer
        # fills at three IDs and silently drops EOS.
        short = (ctypes.c_int32 * 3)()
        self.assertEqual(native.ck_tokenizer_encode(tok, b"a", 1, short, 3), 3)
        self.assertEqual(list(short), [1, 3, 4])
        self.assertEqual(generated.ck_model_count_text_tokens(b"a", 1), 4)
        self.assertEqual(native.ck_tokenizer_add_token(tok, b"b", 5, 0), 0)
        self.assertEqual(native.ck_tokenizer_add_special_token(tok, b"<tool>", 6), 0)
        for text in ("a", "aa", "a b", "a☃b", "a<tool>b", "a" * 70):
            payload = text.encode()
            ids = (ctypes.c_int32 * 1024)()
            expected = native.ck_tokenizer_encode(tok, payload, len(payload), ids, 1024)
            self.assertGreater(expected, 0)
            self.assertEqual(ids[0], 1)
            self.assertEqual(ids[expected - 1], 2)
            generated.fill_count_buffer()
            self.assertEqual(generated.ck_model_count_text_tokens(payload, len(payload)), expected)
            self.assertEqual(generated.count_buffer_unchanged(), 1)

    def test_bpe_count_matches_complete_encode(self):
        generated, native = self._library("bpe")
        native.ck_true_bpe_create.restype = ctypes.c_void_p
        native.ck_true_bpe_free.argtypes = [ctypes.c_void_p]
        native.ck_true_bpe_add_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                 ctypes.c_int32, ctypes.c_float]
        native.ck_true_bpe_add_token.restype = ctypes.c_int
        native.ck_true_bpe_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                              ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        native.ck_true_bpe_encode.restype = ctypes.c_int
        tok = native.ck_true_bpe_create()
        self.assertTrue(tok)
        self.addCleanup(native.ck_true_bpe_free, tok)
        for index, piece in enumerate(("<unk>", "<s>", "</s>", "a", "b", " ", "☃")):
            self.assertEqual(native.ck_true_bpe_add_token(tok, piece.encode(), index, 0), 0)
        generated.enable_count_bpe_bos_eos.argtypes = [ctypes.c_void_p]
        generated.enable_count_bpe_bos_eos(tok)
        generated.set_count_tokenizer(tok)
        for payload in (b"a", b"ab", b"a b", "a☃b".encode(), b"ab" * 70):
            ids = (ctypes.c_int32 * 1024)()
            expected = native.ck_true_bpe_encode(tok, payload, len(payload), ids, 1024)
            self.assertGreater(expected, 0)
            self.assertEqual(ids[0], 1)
            self.assertEqual(ids[expected - 1], 2)
            generated.fill_count_buffer()
            self.assertEqual(generated.ck_model_count_text_tokens(payload, len(payload)), expected)
            self.assertEqual(generated.count_buffer_unchanged(), 1)


if __name__ == "__main__":
    unittest.main()
