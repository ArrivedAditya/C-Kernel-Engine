#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEGEN_PREFILL_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_prefill_v8.py"


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codegen_prefill_v8 = _load_module("codegen_prefill_v8_tests", CODEGEN_PREFILL_PATH)


class TestV8PrefillCodegen(unittest.TestCase):
    def test_qwen35_post_attention_prefill_exports_full_checkpoint_extents(self) -> None:
        cases = [
            (
                {
                    "function": "attn_gate_sigmoid_mul_forward",
                    "op": "attn_gate_sigmoid_mul",
                    "layer": 3,
                    "args": [
                        {"name": "out", "expr": "ATTN"},
                        {"name": "rows", "expr": "1034"},
                        {"name": "num_heads", "expr": "8"},
                        {"name": "state_dim", "expr": "256"},
                    ],
                },
                '"attn_out", (const float*)ATTN, (8) * (num_tokens) * (256)',
            ),
            (
                {
                    "function": "rmsnorm_forward_llama_production",
                    "op": "post_attention_norm",
                    "layer": 3,
                    "args": [
                        {"name": "output", "expr": "NORM"},
                        {"name": "tokens", "expr": "1034"},
                        {"name": "d_model", "expr": "1024"},
                    ],
                },
                '"post_attn_norm", (const float*)NORM, (num_tokens) * (1024)',
            ),
            (
                {
                    "function": "swiglu_forward_ggml",
                    "op": "silu_mul",
                    "layer": 3,
                    "args": [
                        {"name": "output", "expr": "MLP"},
                        {"name": "tokens", "expr": "1034"},
                        {"name": "dim", "expr": "3584"},
                    ],
                },
                '"mlp_swiglu", (const float*)MLP, (num_tokens) * (3584)',
            ),
        ]
        for op, expected in cases:
            with self.subTest(op=op["op"]):
                emitted = codegen_prefill_v8.emit_prefill_op(op, 1, {"embed_dim": 1024})
                self.assertIn(expected, emitted)

    def test_fp16_cache_batch_copy_targets_physical_cache_before_attention(self) -> None:
        op = {
            "function": "kv_cache_batch_copy",
            "op": "kv_cache_batch_copy",
            "layer": 3,
            "section": "body",
            "args": [],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            11,
            {
                "decode_kv_cache_dtype": "fp16",
                "num_kv_heads": 8,
                "head_dim": 64,
                "context_len": 1034,
            },
        )

        self.assertIn("uint16_t *kv_cache = (uint16_t*)model->kv_cache_f16;", emitted)
        self.assertIn("ck_fp32_to_fp16_soft(ks[d])", emitted)
        self.assertIn("ck_fp32_to_fp16_soft(vs[d])", emitted)
        self.assertIn("(size_t)prefill_start_pos", emitted)
        self.assertNotIn("memcpy(", emitted)

    def test_recurrent_prefill_seq_len_args_use_runtime_num_tokens(self) -> None:
        op = {
            "function": "recurrent_split_qkv_forward",
            "op": "recurrent_split_qkv",
            "layer": 0,
            "section": "body",
            "args": [
                {"name": "packed_qkv", "expr": "(const float*)(model->bump + A_RECURRENT_PACKED)"},
                {"name": "q", "expr": "(float*)(model->bump + A_RECURRENT_Q)"},
                {"name": "k", "expr": "(float*)(model->bump + A_RECURRENT_K)"},
                {"name": "v", "expr": "(float*)(model->bump + A_RECURRENT_V)"},
                {"name": "seq_len", "expr": "1034"},
                {"name": "q_dim", "expr": "2048"},
                {"name": "k_dim", "expr": "2048"},
                {"name": "v_dim", "expr": "2048"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(op, 7, {"embed_dim": 1024})

        self.assertIn("num_tokens", emitted)
        self.assertNotIn("\n        1034,", emitted)
        self.assertIn("\n        2048,", emitted)

    def test_scalar_constant_is_not_rewritten_as_runtime_token_count(self) -> None:
        op = {
            "function": "recurrent_dt_gate_forward",
            "op": "recurrent_dt_gate",
            "layer": 0,
            "section": "body",
            "args": [
                {"name": "alpha", "expr": "alpha"},
                {"name": "dt_bias", "expr": "dt_bias"},
                {"name": "a", "expr": "a"},
                {"name": "gate", "expr": "gate"},
                {"name": "rows", "source": "dim:seq_len", "expr": "1034"},
                {"name": "num_heads", "source": "dim:gate_dim", "expr": "16"},
                {"name": "state_dim", "source": "const:1", "expr": "1"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(op, 9, {"embed_dim": 1024})

        self.assertIn("\n        num_tokens,\n        16,\n        1\n", emitted)
        self.assertNotIn("\n        num_tokens,\n        16,\n        num_tokens\n", emitted)

    def test_recurrent_prefill_exports_full_batched_projection_boundary(self) -> None:
        op = {
            "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
            "op": "recurrent_gate_proj",
            "layer": 0,
            "section": "body",
            "args": [
                {"name": "A", "expr": "activation"},
                {"name": "B", "expr": "weights"},
                {"name": "bias", "expr": "NULL"},
                {"name": "C", "expr": "gate_output"},
                {"name": "M", "source": "dim:seq_len", "expr": "1034"},
                {"name": "N", "source": "dim:attn_gate_dim", "expr": "2048"},
                {"name": "K", "source": "dim:embed_dim", "expr": "1024"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(op, 3, {"embed_dim": 1024})

        self.assertIn('ck_debug_export_hidden(model, 0, "z"', emitted)
        self.assertIn("(num_tokens) * (2048)", emitted)


if __name__ == "__main__":
    unittest.main()
