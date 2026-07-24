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
    def test_last_logits_preserves_resolved_exact_gemm_provider(self) -> None:
        function = "gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage"
        op = {
            "function": function,
            "op": "logits",
            "layer": -1,
            "args": [
                {
                    "name": "A",
                    "source": "activation:a",
                    "expr": "(const float*)(model->bump + A_MAIN_STREAM)",
                },
                {
                    "name": "B",
                    "source": "weight:_first_weight",
                    "expr": "(const void*)(model->bump + W_LM_HEAD)",
                },
                {
                    "name": "C",
                    "source": "output:c",
                    "expr": "(float*)(model->bump + A_LOGITS)",
                },
                {"name": "M", "source": "dim:_m", "expr": "num_tokens"},
                {"name": "N", "source": "dim:_output_dim", "expr": "151936"},
                {"name": "K", "source": "dim:_input_dim", "expr": "4096"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(
            op,
            758,
            {"embed_dim": 4096, "vocab_size": 151936, "logits_layout": "last"},
        )

        self.assertIn("logits (last-only exact GEMM contract)", emitted)
        self.assertIn(function + "(", emitted)
        self.assertIn("(size_t)(num_tokens - 1) * 4096", emitted)
        self.assertNotIn(
            "gemv_bf16_pytorch_onednn_brgemm_bf16_storage",
            emitted,
        )

    def test_residual_save_exports_prefill_layer_input_before_normalization(self) -> None:
        op = {
            "function": "memcpy",
            "op": "residual_save",
            "layer": 0,
            "op_instance_idx": 0,
            "args": [
                {"name": "dst", "expr": "RESIDUAL"},
                {"name": "src", "expr": "LAYER_INPUT"},
                {"name": "size", "source": "dim:_memcpy_bytes", "expr": "4096"},
            ],
        }

        emitted = codegen_prefill_v8.emit_prefill_op(op, 1, {"embed_dim": 4096})

        self.assertIn(
            'ck_debug_export_hidden(model, 0, "layer_input", '
            "(const float*)LAYER_INPUT, (num_tokens) * (EMBED_DIM))",
            emitted,
        )

        op["op_instance_idx"] = 1
        emitted_after_attention = codegen_prefill_v8.emit_prefill_op(
            op, 16, {"embed_dim": 4096}
        )
        self.assertIn('"after_attn"', emitted_after_attention)
        self.assertNotIn('"layer_input"', emitted_after_attention)

    @staticmethod
    def _bridge_embedding_ops() -> list[dict]:
        return [{
            "op": "dense_embedding_lookup",
            "function": "embedding_forward_bf16_fp32",
            "args": [
                {"name": "token_ids", "expr": "TOKENS"},
                {"name": "token_embeddings", "expr": "WEIGHTS"},
                {"name": "output", "expr": "OUT"},
                {"name": "vocab_size", "expr": "VOCAB_SIZE"},
                {"name": "embed_dim", "expr": "EMBED_DIM"},
                {"name": "aligned_embed_dim", "expr": "EMBED_DIM"},
            ],
        }]

    def test_unified_mixed_bridge_emits_one_full_prefill(self) -> None:
        config = {
            "embed_dim": 16,
            "num_deepstack_layers": 3,
            "context_length": 32,
            "multimodal_bridge_contract": {
                "prefix_policy": "mixed_visual_text_prefill",
                "prefill_batching": "unified_mixed",
                "prefill_schedule": {
                    "segments": ["text_before", "visual", "text_after"],
                    "cache_transition": "single_pass",
                    "position_transition": "explicit_full_sequence",
                    "position_transform": {
                        "kernel_id": "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
                        "contract_id": "text_imrope_positions_bf16_input_pytorch_bf16_compute_bf16_output",
                        "resolved_function": "mrope_qk_text_imrope_positions_bf16_pytorch_storage",
                    },
                    "deepstack_injection": {
                        "kernel_id": "ck_residual_add_token_major_bf16_storage",
                        "contract_id": "residual_add_bf16_input_fp32_add_bf16_output",
                        "resolved_function": "ck_residual_add_token_major_bf16_storage",
                        "target": "visual_rows_after_decoder_layer",
                        "layers_from_config": "num_deepstack_layers",
                    },
                },
            },
        }
        emitted = codegen_prefill_v8.emit_multimodal_bridge_api(
            self._bridge_embedding_ops(), config
        )
        self.assertEqual(
            emitted.count("ck_prefill_from_embedded(g_model, total_tokens);"), 2
        )
        self.assertNotIn(
            "ck_prefill_from_embedded_range(g_model, prefix_tokens", emitted
        )
        helpers = codegen_prefill_v8._emit_multimodal_prefill_bridge_helpers(
            config, "mrope_qk_text_imrope_bf16_pytorch_storage"
        )
        self.assertIn("mrope_qk_text_imrope_positions_bf16_pytorch_storage(q, k,", helpers)
        self.assertIn("ck_residual_add_token_major_bf16_storage(dst_row, src, dst_row,", helpers)
        self.assertNotIn("mrope_qk_imrope_positions(q, k,", helpers)

    def test_unified_mixed_bridge_rejects_unresolved_position_provider(self) -> None:
        config = {
            "embed_dim": 16,
            "num_deepstack_layers": 3,
            "multimodal_bridge_contract": {
                "prefix_policy": "mixed_visual_text_prefill",
                "prefill_batching": "unified_mixed",
                "prefill_schedule": {
                    "segments": ["text_before", "visual", "text_after"],
                    "cache_transition": "single_pass",
                    "position_transition": "explicit_full_sequence",
                },
            },
        }
        with self.assertRaisesRegex(RuntimeError, "positions-aware M-RoPE provider"):
            codegen_prefill_v8._emit_multimodal_prefill_bridge_helpers(
                config, "mrope_qk_text_imrope_bf16_pytorch_storage"
            )

    def test_segmented_bridge_retains_three_cache_preserving_prefills(self) -> None:
        config = {
            "embed_dim": 16,
            "num_deepstack_layers": 3,
            "context_length": 32,
            "multimodal_bridge_contract": {
                "prefix_policy": "mixed_visual_text_prefill",
                "prefill_batching": "segmented_append",
                "prefill_schedule": {
                    "segments": ["text_before", "visual", "text_after"],
                    "cache_transition": "append_preserve",
                    "position_transition": "segment_defined",
                },
            },
        }
        emitted = codegen_prefill_v8.emit_multimodal_bridge_api(
            self._bridge_embedding_ops(), config
        )
        self.assertIn(
            "ck_prefill_from_embedded_range(g_model, prefix_tokens", emitted
        )
        self.assertEqual(
            emitted.count("ck_prefill_from_embedded(g_model, total_tokens);"), 1
        )

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
