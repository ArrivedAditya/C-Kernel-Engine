#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEGEN_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_core_v8.py"
sys.path.insert(0, str(CODEGEN_PATH.parent))


def _load_codegen():
    spec = importlib.util.spec_from_file_location("codegen_core_hidden_export_tests", CODEGEN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {CODEGEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


codegen = _load_codegen()


def _arg(name: str, expr: str) -> dict[str, str]:
    return {"name": name, "expr": expr}


class HiddenExportExtentTests(unittest.TestCase):
    def test_mlp_projection_exports_cover_all_rows_and_channels(self) -> None:
        up = codegen.emit_op(
            {
                "op": "mlp_up",
                "function": "gemm_nt_bf16",
                "layer": 1,
                "args": [
                    _arg("a", "A"),
                    _arg("b", "B"),
                    _arg("bias", "BIAS"),
                    _arg("c", "UP"),
                    _arg("m", "4032"),
                    _arg("n", "4304"),
                    _arg("k", "1152"),
                ],
            }
        )
        down = codegen.emit_op(
            {
                "op": "mlp_down",
                "function": "gemm_nt_bf16",
                "layer": 1,
                "args": [
                    _arg("a", "UP"),
                    _arg("b", "B"),
                    _arg("bias", "BIAS"),
                    _arg("c", "DOWN"),
                    _arg("m", "4032"),
                    _arg("n", "1152"),
                    _arg("k", "4304"),
                ],
            }
        )

        self.assertIn('"mlp_up", (const float*)UP, (4032) * (4304)', up)
        self.assertIn('"mlp_up_last"', up)
        self.assertIn('(size_t)(4304)', up)
        self.assertIn('"mlp_down", (const float*)DOWN, (4032) * (1152)', down)
        self.assertIn('"mlp_down_last"', down)
        self.assertIn('(size_t)(1152)', down)

    def test_gelu_exports_the_full_post_activation_tensor(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "gelu",
                "function": "gelu_ggml_inplace",
                "layer": 1,
                "args": [_arg("data", "UP"), _arg("n", "17353728")],
            }
        )

        self.assertIn('"ffn_gelu", (const float*)UP, 17353728', emitted)

    def test_attention_norm_exports_the_projection_input_boundary(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "attn_norm",
                "function": "rmsnorm_forward",
                "layer": 0,
                "args": [
                    _arg("input", "X"),
                    _arg("weight", "W"),
                    _arg("output", "Y"),
                    _arg("bias", "NULL"),
                    _arg("num_tokens", "1"),
                    _arg("dim", "1024"),
                    _arg("aligned_dim", "1024"),
                    _arg("eps", "1e-6f"),
                ],
            }
        )

        self.assertIn('"attn_norm", (const float*)Y, (1) * (1024)', emitted)

    def test_qwen35_attention_exports_gate_and_pregate_boundaries(self) -> None:
        split = codegen.emit_op(
            {
                "op": "split_q_gate",
                "function": "split_q_gate_forward",
                "layer": 3,
                "args": [
                    _arg("packed_qg", "PACKED"),
                    _arg("q", "Q"),
                    _arg("gate", "GATE"),
                    _arg("rows", "1"),
                    _arg("q_dim", "2048"),
                    _arg("gate_dim", "2048"),
                ],
            }
        )
        attention = codegen.emit_op(
            {
                "op": "attn",
                "function": "attention_forward_decode_head_major_gqa_ggml_regular",
                "layer": 3,
                "args": [
                    _arg("q_token", "Q"),
                    _arg("k_cache", "K"),
                    _arg("v_cache", "V"),
                    _arg("out_token", "ATTN"),
                    _arg("num_heads", "8"),
                    _arg("num_tokens", "1"),
                    _arg("aligned_head_dim", "256"),
                ],
            }
        )
        gated = codegen.emit_op(
            {
                "op": "attn_gate_sigmoid_mul",
                "function": "attn_gate_sigmoid_mul_forward",
                "layer": 3,
                "args": [
                    _arg("x", "ATTN"),
                    _arg("gate", "GATE"),
                    _arg("out", "ATTN"),
                    _arg("rows", "18"),
                    _arg("num_heads", "8"),
                    _arg("state_dim", "256"),
                ],
            }
        )

        self.assertIn('"attn_gate", (const float*)GATE, (1) * (2048)', split)
        self.assertIn('"attn_pregate", (const float*)ATTN, (8) * (1) * (256)', attention)
        self.assertIn('"attn_out", (const float*)ATTN, (8) * (18) * (256)', gated)

    def test_attention_checkpoint_name_comes_from_call_ir_contract(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "attn",
                "function": "attention_forward",
                "layer": 2,
                "args": [
                    _arg("out_token", "ATTN"),
                    _arg("num_heads", "16"),
                    _arg("num_tokens", "1008"),
                    _arg("aligned_head_dim", "72"),
                ],
                "semantic_checkpoints": [
                    {
                        "id": "vision.layer.2.attention.output",
                        "tensor": "attn_out_head_major",
                    }
                ],
            }
        )

        self.assertIn(
            '"attn_out_head_major", (const float*)ATTN, (16) * (1008) * (72)',
            emitted,
        )
        self.assertNotIn('"attn_pregate"', emitted)

    def test_qwen35_prefill_exports_full_norm_and_swiglu_extents(self) -> None:
        norm = codegen.emit_op(
            {
                "op": "post_attention_norm",
                "function": "rmsnorm_forward_llama_production",
                "layer": 3,
                "args": [
                    _arg("output", "NORM"),
                    _arg("tokens", "18"),
                    _arg("d_model", "1024"),
                ],
            }
        )
        swiglu = codegen.emit_op(
            {
                "op": "silu_mul",
                "function": "swiglu_forward_ggml",
                "layer": 3,
                "args": [
                    _arg("output", "MLP"),
                    _arg("tokens", "18"),
                    _arg("dim", "3584"),
                ],
            }
        )
        self.assertIn('"post_attn_norm", (const float*)NORM, (18) * (1024)', norm)
        self.assertIn('"mlp_swiglu", (const float*)MLP, (18) * (3584)', swiglu)

    def test_quantized_projection_exports_full_prefill_extents(self) -> None:
        resolved = {
            "numerical_contract": "q4_k_x_q8_k_repacked_matmul_fp32",
            "implementation": {
                "weight_storage": {"format": "q4_k", "block_elements": 256, "block_bytes": 144},
                "activation_storage": {"format": "q8_k", "block_elements": 256},
                "diagnostic_providers": {"fp32_activation": "gemm_nt_q4_k"},
            }
        }
        emitted = codegen.emit_op(
            {
                "op": "out_proj",
                "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
                "layer": 3,
                "resolved_execution": resolved,
                "args": [
                    _arg("A", "Q8"),
                    _arg("B", "WEIGHT"),
                    _arg("C", "OUT"),
                    _arg("M", "18"),
                    _arg("N", "1024"),
                    _arg("K", "2048"),
                ],
            }
        )
        self.assertIn('"out_proj", (const float*)OUT, (18) * (1024)', emitted)

    def test_fused_gate_up_prefill_exports_combined_row_major_matrix(self) -> None:
        emitted = codegen.emit_op(
            {
                "op": "mlp_gate_up",
                "function": "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch",
                "layer": 3,
                "args": [
                    _arg("A", "Q8"),
                    _arg("B", "WEIGHT"),
                    _arg("C", "GATE_UP"),
                    _arg("M", "18"),
                    _arg("N", "7168"),
                    _arg("K", "1024"),
                ],
            }
        )
        self.assertIn('"mlp_gate_up", (const float*)GATE_UP, (18) * (7168)', emitted)
        self.assertIn('"mlp_gate_up_last"', emitted)
        self.assertIn('(size_t)(7168)', emitted)


if __name__ == "__main__":
    unittest.main()
