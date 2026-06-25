#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V8_BUILD_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"


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


build_ir_v8 = _load_module("build_ir_v8_kimi_tests", V8_BUILD_PATH)


def _entry(name: str, dtype: str, shape: list[int], offset: int) -> dict:
    nbytes_per = {"fp32": 4, "bf16": 2, "fp16": 2, "q8_0": 1, "q5_0": 1, "q6_k": 1, "q4_k": 1}.get(dtype, 4)
    size = 1
    for dim in shape:
        size *= int(dim)
    return {"name": name, "dtype": dtype, "offset": offset, "shape": shape, "nbytes": size * nbytes_per}


def _make_tiny_kimi_manifest() -> dict:
    offset = 0
    entries = []

    def add(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        item = _entry(name, dtype, shape, offset)
        entries.append(item)
        offset += int(item["nbytes"])

    add("token_emb", "bf16", [32, 8])
    add("final_ln_weight", "fp32", [8])
    add("final_ln_bias", "fp32", [8])
    add("output.weight", "bf16", [32, 8])
    for layer in range(2):
        add(f"layer.{layer}.block_norm", "fp32", [8])
        add(f"layer.{layer}.post_attention_norm", "fp32", [8])
        add(f"layer.{layer}.mla_q_proj", "bf16", [8, 8])
        add(f"layer.{layer}.mla_kv_a_proj", "bf16", [6, 8])
        add(f"layer.{layer}.mla_kv_a_norm", "fp32", [4])
        add(f"layer.{layer}.mla_kv_b_proj", "bf16", [8, 4])
        add(f"layer.{layer}.mla_out_proj", "bf16", [8, 8])
    add("layer.0.mlp_gate", "bf16", [16, 8])
    add("layer.0.mlp_up", "bf16", [16, 8])
    add("layer.0.mlp_down", "bf16", [8, 16])
    add("layer.1.moe_router", "fp32", [2, 8])
    add("layer.1.moe_router_bias", "fp32", [2])
    add("layer.1.moe_expert_gate", "bf16", [2, 4, 8])
    add("layer.1.moe_expert_up", "bf16", [2, 4, 8])
    add("layer.1.moe_expert_down", "bf16", [2, 8, 4])
    add("layer.1.moe_shared_gate", "bf16", [4, 8])
    add("layer.1.moe_shared_up", "bf16", [4, 8])
    add("layer.1.moe_shared_down", "bf16", [8, 4])

    return {
        "config": {
            "model": "kimi_vl",
            "arch": "kimi_vl",
            "model_type": "kimi_vl",
            "num_layers": 2,
            "embed_dim": 8,
            "hidden_size": 8,
            "num_heads": 2,
            "num_kv_heads": 2,
            "head_dim": 4,
            "intermediate_size": 16,
            "intermediate_dim": 16,
            "moe_intermediate_size": 4,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "vocab_size": 32,
            "context_length": 16,
            "layer_kinds": ["mla_dense_mlp", "mla_moe"],
            "layer_attention_policy": ["mla", "mla"],
            "layer_moe_policy": ["none", "routed_swiglu"],
            "rope_layout": "partial_pairwise_concat",
        },
        "quant_summary": {
            "token_emb": "bf16",
            "lm_head": "bf16",
            "final_ln_weight": "fp32",
            "layer.0": {
                "mla_q_proj": "bf16",
                "mla_kv_a_proj": "bf16",
                "mla_kv_a_norm": "fp32",
                "mla_kv_b_proj": "bf16",
                "mla_out_proj": "bf16",
                "mlp_gate": "bf16",
                "mlp_up": "bf16",
                "mlp_down": "bf16",
            },
            "layer.1": {
                "mla_q_proj": "bf16",
                "mla_kv_a_proj": "bf16",
                "mla_kv_a_norm": "fp32",
                "mla_kv_b_proj": "bf16",
                "mla_out_proj": "bf16",
                "moe_router": "fp32",
                "moe_router_bias": "fp32",
                "moe_expert_gate": "bf16",
                "moe_expert_up": "bf16",
                "moe_expert_down": "bf16",
                "moe_shared_gate": "bf16",
                "moe_shared_up": "bf16",
                "moe_shared_down": "bf16",
            },
        },
        "entries": entries,
        "template": build_ir_v8._load_builtin_template_doc("kimi_vl"),
    }


class V8KimiTemplateTests(unittest.TestCase):
    def test_kimi_mla_moe_template_lowers_to_reference_kernels(self) -> None:
        manifest = _make_tiny_kimi_manifest()
        ops = build_ir_v8.build_ir1_direct(manifest, ROOT / "tests" / "kimi.synthetic.json", mode="decode")
        by_layer_op = {(op.get("layer"), op.get("op"), op.get("instance", 0)): op for op in ops}

        self.assertEqual([op["op"] for op in ops].count("residual_save"), 4)
        self.assertEqual(by_layer_op[(0, "q_proj", 0)]["kernel"], "gemv_bf16")
        self.assertEqual(by_layer_op[(0, "kv_a_proj", 0)]["kernel"], "gemv_bf16")
        self.assertEqual(by_layer_op[(0, "kv_lora_decompress", 0)]["kernel"], "deepseek_mla_kv_decompress_f32")
        self.assertEqual(by_layer_op[(0, "partial_rope_concat", 0)]["kernel"], "deepseek_mla_partial_rope_concat_packed_f32")
        self.assertEqual(by_layer_op[(1, "moe_swiglu_expert_mlp", 0)]["kernel"], "moe_swiglu_expert_forward_f32")
        self.assertEqual(by_layer_op[(1, "shared_swiglu_expert_mlp", 0)]["kernel"], "moe_swiglu_shared_forward_f32")

        q_source = by_layer_op[(0, "q_proj", 0)]["dataflow"]["inputs"]["x"]
        kv_source = by_layer_op[(0, "kv_a_proj", 0)]["dataflow"]["inputs"]["x"]
        router_source = by_layer_op[(1, "moe_router", 0)]["dataflow"]["inputs"]["x"]
        self.assertEqual(q_source["slot"], "layer_input")
        self.assertEqual(q_source["from_op"], by_layer_op[(0, "block_rmsnorm", 0)]["op_id"])
        self.assertEqual(kv_source["slot"], "layer_input")
        self.assertEqual(router_source["slot"], "layer_input")
        self.assertEqual(router_source["from_op"], by_layer_op[(1, "block_rmsnorm", 1)]["op_id"])

        routed_weights = by_layer_op[(1, "moe_swiglu_expert_mlp", 0)]["weights"]
        self.assertIn("moe_expert_gate", routed_weights)
        self.assertIn("moe_expert_up", routed_weights)
        self.assertIn("moe_expert_down", routed_weights)


if __name__ == "__main__":
    unittest.main()
