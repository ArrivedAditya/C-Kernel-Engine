import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str((REPO_ROOT / "version" / "v8" / "scripts").resolve()))

from scripts.chat_contract import build_chat_contract, load_template_chat_contract  # type: ignore
from convert_gguf_to_bump_v8 import (  # type: ignore
    TensorInfo,
    build_gemma4_attention_plan,
    classify_layer_contract,
    describe_layer_contract,
    _inject_runtime_config_defaults,
)


def _tensor(name: str, dims: tuple[int, ...]) -> TensorInfo:
    return TensorInfo(name=name, dims=dims, ggml_type=12, offset=0)


class V8Gemma4ScaffoldTests(unittest.TestCase):
    def test_load_template_chat_contract_gemma4(self) -> None:
        contract = load_template_chat_contract("gemma4")
        self.assertIsNotNone(contract)
        self.assertEqual(contract["turn_prefix"], "<|turn>{role}\n")
        self.assertEqual(contract["turn_suffix"], "<turn|>\n")
        self.assertEqual(contract["assistant_generation_prefix"], "<|turn>model\n")
        self.assertEqual(contract["token_stop_markers"], ["<turn|>"])

    def test_gemma4_uses_q8_activation_logits_by_default(self) -> None:
        gemma4 = _inject_runtime_config_defaults({}, "gemma4")
        gemma3 = _inject_runtime_config_defaults({}, "gemma3")
        self.assertFalse(gemma4["prefer_fp32_logits"])
        self.assertTrue(gemma4["prefer_q8_0_contract"])
        self.assertTrue(gemma3["prefer_fp32_logits"])

    def test_build_chat_contract_detects_gemma4_markers(self) -> None:
        chat_template = "<|turn>user\nHello<turn|>\n<|turn>model\n"
        contract = build_chat_contract(
            template_data=None,
            chat_template=chat_template,
            finetune="it",
            model_name="Gemma-4-E4B-It",
            model_type="gemma4",
        )
        self.assertIsNotNone(contract)
        self.assertEqual(contract["name"], "gemma4")
        self.assertIn("<|turn>", contract["template_markers"])
        self.assertIn("<turn|>", contract["template_markers"])

    def test_classify_layer_contract_gemma4_hybrid(self) -> None:
        tensors = {
            "blk.0.attn_q.weight": _tensor("blk.0.attn_q.weight", (2560, 2048)),
            "blk.0.attn_k.weight": _tensor("blk.0.attn_k.weight", (2560, 512)),
            "blk.0.attn_v.weight": _tensor("blk.0.attn_v.weight", (2560, 512)),
            "blk.0.attn_output.weight": _tensor("blk.0.attn_output.weight", (2048, 2560)),
            "blk.0.attn_q_norm.weight": _tensor("blk.0.attn_q_norm.weight", (256,)),
            "blk.0.attn_k_norm.weight": _tensor("blk.0.attn_k_norm.weight", (256,)),
            "blk.0.attn_norm.weight": _tensor("blk.0.attn_norm.weight", (2560,)),
            "blk.0.post_attention_norm.weight": _tensor("blk.0.post_attention_norm.weight", (2560,)),
            "blk.0.ffn_gate.weight": _tensor("blk.0.ffn_gate.weight", (2560, 10240)),
            "blk.0.ffn_up.weight": _tensor("blk.0.ffn_up.weight", (2560, 10240)),
            "blk.0.ffn_down.weight": _tensor("blk.0.ffn_down.weight", (10240, 2560)),
            "blk.0.inp_gate.weight": _tensor("blk.0.inp_gate.weight", (2560, 256)),
            "blk.0.proj.weight": _tensor("blk.0.proj.weight", (256, 2560)),
        }
        self.assertEqual(classify_layer_contract(tensors, 0), "gemma4_hybrid")

    def test_describe_layer_contract_gemma4_hybrid_is_specific(self) -> None:
        tensors = {
            "blk.0.attn_q.weight": _tensor("blk.0.attn_q.weight", (2560, 2048)),
            "blk.0.attn_k.weight": _tensor("blk.0.attn_k.weight", (2560, 512)),
            "blk.0.attn_v.weight": _tensor("blk.0.attn_v.weight", (2560, 512)),
            "blk.0.attn_output.weight": _tensor("blk.0.attn_output.weight", (2048, 2560)),
            "blk.0.attn_q_norm.weight": _tensor("blk.0.attn_q_norm.weight", (256,)),
            "blk.0.attn_k_norm.weight": _tensor("blk.0.attn_k_norm.weight", (256,)),
            "blk.0.attn_norm.weight": _tensor("blk.0.attn_norm.weight", (2560,)),
            "blk.0.post_attention_norm.weight": _tensor("blk.0.post_attention_norm.weight", (2560,)),
            "blk.0.ffn_gate.weight": _tensor("blk.0.ffn_gate.weight", (2560, 10240)),
            "blk.0.ffn_up.weight": _tensor("blk.0.ffn_up.weight", (2560, 10240)),
            "blk.0.ffn_down.weight": _tensor("blk.0.ffn_down.weight", (10240, 2560)),
            "blk.0.inp_gate.weight": _tensor("blk.0.inp_gate.weight", (2560, 256)),
            "blk.0.proj.weight": _tensor("blk.0.proj.weight", (256, 2560)),
        }
        detail = describe_layer_contract(tensors, 0, arch="gemma4")
        self.assertIsNotNone(detail)
        self.assertIn("Gemma4 hybrid block", detail)
        self.assertIn("per-layer projection lowering", detail)

    def test_build_gemma4_attention_plan_makes_shared_kv_explicit(self) -> None:
        plan = build_gemma4_attention_plan(
            {
                "gemma4.attention.sliding_window_pattern": [True, False, True, False],
                "gemma4.attention.shared_kv_layers": 2,
                "gemma4.attention.sliding_window": 1024,
                "gemma4.attention.head_count": 16,
                "gemma4.attention.head_count_kv": 4,
                "gemma4.attention.key_length": 256,
                "gemma4.attention.value_length": 256,
                "gemma4.attention.key_length_swa": 128,
                "gemma4.attention.value_length_swa": 128,
                "gemma4.rope.dimension_count": 512,
                "gemma4.rope.dimension_count_swa": 256,
            },
            4,
        )
        self.assertEqual(
            plan["layer_kinds"],
            [
                "sliding_attention_kv",
                "full_attention_kv",
                "sliding_attention_shared_kv",
                "full_attention_shared_kv",
            ],
        )
        self.assertEqual(plan["layer_kv_policy"], ["produce", "produce", "reuse", "reuse"])
        self.assertEqual(plan["layer_kv_source"], [0, 1, 0, 1])
        self.assertEqual(plan["layer_sliding_window"], [1024, 0, 1024, 0])
        self.assertEqual(plan["layer_rope_kind"], ["swa", "full", "swa", "full"])
        self.assertEqual(plan["layer_q_head_dim"], [128, 256, 128, 256])
        self.assertEqual(plan["layer_k_head_dim"], [128, 256, 128, 256])
        self.assertEqual(plan["layer_v_head_dim"], [128, 256, 128, 256])
        self.assertEqual(plan["layer_rotary_dim"], [256, 512, 256, 512])
        self.assertEqual(plan["layer_q_dim"], [2048, 4096, 2048, 4096])
        self.assertEqual(plan["layer_kv_dim"], [512, 1024, 512, 1024])

    def test_gemma4_template_declares_q_only_shared_kv_kinds(self) -> None:
        template_path = REPO_ROOT / "version" / "v8" / "templates" / "gemma4.json"
        import json

        template = json.loads(template_path.read_text(encoding="utf-8"))
        body = template["block_types"]["decoder"]["body"]
        self.assertEqual(body["kind_config_key"], "layer_kinds")
        self.assertIn("sliding_attention_shared_kv", body["ops_by_kind"])
        self.assertIn("full_attention_shared_kv", body["ops_by_kind"])
        shared_ops = body["ops_by_kind"]["sliding_attention_shared_kv"]
        self.assertIn("q_proj", shared_ops)
        self.assertIn("q_norm", shared_ops)
        self.assertIn("rope_q", shared_ops)
        self.assertIn("attn_sliding_shared_kv", shared_ops)
        self.assertNotIn("k_proj", shared_ops)
        self.assertNotIn("v_proj", shared_ops)

    def test_gemma4_assistant_q_only_ops_are_lowerable(self) -> None:
        import json
        import build_ir_v8  # type: ignore

        template_path = REPO_ROOT / "version" / "v8" / "templates" / "gemma4_assistant.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        ops = build_ir_v8._collect_template_ops(
            template,
            {
                "num_layers": 2,
                "layer_kinds": [
                    "sliding_attention_q_only_k_eq_v",
                    "full_attention_q_only_k_eq_v",
                ],
            },
        )
        self.assertEqual(build_ir_v8.validate_template_ops(ops), [])
        for op in (
            "assistant_pre_projection",
            "q_norm",
            "rope_q",
            "attn_sliding_shared_kv",
            "attn_shared_kv",
            "assistant_layer_scale",
            "assistant_post_projection",
        ):
            self.assertIn(op, ops)

        for kernel_id in (
            "assistant_layer_scale_forward",
            "q_norm_forward",
            "rope_forward_q_gemma4",
            "kv_cache_store_shared_q",
            "attention_forward_causal_head_major_shared_kv_gemma4",
            "attention_forward_causal_head_major_shared_kv_sliding_gemma4",
            "attention_forward_decode_head_major_shared_kv_gemma4",
            "attention_forward_decode_head_major_shared_kv_sliding_gemma4",
        ):
            self.assertTrue((REPO_ROOT / "version" / "v8" / "kernel_maps" / f"{kernel_id}.json").exists())

    def test_gemma4_kv_layers_use_supported_paired_qk_ops_for_first_bringup(self) -> None:
        template_path = REPO_ROOT / "version" / "v8" / "templates" / "gemma4.json"
        import json

        template = json.loads(template_path.read_text(encoding="utf-8"))
        body = template["block_types"]["decoder"]["body"]
        for kind in ("sliding_attention_kv", "full_attention_kv"):
            ops = body["ops_by_kind"][kind]
            self.assertIn("qk_norm", ops)
            self.assertIn("rope_qk", ops)
            self.assertNotIn("q_norm", ops)
            self.assertNotIn("rope_q", ops)

    def test_gemma4_v_norm_is_unweighted_rmsnorm(self) -> None:
        import json

        template_path = REPO_ROOT / "version" / "v8" / "templates" / "gemma4.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        body = template["block_types"]["decoder"]["body"]
        for kind in ("sliding_attention_kv", "full_attention_kv"):
            ops = body["ops_by_kind"][kind]
            self.assertIn("v_norm", ops)
            self.assertLess(ops.index("v_proj"), ops.index("v_norm"))
            self.assertLess(ops.index("v_norm"), ops.index("qk_norm"))

        overlay_path = REPO_ROOT / "version" / "v8" / "kernel_maps" / "kernel_bindings.overlay.json"
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        binding = overlay["bindings"]["rmsnorm_forward_no_weight"]
        self.assertNotIn("gamma", {param["name"] for param in binding["params"]})
        self.assertNotIn("v_norm_gamma", json.dumps(binding))

    def test_gemma4_template_runs_per_layer_embedding_after_ffn_residual(self) -> None:
        import json

        template_path = REPO_ROOT / "version" / "v8" / "templates" / "gemma4.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        body = template["block_types"]["decoder"]["body"]
        for ops in body["ops_by_kind"].values():
            self.assertIn("gemma4_per_layer_embed", ops)
            self.assertLess(ops.index("post_ffn_norm"), ops.index("gemma4_per_layer_embed"))

        kernel_map = json.loads(
            (REPO_ROOT / "version" / "v8" / "kernel_maps" / "gemma4_per_layer_embed_forward.json").read_text(encoding="utf-8")
        )
        self.assertEqual(kernel_map["op"], "gemma4_per_layer_embed")
        self.assertEqual(kernel_map["impl"]["function"], "gemma4_per_layer_embed_forward")
        prepare_map = json.loads(
            (REPO_ROOT / "version" / "v8" / "kernel_maps" / "gemma4_per_layer_prepare_forward.json").read_text(encoding="utf-8")
        )
        self.assertEqual(prepare_map["op"], "gemma4_per_layer_prepare")


if __name__ == "__main__":
    unittest.main()
