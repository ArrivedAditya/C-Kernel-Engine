import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str((REPO_ROOT / "version" / "v8" / "scripts").resolve()))

from convert_gguf_to_bump_v8 import (  # type: ignore
    GGUFError,
    build_qwen35_execution_plan,
    qwen35_decoder_layer_count,
    qwen35_mtp_source_reason,
    resolve_qwen35_recurrent_qkv_weight_dtype,
)
from build_ir_v8 import _hydrate_manifest_template  # type: ignore


class V8Qwen35PolicyTests(unittest.TestCase):
    def test_qwen36_gguf_block_count_excludes_mtp_from_runtime_layers(self) -> None:
        self.assertEqual(qwen35_decoder_layer_count(65, 1), 64)
        self.assertEqual(qwen35_decoder_layer_count(24, 0), 24)

    def test_qwen36_gguf_rejects_invalid_mtp_layer_count(self) -> None:
        with self.assertRaisesRegex(GGUFError, "smaller than block_count"):
            qwen35_decoder_layer_count(1, 1)

    def test_qwen36_gguf_mtp_tensors_are_scoped_after_decoder_layers(self) -> None:
        for suffix in ("eh_proj", "enorm", "hnorm", "shared_head_norm"):
            self.assertEqual(
                qwen35_mtp_source_reason(
                    f"blk.64.nextn.{suffix}.weight",
                    decoder_layers=64,
                    nextn_layers=1,
                ),
                "mtp_decoder_block_not_in_main_pass",
            )
        for suffix in (
            "attn_norm.weight",
            "post_attention_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ):
            self.assertEqual(
                qwen35_mtp_source_reason(
                    f"blk.64.{suffix}",
                    decoder_layers=64,
                    nextn_layers=1,
                ),
                "mtp_decoder_block_not_in_main_pass",
            )

    def test_qwen36_gguf_mtp_classifier_rejects_unknown_or_unscoped_tensors(self) -> None:
        for name in (
            "blk.63.nextn.eh_proj.weight",
            "blk.64.nextn.unknown.weight",
            "blk.65.nextn.eh_proj.weight",
            "blk.64.attn_q.bias",
        ):
            self.assertIsNone(
                qwen35_mtp_source_reason(
                    name,
                    decoder_layers=64,
                    nextn_layers=1,
                )
            )

    def test_recurrent_qkv_dtype_uses_normalized_layer_kinds(self) -> None:
        self.assertEqual(
            resolve_qwen35_recurrent_qkv_weight_dtype(
                ["recurrent", "full_attention", "recurrent"],
                {
                    "layer.0": {"attn_qkv": "q8_0"},
                    "layer.1": {"attn_q": "q8_0"},
                    "layer.2": {"attn_qkv": "q8_0"},
                },
            ),
            "q8_0",
        )

    def test_recurrent_qkv_dtype_accepts_uniform_q6_k(self) -> None:
        self.assertEqual(
            resolve_qwen35_recurrent_qkv_weight_dtype(
                ["recurrent", "full_attention", "recurrent"],
                {
                    "layer.0": {"attn_qkv": "q6_k"},
                    "layer.1": {"attn_q": "q4_k"},
                    "layer.2": {"attn_qkv": "q6_k"},
                },
            ),
            "q6_k",
        )

    def test_mixed_recurrent_qkv_dtypes_hard_fail(self) -> None:
        with self.assertRaisesRegex(GGUFError, "mixed dtypes"):
            resolve_qwen35_recurrent_qkv_weight_dtype(
                ["recurrent", "recurrent"],
                {
                    "layer.0": {"attn_qkv": "q5_k"},
                    "layer.1": {"attn_qkv": "q8_0"},
                },
            )

    def test_build_qwen35_execution_plan_makes_state_ownership_explicit(self) -> None:
        plan = build_qwen35_execution_plan(
            ["recurrent", "recurrent", "recurrent", "full_attention"]
        )
        self.assertEqual(
            plan["layer_state_policy"],
            ["recurrent_state", "recurrent_state", "recurrent_state", "kv_cache"],
        )
        self.assertEqual(plan["layer_attention_policy"], ["none", "none", "none", "full"])
        self.assertEqual(plan["layer_recurrent_policy"], ["deltanet", "deltanet", "deltanet", "none"])
        self.assertEqual(plan["layer_kv_policy"], ["none", "none", "none", "produce"])
        self.assertEqual(
            plan["layer_execution_plan"][3],
            {
                "layer": 3,
                "kind": "full_attention",
                "state_policy": "kv_cache",
                "attention_policy": "full",
                "recurrent_policy": "none",
                "kv_policy": "produce",
            },
        )

    def test_qwen35_template_declares_policy_config_keys(self) -> None:
        template_path = REPO_ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        contract = template["contract"]["attention_contract"]
        self.assertEqual(contract["layer_policy_config_key"], "layer_execution_plan")
        self.assertEqual(contract["state_policy_config_key"], "layer_state_policy")
        self.assertEqual(contract["attention_policy_config_key"], "layer_attention_policy")
        self.assertEqual(contract["recurrent_policy_config_key"], "layer_recurrent_policy")
        self.assertEqual(contract["kv_policy_config_key"], "layer_kv_policy")

    def test_qwen35_production_does_not_force_diagnostic_fp32_mlp_layers(self) -> None:
        template_path = REPO_ROOT / "version" / "v8" / "circuits" / "qwen35.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        defaults = template["contract"]["runtime_defaults"]
        self.assertNotIn("mlp_gate_up_fp32_layers", defaults)

    def test_qwen35_cache_storage_is_derived_from_attention_contract(self) -> None:
        manifest = {"config": {"model": "qwen35", "decode_kv_cache_dtype": "fp32"}}
        _hydrate_manifest_template(manifest)
        self.assertEqual(manifest["config"]["decode_kv_cache_dtype"], "fp16")


if __name__ == "__main__":
    unittest.main()
