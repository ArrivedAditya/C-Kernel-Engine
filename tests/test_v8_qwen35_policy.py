import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str((REPO_ROOT / "version" / "v8" / "scripts").resolve()))

from convert_gguf_to_bump_v8 import build_qwen35_execution_plan  # type: ignore
from build_ir_v8 import _hydrate_manifest_template  # type: ignore


class V8Qwen35PolicyTests(unittest.TestCase):
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
