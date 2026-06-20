from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "version/v8/scripts/inspect_model_contract_v8.py"
AUDIT_SCRIPT = REPO / "version/v8/scripts/audit_safetensors_index_v8.py"


class ModelContractInspectorTests(unittest.TestCase):
    def test_nemotron_h_reports_mamba_kernel_gap(self) -> None:
        cfg = {
            "architectures": ["NemotronHForCausalLM"],
            "model_type": "nemotron_h",
            "hidden_size": 4096,
            "intermediate_size": 21504,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 10,
            "hybrid_override_pattern": "MEMEM*EMEM",
            "mamba_num_heads": 128,
            "mamba_head_dim": 64,
            "ssm_state_size": 128,
            "conv_kernel": 4,
            "mlp_hidden_act": "relu2",
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.assertEqual(proc.returncode, 2, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["arch"], "nemotron_h")
        self.assertEqual(report["status"], "bringup_required")
        self.assertEqual(report["layer_kind_counts"], {"attention": 1, "mamba": 5, "moe": 4})
        self.assertIn("mamba_selective_scan", report["missing_ops"])
        self.assertNotIn("mamba_in_proj_split", report["missing_ops"])
        self.assertNotIn("mamba_conv1d_state_update", report["missing_ops"])
        self.assertNotIn("mamba_dt_softplus", report["missing_ops"])
        self.assertNotIn("mamba_rmsnorm_gate", report["missing_ops"])
        self.assertNotIn("mamba_out_proj", report["missing_ops"])
        self.assertNotIn("relu2_mlp", report["missing_ops"])
        self.assertNotIn("shared_expert_mlp", report["missing_ops"])
        self.assertNotIn("group_limited_topk_router", report["missing_ops"])
        self.assertNotIn("moe_relu2_expert_mlp", report["missing_ops"])
        self.assertNotIn("safetensors_to_bump_mapping", report["missing_ops"])
        self.assertNotIn("v8_template_contract", report["missing_ops"])

    def test_safetensors_index_audit_classifies_nemotron_families(self) -> None:
        index = {
            "metadata": {"total_parameters": 123, "total_size": 456},
            "weight_map": {
                "backbone.embeddings.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.norm.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.in_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.conv1d.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.out_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.1.norm.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.1.mixer.q_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.1.mixer.k_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.1.mixer.v_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.1.mixer.o_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.2.norm.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.2.mixer.gate.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.2.mixer.gate.e_score_correction_bias": "model-00001-of-00001.safetensors",
                "backbone.layers.2.mixer.experts.0.up_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.2.mixer.experts.0.down_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.2.mixer.shared_experts.up_proj.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.2.mixer.shared_experts.down_proj.weight": "model-00001-of-00001.safetensors",
                "lm_head.weight": "model-00001-of-00001.safetensors",
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.safetensors.index.json"
            path.write_text(json.dumps(index), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(AUDIT_SCRIPT), str(path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
        report = json.loads(proc.stdout)
        self.assertEqual(report["families"]["mamba"], 3)
        self.assertEqual(report["families"]["attention"], 4)
        self.assertEqual(report["families"]["moe_router"], 2)
        self.assertEqual(report["families"]["moe_expert"], 2)
        self.assertEqual(report["families"]["moe_shared_expert"], 2)
        self.assertEqual(report["layers"]["2"]["expert_count"], 1)

    def test_qwen3_dense_config_is_supported_at_contract_level(self) -> None:
        cfg = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_hidden_layers": 4,
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
        report = json.loads(proc.stdout)
        self.assertEqual(report["arch"], "qwen3")
        self.assertEqual(report["status"], "supported")
        self.assertEqual(report["layer_kind_counts"], {"attention": 4})
        self.assertEqual(report["missing_ops"], [])


if __name__ == "__main__":
    unittest.main()
