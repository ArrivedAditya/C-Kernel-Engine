from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "convert_safetensors_to_bump_v8.py"


def _load_converter():
    scripts = str(SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("convert_instella_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


converter = _load_converter()


def _tiny_config() -> dict:
    return {
        "architectures": ["InstellaMoEForCausalLM"],
        "model_type": "deepseek_v3",
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "kv_lora_rank": 4,
        "q_lora_rank": None,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 2,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "routed_scaling_factor": 2.5,
        "gated_attention": True,
        "farskip": True,
        "rope_interleave": True,
        "rope_scaling": {"type": "yarn", "factor": 40},
        "rope_theta": 8000000,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
    }


def _headers() -> dict:
    fake = Path("/tmp/instella-fixture.safetensors")
    rows: dict[str, object] = {}

    def add(name: str, shape: list[int], dtype: str = "BF16") -> None:
        rows[name] = converter.HeaderTensor(name, dtype, shape, fake)

    add("model.embed_tokens.weight", [32, 8])
    add("model.norm.weight", [8], "F32")
    add("lm_head.weight", [32, 8])
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        add(f"{prefix}.input_layernorm.weight", [8], "F32")
        add(f"{prefix}.post_attention_layernorm.weight", [8], "F32")
        add(f"{prefix}.self_attn.q_proj.weight", [8, 8])
        add(f"{prefix}.self_attn.kv_a_proj_with_mqa.weight", [6, 8])
        add(f"{prefix}.self_attn.kv_a_layernorm.weight", [4], "F32")
        add(f"{prefix}.self_attn.kv_b_proj.weight", [8, 4])
        add(f"{prefix}.self_attn.gate_proj.weight", [4, 8])
        add(f"{prefix}.self_attn.o_proj.weight", [8, 4])
    add("model.layers.0.mlp.gate_proj.weight", [16, 8])
    add("model.layers.0.mlp.up_proj.weight", [16, 8])
    add("model.layers.0.mlp.down_proj.weight", [8, 16])
    add("model.layers.1.mlp.gate.weight", [2, 8], "F32")
    add("model.layers.1.mlp.gate.e_score_correction_bias", [2], "F32")
    for expert in range(2):
        prefix = f"model.layers.1.mlp.experts.{expert}"
        add(f"{prefix}.gate_proj.weight", [4, 8])
        add(f"{prefix}.up_proj.weight", [4, 8])
        add(f"{prefix}.down_proj.weight", [8, 4])
    add("model.layers.1.mlp.shared_experts.gate_proj.weight", [8, 8])
    add("model.layers.1.mlp.shared_experts.up_proj.weight", [8, 8])
    add("model.layers.1.mlp.shared_experts.down_proj.weight", [8, 8])
    return rows


class InstellaMoEBringupTests(unittest.TestCase):
    def test_architecture_class_overrides_deepseek_model_type(self) -> None:
        self.assertEqual(converter._infer_arch(_tiny_config()), "instella_moe")

    def test_tensor_role_mapping_consumes_every_fixture_header(self) -> None:
        headers = _headers()
        refs = converter._refs_for_arch("instella_moe", _tiny_config(), headers)
        consumed = {source for ref in refs for source in ref.source_names}
        self.assertEqual(consumed, set(headers))
        by_name = {ref.ck_name: ref for ref in refs}
        self.assertIn("layer.0.mla_gate_proj", by_name)
        self.assertIn("layer.1.mla_gate_proj", by_name)
        self.assertEqual(by_name["layer.1.moe_expert_gate"].shape, (2, 4, 8))
        self.assertEqual(by_name["layer.1.moe_shared_gate"].source_names, (
            "model.layers.1.mlp.shared_experts.gate_proj.weight",
        ))

    def test_runtime_config_preserves_farskip_and_interleaved_yarn(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text(
                json.dumps(_tiny_config()), encoding="utf-8"
            )
            config = converter._build_config(root, "instella_moe", None)
        self.assertEqual(config["model"], "instella_moe")
        self.assertEqual(config["layer_kinds"], ["mla_dense_mlp", "mla_moe"])
        self.assertTrue(config["gated_attention"])
        self.assertTrue(config["farskip"])
        self.assertTrue(config["rope_interleave"])
        self.assertEqual(config["rope_layout"], "partial_interleaved_yarn")
        self.assertEqual(config["moe_shared_intermediate_size"], 8)


if __name__ == "__main__":
    unittest.main()
