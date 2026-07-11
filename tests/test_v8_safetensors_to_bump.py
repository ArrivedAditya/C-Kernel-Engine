from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _require_torch_safetensors() -> tuple[object, object]:
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    return torch, st




def _write_tiny_bpe_tokenizer(checkpoint: Path, vocab_size: int) -> None:
    vocab: dict[str, int] = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "Hello": 3,
        "world": 4,
        "!": 5,
        "Ġtest": 6,
        "Ġcode": 7,
        "Helloworld": 8,
        "ĠtestĠcode": 9,
    }
    for idx in range(len(vocab), vocab_size):
        vocab[f"<tok_{idx}>"] = idx

    (checkpoint / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "unk_token": "<unk>",
                    "vocab": vocab,
                    "merges": ["Hello world", "Ġtest Ġcode"],
                },
                "added_tokens": [
                    {"id": 0, "content": "<unk>"},
                    {"id": 1, "content": "<s>"},
                    {"id": 2, "content": "</s>"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (checkpoint / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": {"content": "<s>"},
                "eos_token": {"content": "</s>"},
                "unk_token": {"content": "<unk>"},
                "add_bos_token": True,
                "add_eos_token": False,
                "added_tokens_decoder": {
                    "0": {"content": "<unk>"},
                    "1": {"content": "<s>"},
                    "2": {"content": "</s>"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

def test_qwen3_safetensors_to_bump_smoke(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "qwen3"
    out = tmp_path / "out"
    checkpoint.mkdir()
    out.mkdir()

    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_hidden_layers": 1,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
                "max_position_embeddings": 64,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tiny_bpe_tokenizer(checkpoint, vocab_size=32)

    tensors = {
        "model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.float32),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(8, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_norm.weight": torch.ones(4, dtype=torch.float32),
        "model.layers.0.self_attn.k_norm.weight": torch.ones(4, dtype=torch.float32),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(8, dtype=torch.float32),
        "lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
    }
    st.save_file(tensors, checkpoint / "model.safetensors")

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    names = [entry["name"] for entry in manifest["entries"]]
    assert manifest["model"] == "qwen3"
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert names[0] == "token_emb"
    assert "layer.0.q_norm" in names
    assert "layer.0.k_norm" in names
    assert "output.weight" in names
    assert {"vocab_offsets", "vocab_strings", "vocab_merges"}.issubset(set(names))
    assert names[-3:] == ["vocab_offsets", "vocab_strings", "vocab_merges"]
    entries = {entry["name"]: entry for entry in manifest["entries"]}
    assert entries["vocab_offsets"]["dtype"] == "i32"
    assert entries["vocab_strings"]["dtype"] == "u8"
    assert entries["vocab_merges"]["dtype"] == "i32"
    assert entries["vocab_offsets"]["shape"] == [32]
    assert entries["vocab_strings"]["size"] > 0
    assert entries["vocab_merges"]["shape"] == [6]
    assert entries["vocab_merges"]["size"] == 24
    assert manifest["tokenizer_contract"]["tokenizer_type"] == "bpe"
    assert manifest["config"]["tokenizer_contract"]["tokenizer_type"] == "bpe"
    assert manifest["special_tokens"]["bos_token"] == "<s>"
    assert manifest["special_tokens"]["bos_token_id"] == 1
    assert manifest["special_tokens"]["eos_token"] == "</s>"
    assert manifest["special_tokens"]["eos_token_id"] == 2
    assert manifest["special_tokens"]["unk_token"] == "<unk>"
    assert manifest["special_tokens"]["unk_token_id"] == 0
    assert manifest["template"]["flags"]["tokenizer"] == "bpe"
    assert manifest["template"]["contract"]["tokenizer_contract"]["tokenizer_type"] == "bpe"
    assert (out / "weights.bump").stat().st_size > 0


def test_qwen35_safetensors_to_bump_smoke(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "qwen35"
    out = tmp_path / "out_qwen35"
    checkpoint.mkdir()
    out.mkdir()

    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "tie_word_embeddings": True,
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "num_hidden_layers": 4,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "vocab_size": 32,
                    "max_position_embeddings": 64,
                    "full_attention_interval": 4,
                    "layer_types": layer_types,
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 4,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 2,
                    "linear_value_head_dim": 4,
                    "rms_norm_eps": 1e-6,
                    "tie_word_embeddings": True,
                    "rope_parameters": {
                        "rope_theta": 10000000.0,
                        "partial_rotary_factor": 0.25,
                        "mrope_section": [1, 1, 0],
                        "rope_type": "default",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    tensors = {
        "model.language_model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "model.language_model.norm.weight": torch.ones(8, dtype=torch.float32),
    }
    for layer, kind in enumerate(layer_types):
        prefix = f"model.language_model.layers.{layer}"
        tensors[f"{prefix}.input_layernorm.weight"] = torch.ones(8, dtype=torch.float32)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(8, dtype=torch.float32)
        tensors[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(16, 8, dtype=torch.bfloat16)
        tensors[f"{prefix}.mlp.up_proj.weight"] = torch.randn(16, 8, dtype=torch.bfloat16)
        tensors[f"{prefix}.mlp.down_proj.weight"] = torch.randn(8, 16, dtype=torch.bfloat16)
        if kind == "linear_attention":
            tensors[f"{prefix}.linear_attn.in_proj_qkv.weight"] = torch.randn(24, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.linear_attn.in_proj_z.weight"] = torch.randn(8, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.linear_attn.in_proj_a.weight"] = torch.randn(8, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.linear_attn.in_proj_b.weight"] = torch.randn(8, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.linear_attn.conv1d.weight"] = torch.randn(24, 1, 4, dtype=torch.float32)
            tensors[f"{prefix}.linear_attn.dt_bias"] = torch.randn(8, dtype=torch.float32)
            tensors[f"{prefix}.linear_attn.A_log"] = torch.randn(8, 4, dtype=torch.float32)
            tensors[f"{prefix}.linear_attn.norm.weight"] = torch.ones(8, dtype=torch.float32)
            tensors[f"{prefix}.linear_attn.out_proj.weight"] = torch.randn(8, 8, dtype=torch.bfloat16)
        else:
            tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(16, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(4, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(4, 8, dtype=torch.bfloat16)
            tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(8, 16, dtype=torch.bfloat16)
            tensors[f"{prefix}.self_attn.q_norm.weight"] = torch.ones(4, dtype=torch.float32)
            tensors[f"{prefix}.self_attn.k_norm.weight"] = torch.ones(4, dtype=torch.float32)

    st.save_file(tensors, checkpoint / "model.safetensors")

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    names = [entry["name"] for entry in manifest["entries"]]
    assert manifest["model"] == "qwen35"
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert any(row["target"] == "layer.0.ssm_a" and row["transform"] == "neg_exp_a_log" for row in audit["transforms"])
    assert any(row["target"] == "layer.0.attn_norm" and row["transform"] == "qwen35_norm_plus_one" for row in audit["transforms"])
    assert any(row["target"] == "layer.3.attn_q_norm" and row["transform"] == "qwen35_norm_plus_one" for row in audit["transforms"])
    assert any(row["target"] == "final_ln_weight" and row["transform"] == "qwen35_norm_plus_one" for row in audit["transforms"])
    assert not any(row["target"] == "layer.0.ssm_norm" and row.get("transform") == "qwen35_norm_plus_one" for row in audit["transforms"])
    assert config["layer_kinds"] == ["recurrent", "recurrent", "recurrent", "full_attention"]
    assert config["layer_recurrent_policy"] == ["deltanet", "deltanet", "deltanet", "none"]
    assert config["attn_q_gate_proj_dim"] == 16
    assert config["attn_out_dim"] == 8
    assert config["q_dim"] == 8
    assert config["k_dim"] == 8
    assert config["v_dim"] == 8
    assert config["gate_dim"] == 8
    dtypes = {entry["name"]: entry["dtype"] for entry in manifest["entries"]}
    assert dtypes["layer.0.ssm_conv1d"] == "fp32"
    assert "layer.0.attn_qkv" in names
    assert "layer.0.ssm_alpha" in names
    assert "layer.0.ssm_beta" in names
    assert "layer.0.ssm_conv1d" in names
    assert "layer.3.attn_q_gate" in names
    assert "layer.3.attn_q_norm" in names
    assert "final_ln_weight" in names
    assert "output.weight" not in names
    assert (out / "weights.bump").stat().st_size > 0



def test_nemotron_h_safetensors_to_bump_dry_run_maps_hybrid_mamba_attention_moe(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "nemotron_h"
    out = tmp_path / "out_nemotron_h"
    checkpoint.mkdir()
    out.mkdir()

    config = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "num_hidden_layers": 4,
        "hidden_size": 8,
        "intermediate_size": 6,
        "moe_intermediate_size": 6,
        "moe_shared_expert_intermediate_size": 12,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "hybrid_override_pattern": "M*E-",
        "mamba_num_heads": 2,
        "mamba_head_dim": 4,
        "ssm_state_size": 3,
        "conv_kernel": 4,
        "n_groups": 2,
        "chunk_size": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "mlp_hidden_act": "relu2",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "rope_theta": 10000.0,
        "layer_norm_epsilon": 1e-5,
    }
    (checkpoint / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")

    tensors = {
        "backbone.embeddings.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "backbone.norm_f.weight": torch.ones(8, dtype=torch.float32),
        "lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
    }
    # Layer 0: Mamba2
    tensors.update({
        "backbone.layers.0.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.0.mixer.in_proj.weight": torch.randn(34, 8, dtype=torch.bfloat16),
        "backbone.layers.0.mixer.conv1d.weight": torch.randn(20, 1, 4, dtype=torch.float32),
        "backbone.layers.0.mixer.conv1d.bias": torch.randn(20, dtype=torch.float32),
        "backbone.layers.0.mixer.dt_bias": torch.randn(2, dtype=torch.float32),
        "backbone.layers.0.mixer.A_log": torch.randn(2, dtype=torch.float32),
        "backbone.layers.0.mixer.D": torch.randn(2, dtype=torch.float32),
        "backbone.layers.0.mixer.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.0.mixer.out_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
    })
    # Layer 1: attention
    tensors.update({
        "backbone.layers.1.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.1.mixer.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "backbone.layers.1.mixer.k_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "backbone.layers.1.mixer.v_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "backbone.layers.1.mixer.o_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
    })
    # Layer 2: MoE
    tensors.update({
        "backbone.layers.2.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.2.mixer.gate.weight": torch.randn(4, 8, dtype=torch.float32),
        "backbone.layers.2.mixer.gate.e_score_correction_bias": torch.randn(4, dtype=torch.float32),
        "backbone.layers.2.mixer.shared_experts.up_proj.weight": torch.randn(12, 8, dtype=torch.bfloat16),
        "backbone.layers.2.mixer.shared_experts.down_proj.weight": torch.randn(8, 12, dtype=torch.bfloat16),
    })
    for expert in range(4):
        tensors[f"backbone.layers.2.mixer.experts.{expert}.up_proj.weight"] = torch.randn(6, 8, dtype=torch.bfloat16)
        tensors[f"backbone.layers.2.mixer.experts.{expert}.down_proj.weight"] = torch.randn(8, 6, dtype=torch.bfloat16)
    # Layer 3: dense ReLU2 MLP
    tensors.update({
        "backbone.layers.3.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.3.mixer.up_proj.weight": torch.randn(6, 8, dtype=torch.bfloat16),
        "backbone.layers.3.mixer.down_proj.weight": torch.randn(8, 6, dtype=torch.bfloat16),
    })

    st.save_file(tensors, checkpoint / "model.safetensors")
    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    names = [entry["name"] for entry in manifest["entries"]]
    assert manifest["model"] == "nemotron_h"
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert cfg["layer_kinds"] == ["mamba", "attention", "moe", "mlp"]
    assert cfg["layer_state_policy"] == ["mamba2", "none", "none", "none"]
    assert cfg["ssm_conv_kernel"] == 4
    assert cfg["ssm_conv_history"] == 4
    assert cfg["layer_moe_policy"] == ["none", "none", "routed_relu2", "none"]
    assert "layer.0.mamba_in_proj" in names
    assert "layer.0.mamba_conv1d" in names
    assert "layer.1.attn_q" in names
    assert "layer.2.moe_router" in names
    assert "layer.2.moe_router_bias" in names
    assert "layer.2.moe_expert.3.up" in names
    assert "layer.2.moe_shared_up" in names
    assert "layer.3.mlp_up" in names
    assert "output.weight" in names
    assert any(row["target"] == "layer.0.mamba_a" and row["transform"] == "neg_exp_a_log" for row in audit["transforms"])


def test_nemotron_h_safetensors_to_bump_dry_run_maps_dense_mamba_attention_relu2_without_moe(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "nemotron_h_dense"
    out = tmp_path / "out_nemotron_h_dense"
    checkpoint.mkdir()
    out.mkdir()

    config = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "num_hidden_layers": 3,
        "hidden_size": 8,
        "intermediate_size": 6,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "hybrid_override_pattern": "M*-",
        "mamba_num_heads": 2,
        "mamba_head_dim": 4,
        "ssm_state_size": 3,
        "conv_kernel": 4,
        "n_groups": 2,
        "chunk_size": 8,
        "mlp_hidden_act": "relu2",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "rope_theta": 10000.0,
        "layer_norm_epsilon": 1e-5,
    }
    (checkpoint / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")

    tensors = {
        "backbone.embeddings.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "backbone.norm_f.weight": torch.ones(8, dtype=torch.float32),
        "lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "backbone.layers.0.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.0.mixer.in_proj.weight": torch.randn(34, 8, dtype=torch.bfloat16),
        "backbone.layers.0.mixer.conv1d.weight": torch.randn(20, 1, 4, dtype=torch.float32),
        "backbone.layers.0.mixer.conv1d.bias": torch.randn(20, dtype=torch.float32),
        "backbone.layers.0.mixer.dt_bias": torch.randn(2, dtype=torch.float32),
        "backbone.layers.0.mixer.A_log": torch.randn(2, dtype=torch.float32),
        "backbone.layers.0.mixer.D": torch.randn(2, dtype=torch.float32),
        "backbone.layers.0.mixer.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.0.mixer.out_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "backbone.layers.1.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.1.mixer.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "backbone.layers.1.mixer.k_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "backbone.layers.1.mixer.v_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "backbone.layers.1.mixer.o_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "backbone.layers.2.norm.weight": torch.ones(8, dtype=torch.float32),
        "backbone.layers.2.mixer.up_proj.weight": torch.randn(6, 8, dtype=torch.bfloat16),
        "backbone.layers.2.mixer.down_proj.weight": torch.randn(8, 6, dtype=torch.bfloat16),
    }

    st.save_file(tensors, checkpoint / "model.safetensors")
    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    names = [entry["name"] for entry in manifest["entries"]]
    assert manifest["model"] == "nemotron_h"
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert cfg["layer_kinds"] == ["mamba", "attention", "mlp"]
    assert cfg["layer_state_policy"] == ["mamba2", "none", "none"]
    assert cfg["layer_moe_policy"] == ["none", "none", "none"]
    assert cfg["layer_mlp_policy"] == ["none", "none", "relu2"]
    attention_ops = manifest["template"]["block_types"]["decoder"]["body"]["ops_by_kind"]["attention"]
    assert "rope_qk" not in attention_ops
    assert cfg["ssm_conv_kernel"] == 4
    assert cfg["ssm_conv_history"] == 4
    assert cfg["recurrent_num_heads"] == 2
    assert cfg["recurrent_head_dim"] == 4
    assert cfg["recurrent_state_heads"] == 2
    assert cfg["recurrent_state_rows"] == 4
    assert cfg["recurrent_state_cols"] == 3
    assert manifest["config"]["recurrent_state_heads"] == 2
    assert manifest["config"]["recurrent_state_rows"] == 4
    assert manifest["config"]["recurrent_state_cols"] == 3
    assert "layer.0.mamba_in_proj" in names
    assert "layer.1.attn_q" in names
    assert "layer.2.mlp_up" in names
    assert "layer.2.mlp_down" in names
    assert "output.weight" in names
    assert not any(".moe_" in name or name.endswith("moe_router") for name in names)
    assert any(row["target"] == "layer.0.mamba_a" and row["transform"] == "neg_exp_a_log" for row in audit["transforms"])

    build_ir = Path("version/v8/scripts/build_ir_v8.py")
    lowered = out / "lowered_decode.json"
    subprocess.run(
        [
            sys.executable,
            str(build_ir),
            "--manifest",
            str(out / "weights_manifest.json"),
            "--mode",
            "decode",
            "--output",
            str(out / "ir1_decode.json"),
            "--layout-output",
            str(out / "layout_decode.json"),
            "--lowered-output",
            str(lowered),
            "--call-output",
            str(out / "lowered_decode_call.json"),
            "--context-len",
            "8",
        ],
        check=True,
    )
    layout = json.loads((out / "layout_decode.json").read_text(encoding="utf-8"))
    conv_state = next(
        buf for buf in layout["memory"]["activations"]["buffers"] if buf["name"] == "recurrent_conv_state"
    )
    ssm_state = next(
        buf for buf in layout["memory"]["activations"]["buffers"] if buf["name"] == "recurrent_ssm_state"
    )
    assert conv_state["shape"] == "[3, 4, 20]"
    assert ssm_state["shape"] == "[3, 2, 4, 3]"

    ir1_ops = json.loads((out / "ir1_decode.json").read_text(encoding="utf-8"))["ops"]
    ir1_by_op = {(op["layer"], op["op"]): op for op in ir1_ops if "layer" in op}
    ir1_mamba_in = ir1_by_op[(0, "mamba_in_proj")]
    assert ir1_mamba_in["dataflow"]["inputs"]["x"]["slot"] == "layer_input"
    assert ir1_mamba_in["dataflow"]["inputs"]["x"]["from_op"] == ir1_by_op[(0, "block_rmsnorm")]["op_id"]

    lowered_ops = json.loads(lowered.read_text(encoding="utf-8"))["operations"]
    by_op = {(op["layer"], op["op"]): op for op in lowered_ops}

    attention_layer_ops = [op["op"] for op in lowered_ops if op.get("layer") == 1]
    assert "rope_qk" not in attention_layer_ops
    assert "kv_cache_store" in attention_layer_ops
    assert attention_layer_ops.index("v_proj") < attention_layer_ops.index("kv_cache_store") < attention_layer_ops.index("attn")
    attn = by_op[(1, "attn")]
    assert attn["kernel"] == "attention_forward_decode_head_major_gqa_flash"
    assert attn["_kv_cache_read_layer"] == 1
    attn_inputs = attn.get("inputs") or attn.get("activations") or {}
    assert attn_inputs["k_cache"]["buffer"] == "kv_cache"
    assert attn_inputs["v_cache"]["buffer"] == "kv_cache"

    mamba_in = by_op[(0, "mamba_in_proj")]
    assert mamba_in["kernel"] == "gemv_bf16"
    assert mamba_in["activations"]["x"]["buffer"] == "layer_input"

    mlp_up = by_op[(2, "mlp_up")]
    relu2 = by_op[(2, "relu2")]
    mlp_down = by_op[(2, "mlp_down")]
    assert mlp_up["kernel"] == "gemv_bf16"
    assert mlp_up["activations"]["x"]["buffer"] == "layer_input"
    assert relu2["activations"]["x"]["buffer"] == "mlp_scratch"
    assert relu2["outputs"]["out"]["buffer"] == "mlp_scratch"
    assert mlp_down["activations"]["x"]["buffer"] == "mlp_scratch"


def test_glm4_safetensors_to_bump_uses_declarative_source_map(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "glm4"
    out = tmp_path / "out_glm4"
    checkpoint.mkdir()
    out.mkdir()

    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4ForCausalLM"],
                "model_type": "glm4",
                "num_hidden_layers": 1,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "partial_rotary_factor": 0.5,
                "vocab_size": 32,
                "max_position_embeddings": 64,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-5,
                "attention_bias": True,
                "tie_word_embeddings": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tiny_bpe_tokenizer(checkpoint, vocab_size=32)

    tensors = {
        "model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.float32),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(8, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.bias": torch.randn(8, dtype=torch.float32),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.k_proj.bias": torch.randn(4, dtype=torch.float32),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.v_proj.bias": torch.randn(4, dtype=torch.float32),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.o_proj.bias": torch.randn(8, dtype=torch.float32),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(8, dtype=torch.float32),
        "lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
    }
    st.save_file(tensors, checkpoint / "model.safetensors")

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    names = [entry["name"] for entry in manifest["entries"]]
    entries = {entry["name"]: entry for entry in manifest["entries"]}

    assert manifest["model"] == "glm4"
    assert manifest["template"]["name"] == "glm4"
    assert manifest["template"]["contract"]["chat_contract"]["force_bos_text_if_tokenizer_add_bos_false"] == "[gMASK]"
    assert manifest["has_attention_biases"] is True
    assert manifest["has_qk_norm"] is False
    assert manifest["config"]["rotary_dim"] == 2
    assert manifest["config"]["partial_rotary_factor"] == 0.5
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert "layer.0.bq" in names and "layer.0.bo" in names
    assert entries["layer.0.w1"]["source_name"].endswith("gate_proj.weight+model.layers.0.mlp.up_proj.weight")
    assert entries["layer.0.w1"]["shape"] == [16, 8]
    assert entries["layer.0.b1"]["shape"] == [32]
    assert "output.weight" in names


def test_gemma4_assistant_safetensors_to_bump_maps_q_only_drafter(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "gemma4_assistant"
    out = tmp_path / "out_gemma4_assistant"
    checkpoint.mkdir()
    out.mkdir()

    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4AssistantForCausalLM"],
                "model_type": "gemma4_assistant",
                "backbone_hidden_size": 16,
                "tie_word_embeddings": True,
                "use_ordered_embeddings": False,
                "text_config": {
                    "model_type": "gemma4_text",
                    "attention_bias": False,
                    "attention_k_eq_v": True,
                    "bos_token_id": 2,
                    "eos_token_id": 1,
                    "global_head_dim": 8,
                    "head_dim": 4,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "max_position_embeddings": 128,
                    "num_attention_heads": 2,
                    "num_global_key_value_heads": 1,
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 2,
                    "num_kv_shared_layers": 2,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                        "full_attention": {
                            "partial_rotary_factor": 0.25,
                            "rope_theta": 1000000.0,
                            "rope_type": "proportional",
                        },
                        "sliding_attention": {
                            "rope_theta": 10000.0,
                            "rope_type": "default",
                        },
                    },
                    "sliding_window": 32,
                    "tie_word_embeddings": True,
                    "vocab_size": 32,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tiny_bpe_tokenizer(checkpoint, vocab_size=32)

    tensors = {
        "model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
        "pre_projection.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "post_projection.weight": torch.randn(16, 8, dtype=torch.bfloat16),
    }
    q_dims = [8, 16]
    for layer, q_dim in enumerate(q_dims):
        pfx = f"model.layers.{layer}"
        tensors.update(
            {
                f"{pfx}.input_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                f"{pfx}.pre_feedforward_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                f"{pfx}.post_attention_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                f"{pfx}.post_feedforward_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                f"{pfx}.layer_scalar": torch.ones(1, dtype=torch.bfloat16),
                f"{pfx}.self_attn.q_proj.weight": torch.randn(q_dim, 8, dtype=torch.bfloat16),
                f"{pfx}.self_attn.q_norm.weight": torch.ones(q_dim // 2, dtype=torch.bfloat16),
                f"{pfx}.self_attn.o_proj.weight": torch.randn(8, q_dim, dtype=torch.bfloat16),
                f"{pfx}.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
                f"{pfx}.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
                f"{pfx}.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
            }
        )
    st.save_file(tensors, checkpoint / "model.safetensors")

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    names = {entry["name"] for entry in manifest["entries"]}

    assert manifest["model"] == "gemma4_assistant"
    assert manifest["template"]["name"] == "gemma4_assistant"
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert cfg["attention_k_eq_v"] is True
    assert cfg["assistant_role"] == "mtp_drafter"
    assert cfg["assistant_projection_mode"] == "mtp_bridge"
    assert cfg["assistant_layer_scalar_mode"] == "layer_output_scale"
    assert cfg["standalone_text_inference_supported"] is False
    assert cfg["layer_kinds"] == ["sliding_attention_q_only_k_eq_v", "full_attention_q_only_k_eq_v"]
    assert cfg["layer_q_dim"] == [8, 16]
    assert cfg["layer_q_norm_dim"] == [4, 8]
    assert cfg["layer_q_head_dim"] == [4, 8]
    assert cfg["layer_k_head_dim"] == [4, 8]
    assert cfg["layer_v_head_dim"] == [4, 8]
    assert cfg["layer_rotary_dim"] == [4, 8]
    assert "assistant.pre_projection" in names
    assert "assistant.post_projection" in names
    assert "layer.0.wq" in names
    assert "layer.0.q_norm" in names
    assert "layer.0.wk" not in names
    assert "layer.0.wv" not in names


def test_kimi_vl_safetensors_to_bump_dry_run_maps_text_decoder(tmp_path: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint = tmp_path / "kimi_vl"
    out = tmp_path / "out_kimi_vl"
    checkpoint.mkdir()
    out.mkdir()

    config = {
        "architectures": ["KimiVLForConditionalGeneration"],
        "model_type": "kimi_vl",
        "text_config": {
            "num_hidden_layers": 2,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "kv_lora_rank": 4,
            "q_lora_rank": None,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.446,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "rope_theta": 800000.0,
            "tie_word_embeddings": False,
        },
        "vision_config": {
            "model_type": "moonvit",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 14,
        },
    }
    (checkpoint / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")

    tensors = {
        "language_model.model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "language_model.model.norm.weight": torch.ones(8, dtype=torch.float32),
        "language_model.lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
    }
    for layer in range(2):
        pfx = f"language_model.model.layers.{layer}"
        tensors.update(
            {
                f"{pfx}.input_layernorm.weight": torch.ones(8, dtype=torch.float32),
                f"{pfx}.post_attention_layernorm.weight": torch.ones(8, dtype=torch.float32),
                f"{pfx}.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
                f"{pfx}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(6, 8, dtype=torch.bfloat16),
                f"{pfx}.self_attn.kv_a_layernorm.weight": torch.ones(4, dtype=torch.float32),
                f"{pfx}.self_attn.kv_b_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
                f"{pfx}.self_attn.o_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
            }
        )
    tensors.update(
        {
            "language_model.model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
            "language_model.model.layers.0.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
            "language_model.model.layers.0.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
            "language_model.model.layers.1.mlp.gate.weight": torch.randn(2, 8, dtype=torch.float32),
            "language_model.model.layers.1.mlp.gate.e_score_correction_bias": torch.randn(2, dtype=torch.float32),
            "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
            "language_model.model.layers.1.mlp.shared_experts.up_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
            "language_model.model.layers.1.mlp.shared_experts.down_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        }
    )
    for expert in range(2):
        tensors.update(
            {
                f"language_model.model.layers.1.mlp.experts.{expert}.gate_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
                f"language_model.model.layers.1.mlp.experts.{expert}.up_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
                f"language_model.model.layers.1.mlp.experts.{expert}.down_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
            }
        )
    st.save_file(tensors, checkpoint / "model.safetensors")

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    names = {entry["name"] for entry in manifest["entries"]}
    entries = {entry["name"]: entry for entry in manifest["entries"]}

    assert manifest["model"] == "kimi_vl"
    assert manifest["template"]["name"] == "kimi_vl"
    assert manifest["config"]["layer_kinds"] == ["mla_dense_mlp", "mla_moe"]
    assert manifest["config"]["rotary_dim"] == 2
    assert manifest["config"]["mla_q_head_dim"] == 4
    assert manifest["config"]["layer_moe_policy"] == ["none", "routed_swiglu"]
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert entries["layer.1.moe_expert_gate"]["shape"] == [2, 4, 8]
    assert entries["layer.1.moe_expert_up"]["shape"] == [2, 4, 8]
    assert entries["layer.1.moe_expert_down"]["shape"] == [2, 8, 4]
    assert {
        "token_emb",
        "layer.0.mla_q_proj",
        "layer.0.mla_kv_a_proj",
        "layer.0.mla_kv_a_norm",
        "layer.0.mla_kv_b_proj",
        "layer.0.mlp_gate",
        "layer.1.moe_router",
        "layer.1.moe_router_bias",
        "layer.1.moe_expert_gate",
        "layer.1.moe_expert_up",
        "layer.1.moe_expert_down",
        "layer.1.moe_shared_gate",
        "final_ln_weight",
        "final_ln_bias",
        "output.weight",
    }.issubset(names)



def _write_tiny_qwen3vl_checkpoint(checkpoint: Path) -> None:
    torch, st = _require_torch_safetensors()
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
                "image_token_id": 151655,
                "vision_start_token_id": 151652,
                "vision_end_token_id": 151653,
                "tie_word_embeddings": False,
                "text_config": {
                    "model_type": "qwen3_vl_text",
                    "num_hidden_layers": 1,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "vocab_size": 32,
                    "max_position_embeddings": 64,
                    "rope_theta": 5000000.0,
                    "rms_norm_eps": 1e-6,
                    "rope_scaling": {
                        "mrope_interleaved": True,
                        "mrope_section": [1, 1, 2],
                        "rope_type": "default",
                    },
                },
                "vision_config": {
                    "model_type": "qwen3_vl",
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 12,
                    "num_heads": 2,
                    "out_hidden_size": 8,
                    "patch_size": 2,
                    "temporal_patch_size": 2,
                    "spatial_merge_size": 2,
                    "num_position_embeddings": 4,
                    "deepstack_visual_indexes": [0],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (checkpoint / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_mean": [0.1, 0.2, 0.3],
                "image_std": [0.4, 0.5, 0.6],
                "min_pixels": 16,
                "max_pixels": 4096,
                "size": {"shortest_edge": 4},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tiny_bpe_tokenizer(checkpoint, vocab_size=32)

    tensors = {
        "model.language_model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
        "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "model.language_model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        "model.visual.patch_embed.proj.weight": torch.randn(8, 3, 2, 2, 2, dtype=torch.bfloat16),
        "model.visual.patch_embed.proj.bias": torch.randn(8, dtype=torch.bfloat16),
        "model.visual.pos_embed.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.visual.blocks.0.norm1.weight": torch.ones(8, dtype=torch.bfloat16),
        "model.visual.blocks.0.norm1.bias": torch.zeros(8, dtype=torch.bfloat16),
        "model.visual.blocks.0.norm2.weight": torch.ones(8, dtype=torch.bfloat16),
        "model.visual.blocks.0.norm2.bias": torch.zeros(8, dtype=torch.bfloat16),
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(24, 8, dtype=torch.bfloat16),
        "model.visual.blocks.0.attn.qkv.bias": torch.randn(24, dtype=torch.bfloat16),
        "model.visual.blocks.0.attn.proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.visual.blocks.0.attn.proj.bias": torch.randn(8, dtype=torch.bfloat16),
        "model.visual.blocks.0.mlp.linear_fc1.weight": torch.randn(12, 8, dtype=torch.bfloat16),
        "model.visual.blocks.0.mlp.linear_fc1.bias": torch.randn(12, dtype=torch.bfloat16),
        "model.visual.blocks.0.mlp.linear_fc2.weight": torch.randn(8, 12, dtype=torch.bfloat16),
        "model.visual.blocks.0.mlp.linear_fc2.bias": torch.randn(8, dtype=torch.bfloat16),
        "model.visual.merger.norm.weight": torch.ones(8, dtype=torch.bfloat16),
        "model.visual.merger.norm.bias": torch.zeros(8, dtype=torch.bfloat16),
        "model.visual.merger.linear_fc1.weight": torch.randn(32, 32, dtype=torch.bfloat16),
        "model.visual.merger.linear_fc1.bias": torch.randn(32, dtype=torch.bfloat16),
        "model.visual.merger.linear_fc2.weight": torch.randn(8, 32, dtype=torch.bfloat16),
        "model.visual.merger.linear_fc2.bias": torch.randn(8, dtype=torch.bfloat16),
        "model.visual.deepstack_merger_list.0.norm.weight": torch.ones(32, dtype=torch.bfloat16),
        "model.visual.deepstack_merger_list.0.norm.bias": torch.zeros(32, dtype=torch.bfloat16),
        "model.visual.deepstack_merger_list.0.linear_fc1.weight": torch.randn(32, 32, dtype=torch.bfloat16),
        "model.visual.deepstack_merger_list.0.linear_fc1.bias": torch.randn(32, dtype=torch.bfloat16),
        "model.visual.deepstack_merger_list.0.linear_fc2.weight": torch.randn(8, 32, dtype=torch.bfloat16),
        "model.visual.deepstack_merger_list.0.linear_fc2.bias": torch.randn(8, dtype=torch.bfloat16),
    }
    st.save_file(tensors, checkpoint / "model.safetensors")


def test_qwen3vl_safetensors_auto_text_ignores_vision(tmp_path: Path) -> None:
    _require_torch_safetensors()
    checkpoint = tmp_path / "qwen3vl"
    out = tmp_path / "out_qwen3vl_text"
    out.mkdir()
    _write_tiny_qwen3vl_checkpoint(checkpoint)

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "auto",
            "--dry-run",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    names = [entry["name"] for entry in manifest["entries"]]
    assert manifest["model"] == "qwen3vl"
    assert manifest["config"]["mrope_sections"] == [1, 1, 2, 0]
    assert manifest["config"]["mrope_interleaved"] is True
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert any(row["reason"] == "vision_tower_not_in_decoder_pass" for row in audit["ignored_source_tensors"])
    assert "layer.0.q_norm" in names
    assert "layer.0.k_norm" in names
    assert "output.weight" in names
    assert manifest["config"]["tie_word_embeddings"] is False
    assert manifest["config"]["num_deepstack_layers"] == 1

    real_out = tmp_path / "out_qwen3vl_text_real"
    real_out.mkdir()
    subprocess.run(
        [
            sys.executable, str(script),
            "--checkpoint", str(checkpoint),
            "--output", str(real_out / "weights.bump"),
            "--config-out", str(real_out / "config.json"),
            "--manifest-out", str(real_out / "weights_manifest.json"),
            "--arch", "auto",
        ],
        check=True,
    )
    real_manifest = json.loads((real_out / "weights_manifest.json").read_text(encoding="utf-8"))
    preview_offsets = {entry["name"]: entry["file_offset"] for entry in manifest["entries"]}
    real_offsets = {entry["name"]: entry["file_offset"] for entry in real_manifest["entries"]}
    assert preview_offsets == real_offsets
    assert (real_out / "weights.bump").stat().st_size > 0

    build_ir = Path("version/v8/scripts/build_ir_v8.py")
    lowered = out / "lowered_text.json"
    call = out / "call_text.json"
    layout = out / "layout_text.json"
    subprocess.run(
        [
            sys.executable, str(build_ir),
            "--manifest", str(out / "weights_manifest.json"),
            "--mode", "decode",
            "--output", str(out / "ir1_text.json"),
            "--layout-output", str(layout),
            "--lowered-output", str(lowered),
            "--call-output", str(call),
            "--context-len", "4",
        ],
        check=True,
    )
    text_ops = json.loads(lowered.read_text(encoding="utf-8"))["operations"]
    layer0_projection_ops = {
        op["op"]: op["kernel"] for op in text_ops
        if op.get("layer") == 0 and op.get("op") in {"q_proj", "k_proj", "v_proj"}
    }
    assert layer0_projection_ops == {
        "q_proj": "gemm_nt_bf16",
        "k_proj": "gemm_nt_bf16",
        "v_proj": "gemm_nt_bf16",
    }
    assert not any(op.get("op") == "qkv_proj" for op in text_ops)


def test_qwen3vl_safetensors_vision_maps_temporal_patch_split(tmp_path: Path) -> None:
    _require_torch_safetensors()
    checkpoint = tmp_path / "qwen3vl"
    out = tmp_path / "out_qwen3vl_vision"
    out.mkdir()
    _write_tiny_qwen3vl_checkpoint(checkpoint)

    script = Path("version/v8/scripts/convert_safetensors_to_bump_v8.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out / "weights.bump"),
            "--config-out",
            str(out / "config.json"),
            "--manifest-out",
            str(out / "weights_manifest.json"),
            "--arch",
            "qwen3_vl_vision",
        ],
        check=True,
    )

    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "conversion_audit.json").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in manifest["entries"]}
    assert manifest["model"] == "qwen3_vl_vision"
    assert manifest["config"]["deepstack_layer_indices"] == [0]
    assert manifest["config"]["projector_total_out_dim"] == 16
    assert audit["verdict"] == "pass"
    assert audit["unmapped_source_tensors"] == []
    assert entries["v.patch_embd.weight"]["shape"] == [8, 12]
    assert entries["v.patch_embd.weight.1"]["shape"] == [8, 12]
    assert entries["v.patch_embd.weight"]["transform"] == "qwen3vl_patch_temporal_0"
    assert entries["v.patch_embd.weight.1"]["transform"] == "qwen3vl_patch_temporal_1"
    assert entries["v.patch_embd.weight"]["size"] == 8 * 12 * 2
    assert entries["v.position_embd.weight"]["dtype"] == "fp32"
    assert entries["v.deepstack.0.fc1.weight"]["shape"] == [32, 32]
    assert "model.visual.patch_embed.proj.weight" in audit["source_to_targets"]
    assert audit["source_to_targets"]["model.visual.patch_embed.proj.weight"] == [
        "v.patch_embd.weight",
        "v.patch_embd.weight.1",
    ]
    assert manifest["config"]["rope_layout"] == "multi_section_2d"
    assert manifest["config"]["vision_mrope_n_dims"] == 2
    assert manifest["config"]["vision_mrope_sections"] == [1, 1, 0, 0]
    assert manifest["config"]["position_interpolation_policy"] == "align_corners_bilinear"
    assert manifest["config"]["vision_position_storage_boundary"] == "bf16"
    assert (out / "weights.bump").stat().st_size > 0

    build_ir = Path("version/v8/scripts/build_ir_v8.py")
    codegen = Path("version/v8/scripts/codegen_v8.py")
    lowered = out / "lowered_vision.json"
    call = out / "lowered_vision_call.json"
    layout = out / "layout_vision.json"
    generated_c = out / "generated_vision.c"
    subprocess.run(
        [
            sys.executable,
            str(build_ir),
            "--manifest",
            str(out / "weights_manifest.json"),
            "--mode",
            "prefill",
            "--output",
            str(out / "ir1_vision.json"),
            "--layout-output",
            str(layout),
            "--lowered-output",
            str(lowered),
            "--call-output",
            str(call),
            "--context-len",
            "4",
        ],
        check=True,
    )
    lowered_ops = json.loads(lowered.read_text(encoding="utf-8"))["operations"]
    kernels_by_op = {op["op"]: op.get("kernel") for op in lowered_ops}
    assert kernels_by_op["position_embeddings"] == "position_embeddings_add_tiled_2d_align_corners_bf16"
    call_ops = json.loads(call.read_text(encoding="utf-8"))["operations"]
    position_call = next(op for op in call_ops if op["op"] == "position_embeddings")
    assert position_call["function"] == "position_embeddings_add_tiled_2d_align_corners_bf16"
    assert position_call["resolved_contract"]["resolved_contract_id"] == "bf16_tiled_2d_align_corners_rne_residual"
    assert position_call["resolved_contract"]["kernel_id"] == "position_embeddings_add_tiled_2d_align_corners_bf16"
    rope_call = next(op for op in call_ops if op["op"] == "rope_qk")
    rope_args = {arg["name"]: arg["expr"] for arg in rope_call["args"]}
    assert rope_args["n_dims"] == "2"
    assert [rope_args[f"section_{i}"] for i in range(4)] == ["1", "1", "0", "0"]
    for op_name in (
        "patch_proj",
        "patch_proj_aux",
        "qkv_packed_proj",
        "out_proj",
        "mlp_up",
        "mlp_down",
        "projector_fc1",
        "projector_fc2",
    ):
        assert kernels_by_op[op_name] == "gemm_nt_bf16", op_name

    subprocess.run(
        [
            sys.executable,
            str(codegen),
            "--ir",
            str(call),
            "--layout",
            str(layout),
            "--output",
            str(generated_c),
        ],
        check=True,
    )
    generated = generated_c.read_text(encoding="utf-8")
    assert "gemm_nt_bf16(" in generated
    assert "gemm_naive_parallel(" not in generated
    assert "gemm_blocked_serial(" not in generated
