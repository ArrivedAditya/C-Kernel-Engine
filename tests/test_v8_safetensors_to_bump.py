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
    assert names[-1] == "output.weight"
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
