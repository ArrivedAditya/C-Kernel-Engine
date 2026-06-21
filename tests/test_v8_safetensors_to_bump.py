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
    assert cfg["ssm_conv_kernel"] == 4
    assert cfg["ssm_conv_history"] == 4
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
    assert conv_state["shape"] == "[3, 4, 20]"

    lowered_ops = json.loads(lowered.read_text(encoding="utf-8"))["operations"]
    by_op = {(op["layer"], op["op"]): op for op in lowered_ops}

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
