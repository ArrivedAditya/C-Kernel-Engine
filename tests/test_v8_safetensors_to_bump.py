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
    names = [entry["name"] for entry in manifest["entries"]]
    assert manifest["model"] == "qwen3"
    assert names[0] == "token_emb"
    assert "layer.0.q_norm" in names
    assert "layer.0.k_norm" in names
    assert names[-1] == "output.weight"
    assert (out / "weights.bump").stat().st_size > 0
