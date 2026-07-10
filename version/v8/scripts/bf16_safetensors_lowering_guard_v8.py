#!/usr/bin/env python3
"""Synthetic BF16 safetensors -> BUMP -> IR/codegen guard for v8.

This is intentionally small and dependency-light. It catches the class of bug
where BF16 safetensors weights are converted correctly but lowered/codegened as
FP32 GEMM calls.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _optional_deps() -> tuple[object | None, object | None]:
    try:
        import torch  # type: ignore
        import safetensors.torch as st  # type: ignore
    except Exception as exc:
        print(f"SKIP: torch/safetensors unavailable for BF16 guard: {exc}")
        return None, None
    return torch, st


def _assert_qwen3vl_reference_rotary_keeps_fp32(torch: object) -> None:
    """Guard the PyTorch BF16 reference convention used by Qwen3-VL vision.

    The full Hugging Face Qwen3-VL BF16 loader converts visual parameters to
    BF16 but keeps the non-persistent rotary frequency buffer in FP32. A helper
    that blindly calls ``model.to(torch.bfloat16)`` rounds that buffer and no
    longer matches the full-model reference path.
    """
    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    except Exception as exc:
        print(f"SKIP: transformers Qwen3-VL unavailable for BF16 rotary guard: {exc}")
        return

    cfg = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        out_hidden_size=8,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=4,
        deepstack_visual_indexes=[],
    )
    model = Qwen3VLVisionModel(cfg)
    inv_freq_fp32 = model.rotary_pos_emb.inv_freq.detach().clone().float()
    model.to(dtype=torch.bfloat16)
    model.rotary_pos_emb.register_buffer("inv_freq", inv_freq_fp32, persistent=False)
    if model.rotary_pos_emb.inv_freq.dtype is not torch.float32:
        raise AssertionError("Qwen3-VL BF16 reference must keep rotary inv_freq in FP32")


def _assert_qwen3vl_vision_mrope_kernel() -> None:
    sys.path.insert(0, str(Path("unittest").resolve()))
    try:
        import test_vision  # type: ignore
    except OSError as exc:
        print(f"SKIP: Qwen3-VL vision M-RoPE guard needs built vision library: {exc}")
        return
    test_vision.test_mrope_qk_vision_qwen3vl_full_head()


def _assert_qwen3vl_processor_planar_contract() -> None:
    """Guard CK's BF16 Qwen3-VL image-input comparison contract.

    The PyTorch reference consumes normalized patchified ``pixel_values``. CK's
    current generated vision ABI consumes a single normalized CHW plane and then
    applies both temporal patch weight slices internally. This is exact only when
    still-image temporal slices are duplicates, so make that explicit.
    """
    import numpy as np

    sys.path.insert(0, str(Path("version/v8/scripts").resolve()))
    from compare_qwen3vl_bf16_vision_hidden_v8 import _qwen3vl_processor_pixels_to_planar  # type: ignore

    grid_h, grid_w = 4, 6
    patch = 2
    temporal = 2
    merge = 2
    height, width = grid_h * patch, grid_w * patch
    one_slice = np.arange(grid_h * grid_w * 3 * patch * patch, dtype=np.float32).reshape(
        grid_h, grid_w, 3, patch, patch
    )
    hf_order = one_slice.reshape(grid_h // merge, merge, grid_w // merge, merge, 3, patch, patch)
    hf_order = hf_order.transpose(0, 2, 1, 3, 4, 5, 6).reshape(grid_h * grid_w, 3, patch, patch)
    pixels = np.stack([hf_order, hf_order], axis=2).reshape(grid_h * grid_w, 3 * temporal * patch * patch)
    planar = _qwen3vl_processor_pixels_to_planar(
        pixels,
        [1, grid_h, grid_w],
        patch_size=patch,
        temporal_patch_size=temporal,
        height=height,
        width=width,
        merge_size=merge,
    )
    expected = one_slice.transpose(2, 0, 3, 1, 4).reshape(-1)
    got = np.asarray(planar, dtype=np.float32)
    if not np.array_equal(got, expected):
        raise AssertionError("Qwen3-VL processor pixel reconstruction does not match CHW patch order")

    bad = pixels.copy()
    bad[:, 3 * patch * patch :] += 1.0
    try:
        _qwen3vl_processor_pixels_to_planar(
            bad,
            [1, grid_h, grid_w],
            patch_size=patch,
            temporal_patch_size=temporal,
            height=height,
            width=width,
            merge_size=merge,
        )
    except ValueError as exc:
        if "temporal patch slices differ" not in str(exc):
            raise
    else:
        raise AssertionError("Qwen3-VL processor planar contract must reject non-duplicate temporal slices")


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
    }
    for idx in range(len(vocab), vocab_size):
        vocab[f"<tok_{idx}>"] = idx
    merges = ["H e", "w o", "r l", "d </w>", "Ġ t", "t e", "e s", "s t"]
    (checkpoint / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {"type": "BPE", "vocab": vocab, "merges": merges, "unk_token": "<unk>"},
                "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
                "decoder": {"type": "ByteLevel"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (checkpoint / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 64}) + "\n", encoding="utf-8")


def _write_tiny_qwen3vl_checkpoint(checkpoint: Path, torch: object, st: object) -> None:
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
                    "tie_word_embeddings": False,
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


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def run_guard(workdir: Path) -> None:
    torch, st = _optional_deps()
    if torch is None or st is None:
        return
    _assert_qwen3vl_reference_rotary_keeps_fp32(torch)
    _assert_qwen3vl_vision_mrope_kernel()
    _assert_qwen3vl_processor_planar_contract()
    checkpoint = workdir / "tiny_qwen3vl"
    out = workdir / "out"
    out.mkdir(parents=True, exist_ok=True)
    _write_tiny_qwen3vl_checkpoint(checkpoint, torch, st)

    _run(
        [
            sys.executable,
            "version/v8/scripts/convert_safetensors_to_bump_v8.py",
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
        ]
    )
    manifest = json.loads((out / "weights_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["rope_layout"] == "multi_section_2d"
    assert manifest["config"]["vision_mrope_n_dims"] == 4
    assert manifest["config"]["vision_mrope_sections"] == [1, 1, 1, 1]

    lowered = out / "lowered_vision.json"
    call = out / "lowered_vision_call.json"
    layout = out / "layout_vision.json"
    generated_c = out / "generated_vision.c"
    _run(
        [
            sys.executable,
            "version/v8/scripts/build_ir_v8.py",
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
        ]
    )
    lowered_ops = json.loads(lowered.read_text(encoding="utf-8"))["operations"]
    kernels_by_op = {op["op"]: op.get("kernel") for op in lowered_ops}
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
        got = kernels_by_op.get(op_name)
        if got != "gemm_nt_bf16":
            raise AssertionError(f"{op_name}: expected gemm_nt_bf16, got {got}")

    _run(
        [
            sys.executable,
            "version/v8/scripts/codegen_v8.py",
            "--ir",
            str(call),
            "--layout",
            str(layout),
            "--output",
            str(generated_c),
        ]
    )
    generated = generated_c.read_text(encoding="utf-8")
    if "gemm_nt_bf16(" not in generated:
        raise AssertionError("generated C does not call gemm_nt_bf16")
    for forbidden in ("gemm_naive_parallel(", "gemm_blocked_serial("):
        if forbidden in generated:
            raise AssertionError(f"generated C still contains {forbidden}")
    print("PASS: v8 BF16 safetensors lowering/codegen guard")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.workdir is not None:
        args.workdir.mkdir(parents=True, exist_ok=True)
        run_guard(args.workdir)
    else:
        with tempfile.TemporaryDirectory(prefix="ck_bf16_guard_") as td:
            run_guard(Path(td))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
