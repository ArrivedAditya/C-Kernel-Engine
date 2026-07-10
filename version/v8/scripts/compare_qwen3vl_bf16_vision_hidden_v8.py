#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from array import array
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
NUMERIC_PARITY = SCRIPT_DIR / "numeric_parity_qwen3vl_mmproj_v8.py"


def _import_numeric_parity() -> Any:
    sys.path.insert(0, str(SCRIPT_DIR))
    import numeric_parity_qwen3vl_mmproj_v8 as numeric  # type: ignore

    return numeric


def _metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    if ref.shape != got.shape:
        raise RuntimeError(f"shape mismatch: ref={ref.shape} got={got.shape}")
    diff = got.astype(np.float32, copy=False) - ref.astype(np.float32, copy=False)
    denom = float(np.linalg.norm(ref) * np.linalg.norm(got))
    return {
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(math.sqrt(float(np.mean(diff * diff)))) if diff.size else 0.0,
        "cosine": float(np.dot(ref.reshape(-1), got.reshape(-1)) / denom) if denom > 0.0 else 0.0,
    }


def _load_visual_model(checkpoint: Path, attn_implementation: str):
    import torch
    from safetensors.torch import load_file
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    cfg = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    vision_cfg = Qwen3VLVisionConfig(**cfg["vision_config"])
    if attn_implementation != "auto":
        vision_cfg._attn_implementation = attn_implementation
    model = Qwen3VLVisionModel(vision_cfg)
    inv_freq_fp32 = model.rotary_pos_emb.inv_freq.detach().clone().float()
    model.to(dtype=torch.bfloat16)
    # Hugging Face's full Qwen3-VL BF16 loader keeps the rotary frequency
    # buffer in FP32. Converting this buffer to BF16 changes the vision
    # prefix by enough to produce false CK-vs-PyTorch attribution failures.
    model.rotary_pos_emb.register_buffer("inv_freq", inv_freq_fp32, persistent=False)

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    needed_files = sorted({fname for key, fname in weight_map.items() if key.startswith("model.visual.")})
    state: dict[str, torch.Tensor] = {}
    for fname in needed_files:
        tensors = load_file(str(checkpoint / fname), device="cpu")
        for key, value in tensors.items():
            if key.startswith("model.visual."):
                state[key.removeprefix("model.visual.")] = value
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"visual state mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    return model


def _parse_selector(selector: str) -> tuple[str, int | None]:
    if "@" not in selector:
        return selector, None
    name, layer_s = selector.rsplit("@", 1)
    return name, int(layer_s)


def _torch_captures(
    checkpoint: Path,
    image: Path,
    torch_prefix: Path | None,
    out_dir: Path,
    attn_implementation: str,
    selectors: list[str],
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )

    model = _load_visual_model(checkpoint, attn_implementation)
    captures: dict[str, torch.Tensor] = {}
    parsed = [_parse_selector(sel) for sel in selectors]
    frontend_wanted = {name for name, layer in parsed if layer is None}
    wanted_by_layer: dict[int, set[str]] = {}
    for name, layer in parsed:
        if layer is not None:
            wanted_by_layer.setdefault(layer, set()).add(name)

    handles: list[Any] = []
    original_mlp_forwards: list[tuple[Any, Any]] = []
    original_attn_forwards: list[tuple[Any, Any]] = []

    def make_norm1_hook(layer: int):
        def hook(_module, _inputs, output):
            captures[f"ln1@{layer}"] = output.detach().cpu().float()
        return hook

    def make_attn_forward(attn: Any, layer: int, original_forward: Any):
        def attn_forward(
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: Any = None,
            position_embeddings: Any = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            wanted = wanted_by_layer.get(layer, set())
            internal = {
                "q_proj",
                "k_proj",
                "v_proj",
                "rope_q",
                "rope_k",
                "attn_out_head_major",
                "out_proj",
            }
            if not (internal & wanted):
                return original_forward(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            seq_length = hidden_states.shape[0]
            query_states, key_states, value_states = (
                attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            )
            if "q_proj" in wanted:
                captures[f"q_proj@{layer}"] = query_states.permute(1, 0, 2).contiguous().detach().cpu().float()
            if "k_proj" in wanted:
                captures[f"k_proj@{layer}"] = key_states.permute(1, 0, 2).contiguous().detach().cpu().float()
            if "v_proj" in wanted:
                captures[f"v_proj@{layer}"] = value_states.permute(1, 0, 2).contiguous().detach().cpu().float()

            if position_embeddings is None:
                return original_forward(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
            if "rope_q" in wanted:
                captures[f"rope_q@{layer}"] = query_states.permute(1, 0, 2).contiguous().detach().cpu().float()
            if "rope_k" in wanted:
                captures[f"rope_k@{layer}"] = key_states.permute(1, 0, 2).contiguous().detach().cpu().float()

            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)

            attention_interface = eager_attention_forward
            if attn.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]

            if attn.config._attn_implementation == "flash_attention_2":
                max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
                attn_output, _ = attention_interface(
                    attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=None,
                    scaling=attn.scaling,
                    dropout=0.0 if not attn.training else attn.attention_dropout,
                    cu_seq_lens_q=cu_seqlens,
                    cu_seq_lens_k=cu_seqlens,
                    max_length_q=max_seqlen,
                    max_length_k=max_seqlen,
                    is_causal=False,
                    **kwargs,
                )
            else:
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                splits = [
                    torch.split(tensor, lengths.tolist(), dim=2)
                    for tensor in (query_states, key_states, value_states)
                ]
                attn_outputs = [
                    attention_interface(
                        attn,
                        q,
                        k,
                        v,
                        attention_mask=None,
                        scaling=attn.scaling,
                        dropout=0.0 if not attn.training else attn.attention_dropout,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for q, k, v in zip(*splits)
                ]
                attn_output = torch.cat(attn_outputs, dim=1)

            if "attn_out_head_major" in wanted:
                squeezed = attn_output.squeeze(0)
                if squeezed.shape[0] == seq_length:
                    head_major = squeezed.permute(1, 0, 2).contiguous()
                else:
                    head_major = squeezed.contiguous()
                captures[f"attn_out_head_major@{layer}"] = head_major.detach().cpu().float()

            attn_output = attn_output.reshape(seq_length, -1).contiguous()
            attn_output = attn.proj(attn_output)
            if "out_proj" in wanted:
                captures[f"out_proj@{layer}"] = attn_output.detach().cpu().float()
            return attn_output

        return attn_forward

    def make_layer_out_hook(layer: int):
        def hook(_module, _inputs, output):
            captures[f"layer_out@{layer}"] = output.detach().cpu().float()
        return hook

    def make_norm2_pre_hook(layer: int):
        def hook(_module, inputs):
            captures[f"after_attn@{layer}"] = inputs[0].detach().cpu().float()
        return hook

    def make_norm2_hook(layer: int):
        def hook(_module, _inputs, output):
            captures[f"ffn_inp_normed@{layer}"] = output.detach().cpu().float()
        return hook

    def make_mlp_forward(block: Any, layer: int, original_forward: Any):
        def mlp_forward(hidden_state):
            wanted = wanted_by_layer.get(layer, set())
            if {"mlp_up", "ffn_gelu", "mlp_down"} & wanted:
                up = block.mlp.linear_fc1(hidden_state)
                if "mlp_up" in wanted:
                    captures[f"mlp_up@{layer}"] = up.detach().cpu().float()
                gelu = block.mlp.act_fn(up)
                if "ffn_gelu" in wanted:
                    captures[f"ffn_gelu@{layer}"] = gelu.detach().cpu().float()
                down = block.mlp.linear_fc2(gelu)
                if "mlp_down" in wanted:
                    captures[f"mlp_down@{layer}"] = down.detach().cpu().float()
                return down
            return original_forward(hidden_state)
        return mlp_forward

    for layer, wanted in wanted_by_layer.items():
        if layer < 0 or layer >= len(model.blocks):
            raise ValueError(f"selector layer {layer} out of range for {len(model.blocks)} vision blocks")
        block = model.blocks[layer]
        if "ln1" in wanted:
            handles.append(block.norm1.register_forward_hook(make_norm1_hook(layer)))
        if {"q_proj", "k_proj", "v_proj", "rope_q", "rope_k", "attn_out_head_major", "out_proj"} & wanted:
            original_forward = block.attn.forward
            original_attn_forwards.append((block.attn, original_forward))
            block.attn.forward = make_attn_forward(block.attn, layer, original_forward)  # type: ignore[method-assign]
        if "layer_out" in wanted:
            handles.append(block.register_forward_hook(make_layer_out_hook(layer)))
        if "after_attn" in wanted:
            handles.append(block.norm2.register_forward_pre_hook(make_norm2_pre_hook(layer)))
        if "ffn_inp_normed" in wanted:
            handles.append(block.norm2.register_forward_hook(make_norm2_hook(layer)))
        if {"mlp_up", "ffn_gelu", "mlp_down"} & wanted:
            original_forward = block.mlp.forward
            original_mlp_forwards.append((block.mlp, original_forward))
            block.mlp.forward = make_mlp_forward(block, layer, original_forward)  # type: ignore[method-assign]

    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    image_obj = Image.open(image).convert("RGB")
    proc = processor.image_processor(
        images=image_obj,
        return_tensors="pt",
        min_pixels=1,
        max_pixels=1048576,
    )
    pixel_values = proc["pixel_values"].to(dtype=torch.bfloat16)
    grid = proc["image_grid_thw"]

    try:
        with torch.no_grad():
            if {"vision_patch_sum", "vision_position_embeddings"} & frontend_wanted:
                patch_sum = model.patch_embed(pixel_values)
                if "vision_patch_sum" in frontend_wanted:
                    captures["vision_patch_sum"] = patch_sum.detach().cpu().float()
                if "vision_position_embeddings" in frontend_wanted:
                    pos_embeds = model.fast_pos_embed_interpolate(grid)
                    captures["vision_position_embeddings"] = (patch_sum + pos_embeds).detach().cpu().float()
            final, deepstack = model(pixel_values, grid_thw=grid)
    finally:
        for mlp, original_forward in original_mlp_forwards:
            mlp.forward = original_forward  # type: ignore[method-assign]
        for attn, original_forward in original_attn_forwards:
            attn.forward = original_forward  # type: ignore[method-assign]
        for handle in handles:
            handle.remove()

    prefix_orders: dict[str, dict[str, float]] = {}
    if torch_prefix is not None and torch_prefix.exists():
        ref_prefix = np.fromfile(torch_prefix, dtype=np.float32)
        candidates = {
            "final_then_deep": torch.cat([final, *deepstack], dim=-1),
            "deep_then_final": torch.cat([*deepstack, final], dim=-1),
        }
        for name, tensor in candidates.items():
            arr = tensor.detach().cpu().float().numpy().reshape(-1)
            if arr.shape == ref_prefix.shape:
                prefix_orders[name] = _metrics(ref_prefix, arr)

    missing = [selector for selector in selectors if selector not in captures]
    if missing:
        raise RuntimeError(f"requested PyTorch selectors were not captured: {missing}")

    torch_dir = out_dir / "torch"
    torch_dir.mkdir(parents=True, exist_ok=True)
    tensor_meta: dict[str, Any] = {}
    for name, tensor in captures.items():
        arr = tensor.numpy().astype(np.float32, copy=False)
        path = torch_dir / f"{name.replace('@', '_layer_')}.f32"
        arr.reshape(-1).tofile(path)
        tensor_meta[name] = {"path": str(path), "shape": list(arr.shape)}

    return {
        "pixel_values_shape": list(pixel_values.shape),
        "grid_thw": grid.tolist(),
        "prefix_order_metrics": prefix_orders,
        "tensors": tensor_meta,
    }

def _array_to_np(data: array) -> np.ndarray:
    return np.frombuffer(data.tobytes(), dtype=np.float32).copy()


def _qwen3vl_processor_pixels_to_planar(
    pixel_values: np.ndarray,
    grid_thw: list[int] | tuple[int, int, int],
    *,
    patch_size: int,
    temporal_patch_size: int,
    height: int,
    width: int,
    merge_size: int = 2,
    temporal_atol: float = 1.0e-6,
) -> list[float]:
    """Reconstruct CK's planar image input from Qwen3-VL processor patches.

    Hugging Face feeds Qwen3-VL vision patchified, normalized ``pixel_values``
    shaped ``[grid_h * grid_w, 3 * temporal_patch * patch * patch]``. CK's
    generated BF16 vision encoder still accepts a single normalized CHW image
    plane and internally applies the two temporal patch slices. That contract is
    exact for image inputs because Qwen3-VL duplicates the still image across
    temporal slices. If a future processor stops doing that, the generated input
    ABI must change instead of silently comparing different tensors.
    """
    if len(grid_thw) != 3:
        raise ValueError(f"expected grid_thw with 3 entries, got {grid_thw!r}")
    grid_t, grid_h, grid_w = [int(x) for x in grid_thw]
    if grid_t != 1:
        raise ValueError(f"single-image CK planar input expects grid_t=1, got {grid_t}")
    if grid_h * int(patch_size) != int(height) or grid_w * int(patch_size) != int(width):
        raise ValueError(
            "processor grid does not match CK runtime image geometry: "
            f"grid={grid_thw} patch={patch_size} runtime={height}x{width}"
        )

    arr = np.asarray(pixel_values, dtype=np.float32)
    expected_shape = (grid_h * grid_w, 3 * int(temporal_patch_size) * int(patch_size) * int(patch_size))
    if arr.shape != expected_shape:
        raise ValueError(f"pixel_values shape mismatch: got {arr.shape}, expected {expected_shape}")

    merge = int(merge_size)
    if merge <= 0:
        raise ValueError(f"merge_size must be positive, got {merge_size}")
    if grid_h % merge != 0 or grid_w % merge != 0:
        raise ValueError(f"Qwen3-VL grid must be divisible by merge_size: grid={grid_thw} merge={merge}")

    # HF flattens patches after this logical transpose:
    #   (t, gh//m, gw//m, mh, mw, c, temporal, py, px)
    # CK's image ABI wants a planar CHW image whose im2patch pass sees row-major
    # patches. Undo HF's merge-tiled order before reconstructing the plane.
    tiled = arr.reshape(
        grid_t,
        grid_h // merge,
        grid_w // merge,
        merge,
        merge,
        3,
        int(temporal_patch_size),
        int(patch_size),
        int(patch_size),
    )
    patches = tiled[0].transpose(0, 2, 1, 3, 4, 5, 6, 7).reshape(
        grid_h, grid_w, 3, int(temporal_patch_size), int(patch_size), int(patch_size)
    )
    if int(temporal_patch_size) > 1:
        temporal_ref = patches[:, :, :, 0, :, :]
        for t in range(1, int(temporal_patch_size)):
            max_diff = float(np.max(np.abs(temporal_ref - patches[:, :, :, t, :, :])))
            if max_diff > temporal_atol:
                raise ValueError(
                    "processor temporal patch slices differ; CK's single-plane "
                    f"vision input ABI is not exact for this sample (slice={t}, max_diff={max_diff})"
                )

    planar = patches[:, :, :, 0, :, :].transpose(2, 0, 3, 1, 4).reshape(3, int(height), int(width))
    return planar.reshape(-1).astype(np.float32, copy=False).tolist()


def _load_qwen3vl_processor_planar(checkpoint: Path, image: Path, *, height: int, width: int) -> list[float]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    cfg = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    vision_cfg = cfg.get("vision_config", {})
    patch_size = int(vision_cfg.get("patch_size", 16))
    temporal_patch_size = int(vision_cfg.get("temporal_patch_size", 2))
    merge_size = int(vision_cfg.get("spatial_merge_size") or vision_cfg.get("merge_size") or 2)

    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    image_obj = Image.open(image).convert("RGB")
    proc = processor.image_processor(
        images=image_obj,
        return_tensors="pt",
        min_pixels=1,
        max_pixels=1048576,
    )
    pixel_values = proc["pixel_values"].detach().to(dtype=torch.float32).cpu().numpy()
    grid = [int(x) for x in proc["image_grid_thw"][0].tolist()]
    return _qwen3vl_processor_pixels_to_planar(
        pixel_values,
        grid,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        height=int(height),
        width=int(width),
    )


def _run_ck_selector(args: argparse.Namespace, selector: str, numeric: Any) -> np.ndarray:
    runtime_dir = args.runtime_dir.resolve()
    cfg = json.loads((runtime_dir / "config.json").read_text(encoding="utf-8"))
    planar_image = _load_qwen3vl_processor_planar(
        args.checkpoint.resolve(),
        args.image.resolve(),
        height=int(cfg["image_height"]),
        width=int(cfg["image_width"]),
    )
    data = numeric._run_generated_encoder(
        model_so=runtime_dir / "libqwen3vl_bf16_encoder_v8.so",
        weights_bump=args.weights_bump.resolve(),
        manifest_map=runtime_dir / "weights_manifest.map",
        layout_path=runtime_dir / "layout.json",
        planar_image=planar_image,
        output_name=selector,
    )
    return _array_to_np(data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare Qwen3-VL BF16 vision hidden tensors against CK hidden exports.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--runtime-dir", type=Path, required=True)
    ap.add_argument("--weights-bump", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--torch-prefix", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("build/qwen3vl_bf16_hidden_compare"))
    ap.add_argument(
        "--selector",
        action="append",
        default=[],
        help="Tensor selector such as ffn_inp_normed@9. May be repeated.",
    )
    ap.add_argument("--threads", type=int, default=int(os.environ.get("CK_NUM_THREADS", "20") or "20"))
    ap.add_argument("--attn-implementation", choices=("auto", "eager", "sdpa"), default="auto")
    ap.add_argument("--skip-ck", action="store_true", help="Only run the PyTorch hook/reference side")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CK_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    selectors = args.selector or ["ffn_inp_normed@9", "mlp_up@9", "ffn_gelu@9", "mlp_down@9", "layer_out@9"]

    t0 = time.perf_counter()
    torch_report = _torch_captures(
        args.checkpoint.resolve(),
        args.image.resolve(),
        args.torch_prefix,
        args.out_dir,
        args.attn_implementation,
        selectors,
    )
    t_torch = time.perf_counter()

    rows: dict[str, Any] = {}
    if not args.skip_ck:
        numeric = _import_numeric_parity()
        ck_dir = args.out_dir / "ck"
        ck_dir.mkdir(parents=True, exist_ok=True)
        for selector in selectors:
            print(f"[ck] {selector}", flush=True)
            ck = _run_ck_selector(args, selector, numeric)
            ck_path = ck_dir / f"{selector.replace('@', '_layer_')}.f32"
            ck.tofile(ck_path)
            torch_path = Path(torch_report["tensors"][selector]["path"])
            ref = np.fromfile(torch_path, dtype=np.float32)
            rows[selector] = {
                "ck_path": str(ck_path),
                "torch_path": str(torch_path),
                "shape": list(ck.shape),
                **_metrics(ref, ck),
            }

    report = {
        "checkpoint": str(args.checkpoint),
        "runtime_dir": str(args.runtime_dir),
        "weights_bump": str(args.weights_bump),
        "image": str(args.image),
        "selectors": selectors,
        "attn_implementation": args.attn_implementation,
        "timings_sec": {
            "torch": t_torch - t0,
            "total": time.perf_counter() - t0,
        },
        "torch": torch_report,
        "comparisons": rows,
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "comparisons": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
