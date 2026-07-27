#!/usr/bin/env python3
"""Compare generated Whisper encoder checkpoints against PyTorch."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Checkpoint:
    stop: int
    name: str
    buffer: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    layer: int | None = None
    operation: str | None = None


def checkpoint_table(config: dict[str, Any]) -> dict[int, Checkpoint]:
    """Build the stop-op map from architecture dimensions, not model names."""
    channels = int(config["audio_conv1_output_channels"])
    conv1_frames = int(config["audio_conv1_output_frames"])
    conv2_frames = int(config["audio_conv2_output_frames"])
    tokens = int(config["context_length"])
    embed = int(config["embed_dim"])
    heads = int(config["num_heads"])
    head_dim = int(config["head_dim"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_layers"])

    if channels != embed or conv2_frames != tokens or heads * head_dim != embed:
        raise ValueError("Whisper encoder dimensions violate the generated layout contract")

    table = {
        0: Checkpoint(0, "stem.conv1", "audio_conv_1", (channels, conv1_frames), ("channel", "frame"), operation="conv1"),
        1: Checkpoint(1, "stem.gelu1", "audio_conv_1", (channels, conv1_frames), ("channel", "frame"), operation="gelu1"),
        2: Checkpoint(2, "stem.conv2", "audio_conv_2", (embed, conv2_frames), ("channel", "frame"), operation="conv2"),
        3: Checkpoint(3, "stem.gelu2", "audio_conv_2", (embed, conv2_frames), ("channel", "frame"), operation="gelu2"),
        4: Checkpoint(4, "stem.token_major", "embedded_input", (tokens, embed), ("token", "channel"), operation="transpose"),
        5: Checkpoint(5, "stem.position", "embedded_input", (tokens, embed), ("token", "channel"), operation="position"),
    }
    names = (
        ("residual.attention_input", "residual", (tokens, embed), ("token", "channel"), "residual_save"),
        ("attention.layer_norm", "embedded_input", (tokens, embed), ("token", "channel"), "layernorm"),
        ("attention.q_projection", "q_scratch", (tokens, embed), ("token", "channel"), "q_proj"),
        ("attention.q_head_major", "q_scratch", (heads, tokens, head_dim), ("head", "token", "channel"), "q_transpose"),
        ("attention.k_projection", "k_scratch", (tokens, embed), ("token", "channel"), "k_proj"),
        ("attention.k_head_major", "k_scratch", (heads, tokens, head_dim), ("head", "token", "channel"), "k_transpose"),
        ("attention.v_projection", "v_scratch", (tokens, embed), ("token", "channel"), "v_proj"),
        ("attention.v_head_major", "v_scratch", (heads, tokens, head_dim), ("head", "token", "channel"), "v_transpose"),
        ("attention.output_head_major", "attn_scratch", (heads, tokens, head_dim), ("head", "token", "channel"), "attention"),
        ("attention.output_token_major", "attn_scratch", (tokens, embed), ("token", "channel"), "attention_transpose"),
        ("attention.out_projection", "embedded_input", (tokens, embed), ("token", "channel"), "out_proj"),
        ("attention.residual", "embedded_input", (tokens, embed), ("token", "channel"), "attention_residual"),
        ("residual.mlp_input", "residual", (tokens, embed), ("token", "channel"), "residual_save"),
        ("mlp.layer_norm", "embedded_input", (tokens, embed), ("token", "channel"), "layernorm"),
        ("mlp.up_projection", "mlp_scratch", (tokens, intermediate), ("token", "channel"), "mlp_up"),
        ("mlp.gelu", "mlp_scratch", (tokens, intermediate), ("token", "channel"), "gelu"),
        ("mlp.down_projection", "embedded_input", (tokens, embed), ("token", "channel"), "mlp_down"),
        ("mlp.residual", "embedded_input", (tokens, embed), ("token", "channel"), "mlp_residual"),
    )
    for layer in range(layers):
        base = 6 + layer * len(names)
        for relative, (name, buffer, shape, axes, operation) in enumerate(names):
            stop = base + relative
            table[stop] = Checkpoint(
                stop,
                f"layer.{layer}.{name}",
                buffer,
                shape,
                axes,
                layer=layer,
                operation=operation,
            )
    final_stop = 6 + layers * len(names)
    table[final_stop] = Checkpoint(
        final_stop,
        "encoder.final_layer_norm",
        "embedded_input",
        (tokens, embed),
        ("token", "channel"),
        operation="final_layer_norm",
    )
    return table


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _metrics(reference: np.ndarray, actual: np.ndarray, axes: tuple[str, ...]) -> dict[str, Any]:
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {actual.shape}")
    ref64 = reference.astype(np.float64, copy=False)
    got64 = actual.astype(np.float64, copy=False)
    delta = got64 - ref64
    absolute = np.abs(delta)
    flat = int(np.argmax(absolute)) if absolute.size else 0
    coordinate = np.unravel_index(flat, absolute.shape) if absolute.size else ()
    denominator = float(np.linalg.norm(ref64) * np.linalg.norm(got64))
    rmse = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    ref_rms = float(np.sqrt(np.mean(ref64 * ref64))) if delta.size else 0.0
    exact = int(np.count_nonzero(reference == actual))
    return {
        "cosine": float(np.dot(ref64.reshape(-1), got64.reshape(-1)) / denominator) if denominator else 1.0,
        "rmse": rmse,
        "relative_rmse": rmse / ref_rms if ref_rms else (0.0 if rmse == 0.0 else math.inf),
        "mean_abs": float(np.mean(absolute)) if absolute.size else 0.0,
        "max_abs": float(absolute.reshape(-1)[flat]) if absolute.size else 0.0,
        "worst_coordinate": {axis: int(value) for axis, value in zip(axes, coordinate)},
        "exact_elements": exact,
        "total_elements": int(reference.size),
        "exact_ratio": exact / int(reference.size) if reference.size else 1.0,
        "byte_exact": exact == int(reference.size),
        "finite": bool(np.isfinite(reference).all() and np.isfinite(actual).all()),
    }


def _head_major(values: Any, heads: int, head_dim: int) -> Any:
    return values.reshape(values.shape[0], values.shape[1], heads, head_dim).transpose(1, 2).squeeze(0)


def _token_major(values: Any) -> Any:
    if values.ndim == 3:
        return values.permute(1, 0, 2).reshape(values.shape[1], -1)
    return values.transpose(1, 2).reshape(values.shape[0], values.shape[2], -1)


def _pytorch_checkpoints(model: Any, features: Any) -> dict[int, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    encoder = model.model.encoder
    heads = int(encoder.config.encoder_attention_heads)
    embed = int(encoder.config.d_model)
    head_dim = embed // heads
    scaling = head_dim ** -0.5
    refs: dict[int, np.ndarray] = {}

    conv1 = encoder.conv1(features)
    refs[0] = conv1[0].detach().cpu().numpy()
    hidden = functional.gelu(conv1)
    refs[1] = hidden[0].detach().cpu().numpy()
    conv2 = encoder.conv2(hidden)
    refs[2] = conv2[0].detach().cpu().numpy()
    hidden = functional.gelu(conv2)
    refs[3] = hidden[0].detach().cpu().numpy()
    hidden = hidden.permute(0, 2, 1)
    refs[4] = hidden[0].detach().cpu().numpy()
    hidden = hidden + encoder.embed_positions.weight
    refs[5] = hidden[0].detach().cpu().numpy()

    for layer_index, layer in enumerate(encoder.layers):
        base = 6 + layer_index * 18
        residual = hidden
        refs[base] = residual[0].detach().cpu().numpy()
        normalized = layer.self_attn_layer_norm(hidden)
        refs[base + 1] = normalized[0].detach().cpu().numpy()

        q_token = layer.self_attn.q_proj(normalized)
        k_token = layer.self_attn.k_proj(normalized)
        v_token = layer.self_attn.v_proj(normalized)
        refs[base + 2] = q_token[0].detach().cpu().numpy()
        q_head = _head_major(q_token, heads, head_dim)
        refs[base + 3] = q_head.detach().cpu().numpy()
        refs[base + 4] = k_token[0].detach().cpu().numpy()
        k_head = _head_major(k_token, heads, head_dim)
        refs[base + 5] = k_head.detach().cpu().numpy()
        refs[base + 6] = v_token[0].detach().cpu().numpy()
        v_head = _head_major(v_token, heads, head_dim)
        refs[base + 7] = v_head.detach().cpu().numpy()

        # Whisper scales Q before QK. Keeping this explicit reveals arithmetic-order
        # drift even though scaling Q and scaling the completed dot are algebraically equal.
        scores = torch.matmul(q_head * scaling, k_head.transpose(-1, -2))
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        attention_head = torch.matmul(probabilities, v_head)
        refs[base + 8] = attention_head.detach().cpu().numpy()
        attention_token = _token_major(attention_head)
        refs[base + 9] = attention_token.detach().cpu().numpy()
        projected = layer.self_attn.out_proj(attention_token.unsqueeze(0))
        refs[base + 10] = projected[0].detach().cpu().numpy()
        hidden = residual + projected
        refs[base + 11] = hidden[0].detach().cpu().numpy()

        residual = hidden
        refs[base + 12] = residual[0].detach().cpu().numpy()
        normalized = layer.final_layer_norm(hidden)
        refs[base + 13] = normalized[0].detach().cpu().numpy()
        up = layer.fc1(normalized)
        refs[base + 14] = up[0].detach().cpu().numpy()
        activated = functional.gelu(up)
        refs[base + 15] = activated[0].detach().cpu().numpy()
        down = layer.fc2(activated)
        refs[base + 16] = down[0].detach().cpu().numpy()
        hidden = residual + down
        refs[base + 17] = hidden[0].detach().cpu().numpy()

    final_stop = 6 + len(encoder.layers) * 18
    hidden = encoder.layer_norm(hidden)
    refs[final_stop] = hidden[0].detach().cpu().numpy()
    return refs


class GeneratedEncoder:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.library_path = run_dir / "libmodel.so"
        self.weights_path = run_dir / "weights.bump"
        self.manifest_path = run_dir / "weights_manifest.map"
        for path in (self.library_path, self.weights_path, self.manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.lib = ctypes.CDLL(str(self.library_path))
        self.lib.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.ck_model_init_with_manifest.restype = ctypes.c_int
        self.lib.ck_model_free.argtypes = []
        self.lib.ck_model_free.restype = None
        self.lib.ck_model_run_encoder.argtypes = []
        self.lib.ck_model_run_encoder.restype = ctypes.c_int
        self.lib.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
        self.lib.ck_model_get_named_activation_ptr.restype = ctypes.c_void_p
        self.lib.ck_model_get_named_activation_nbytes.argtypes = [ctypes.c_char_p]
        self.lib.ck_model_get_named_activation_nbytes.restype = ctypes.c_ssize_t

    def run(self, features: np.ndarray, checkpoint: Checkpoint) -> tuple[np.ndarray, float]:
        os.environ["CK_STOP_OP"] = str(checkpoint.stop)
        rc = int(self.lib.ck_model_init_with_manifest(
            str(self.weights_path).encode(), str(self.manifest_path).encode()
        ))
        if rc != 0:
            raise RuntimeError(f"ck_model_init_with_manifest failed with code {rc}")
        try:
            feature_ptr = int(self.lib.ck_model_get_named_activation_ptr(b"audio_features") or 0)
            feature_nbytes = int(self.lib.ck_model_get_named_activation_nbytes(b"audio_features"))
            if feature_ptr == 0 or feature_nbytes != features.nbytes:
                raise RuntimeError(
                    f"audio_features contract mismatch: ptr={feature_ptr} "
                    f"nbytes={feature_nbytes} expected={features.nbytes}"
                )
            ctypes.memmove(feature_ptr, features.ctypes.data, features.nbytes)
            started = time.perf_counter()
            rc = int(self.lib.ck_model_run_encoder())
            elapsed = time.perf_counter() - started
            if rc != 0:
                raise RuntimeError(f"ck_model_run_encoder failed with code {rc}")
            output_ptr = int(self.lib.ck_model_get_named_activation_ptr(checkpoint.buffer.encode()) or 0)
            output_nbytes = int(self.lib.ck_model_get_named_activation_nbytes(checkpoint.buffer.encode()))
            count = math.prod(checkpoint.shape)
            required = count * np.dtype(np.float32).itemsize
            if output_ptr == 0 or output_nbytes < required:
                raise RuntimeError(
                    f"{checkpoint.buffer} contract mismatch: ptr={output_ptr} "
                    f"nbytes={output_nbytes} required={required}"
                )
            raw = np.ctypeslib.as_array(
                ctypes.cast(output_ptr, ctypes.POINTER(ctypes.c_float)),
                shape=(count,),
            )
            return raw[:count].copy().reshape(checkpoint.shape), elapsed
        finally:
            self.lib.ck_model_free()


def _parse_stops(text: str, table: dict[int, Checkpoint]) -> list[int]:
    if text == "key":
        key_operations = {
            "conv1", "gelu1", "conv2", "gelu2", "position", "layernorm",
            "q_transpose", "k_transpose", "v_transpose", "attention", "out_proj",
            "attention_residual", "gelu", "mlp_residual", "final_layer_norm",
        }
        return [
            stop for stop, checkpoint in sorted(table.items())
            if checkpoint.operation in key_operations
        ]
    if text == "all":
        return sorted(table)
    stops = [int(value.strip()) for value in text.split(",") if value.strip()]
    invalid = [stop for stop in stops if stop not in table]
    if invalid:
        raise ValueError(f"unknown stop operations: {invalid}")
    return stops


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stops", default="key", help="Comma-separated stop ops, 'key', or 'all'")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--input-stddev", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--material-max-abs", type=float, default=1.0e-4)
    parser.add_argument("--material-relative-rmse", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch
    from transformers import WhisperForConditionalGeneration
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["CK_NUM_THREADS"] = str(args.threads)

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    call = json.loads((run_dir / "call.json").read_text(encoding="utf-8"))
    table = checkpoint_table(config)
    stops = _parse_stops(args.stops, table)

    rng = np.random.default_rng(args.seed)
    features = rng.normal(
        0.0,
        args.input_stddev,
        (int(config["audio_feature_channels"]), int(config["audio_feature_frames"])),
    ).astype(np.float32)
    feature_tensor = torch.from_numpy(features).unsqueeze(0)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(args.checkpoint.resolve()),
        local_files_only=True,
    ).eval()
    with torch.inference_mode():
        references = _pytorch_checkpoints(model, feature_tensor)

    runtime = GeneratedEncoder(run_dir)
    rows = []
    first_non_exact = None
    first_material = None
    for stop in stops:
        checkpoint = table[stop]
        actual, elapsed = runtime.run(features, checkpoint)
        reference = np.asarray(references[stop], dtype=np.float32)
        metrics = _metrics(reference, actual, checkpoint.axes)
        material = (
            not metrics["finite"]
            or metrics["max_abs"] > args.material_max_abs
            or metrics["relative_rmse"] > args.material_relative_rmse
        )
        operation = call["operations"][stop]
        row = {
            "stop": stop,
            "checkpoint": checkpoint.name,
            "layer": checkpoint.layer,
            "operation": operation.get("op"),
            "function": operation.get("function"),
            "buffer": checkpoint.buffer,
            "shape": list(checkpoint.shape),
            "axes": list(checkpoint.axes),
            "elapsed_seconds": elapsed,
            "reference_sha256": _sha256_array(reference),
            "actual_sha256": _sha256_array(actual),
            "metrics": metrics,
            "material_divergence": material,
            "required_contract": operation.get("required_contract"),
            "resolved_contract": operation.get("resolved_contract"),
        }
        rows.append(row)
        if first_non_exact is None and not metrics["byte_exact"]:
            first_non_exact = stop
        if first_material is None and material:
            first_material = stop
        print(
            f"stop={stop:02d} {checkpoint.name:<42} "
            f"max_abs={metrics['max_abs']:.9g} rmse={metrics['rmse']:.9g} "
            f"exact={metrics['exact_ratio']:.4%} material={'yes' if material else 'no'} "
            f"time={elapsed:.3f}s",
            flush=True,
        )

    report = {
        "schema": "cke.whisper_encoder_pytorch_xray",
        "schema_version": 1,
        "status": "fail" if first_material is not None else "pass",
        "first_non_exact_stop": first_non_exact,
        "first_material_divergence_stop": first_material,
        "thresholds": {
            "max_abs": args.material_max_abs,
            "relative_rmse": args.material_relative_rmse,
        },
        "fixture": {
            "seed": args.seed,
            "distribution": "normal",
            "stddev": args.input_stddev,
            "shape": list(features.shape),
            "sha256": _sha256_array(features),
        },
        "provenance": {
            "run_dir": str(run_dir),
            "checkpoint": str(args.checkpoint.resolve()),
            "runtime_sha256": _sha256_file(runtime.library_path),
            "weights_sha256": _sha256_file(runtime.weights_path),
            "call_ir_sha256": _sha256_file(run_dir / "call.json"),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "threads": args.threads,
        },
        "checkpoints": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"status={report['status']} first_non_exact={first_non_exact} "
        f"first_material={first_material} report={args.output}"
    )
    return 1 if first_material is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
