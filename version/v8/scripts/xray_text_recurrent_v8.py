#!/usr/bin/env python3
from __future__ import annotations

"""Schedule-preserving X-ray attribution for recurrent text decoders."""

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from compare_first_token_logits_v8 import load_ck_logits_segmented
from decoder_first_token_parity_v8 import _run_llama_capture


BOUNDARIES = (
    "attn_norm",
    "linear_attn_qkv_mixed",
    "conv_output_raw",
    "conv_output_silu",
    "q_conv_predelta",
    "k_conv_predelta",
    "alpha",
    "beta",
    "new_state",
    "attn_output",
    "final_output",
    "linear_attn_out",
)


@contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def canonicalize_named_axes(
    name: str,
    ck: np.ndarray,
    oracle: np.ndarray,
    state_size: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if name == "new_state":
        state_elements = state_size * state_size
        if state_size <= 0 or ck.size != oracle.size or ck.size % state_elements != 0:
            raise ValueError("new_state requires matching [head, key, value] extents")
        heads = ck.size // state_elements
        # CKE stores [head, key, value]; llama.cpp stores [head, value, key].
        oracle = oracle.reshape(heads, state_size, state_size).transpose(0, 2, 1).reshape(-1)
        return ck, oracle, "oracle:[head,value,key]->[head,key,value]"
    return ck.reshape(-1), oracle.reshape(-1), "identity"


def compare_arrays(name: str, ck: np.ndarray, oracle: np.ndarray, state_size: int = 128) -> dict[str, Any]:
    ck, oracle, transform = canonicalize_named_axes(name, ck, oracle, state_size)
    if ck.shape != oracle.shape:
        return {
            "status": "shape_mismatch",
            "ck_shape": list(ck.shape),
            "oracle_shape": list(oracle.shape),
            "axis_transform": transform,
        }
    delta = ck.astype(np.float64) - oracle.astype(np.float64)
    abs_delta = np.abs(delta)
    return {
        "status": "exact" if np.array_equal(ck, oracle) else "different",
        "elements": int(ck.size),
        "different_elements": int(np.count_nonzero(abs_delta)),
        "max_abs_diff": float(abs_delta.max(initial=0.0)),
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "axis_transform": transform,
    }


def classify(
    rows: list[dict[str, Any]],
    schedules: dict[str, str],
    *,
    material_abs_floor: float = 1e-5,
) -> dict[str, Any] | None:
    for index, row in enumerate(rows):
        if row.get("status") != "different" or float(row.get("max_abs_diff", 0.0)) < material_abs_floor:
            continue
        previous = rows[index - 1] if index else None
        classification = "VALUE_MISMATCH"
        if (
            row.get("boundary") == "linear_attn_qkv_mixed"
            and previous
            and previous.get("boundary") == "attn_norm"
            and previous.get("status") == "exact"
        ):
            classification = "PROJECTION_PROVIDER_MISMATCH"
        elif row.get("boundary") == "new_state":
            classification = "RECURRENT_STATE_REDUCTION_MISMATCH"
        return {
            "classification": classification,
            "logical_token": int(row["logical_token"]),
            "layer": int(row["layer"]),
            "boundary": str(row["boundary"]),
            "previous_exact_boundary": str(previous["boundary"]) if previous and previous.get("status") == "exact" else None,
            "ck_schedule": schedules["ck"],
            "oracle_prefix_schedule": schedules["oracle_prefix"],
            "oracle_decode_schedule": schedules["oracle_decode"],
        }
    return None


def _load_oracle_row(
    root: Path,
    name: str,
    layer: int,
    logical_token: int,
    prompt_tokens: int,
    expected_count: int,
) -> np.ndarray:
    physical_token = prompt_tokens - 1 if logical_token < prompt_tokens else logical_token
    path = root / f"{name}-{layer}-token-{physical_token:06d}-occ-000.bin"
    data = np.fromfile(path, dtype=np.float32)
    if logical_token < prompt_tokens and name != "new_state":
        if data.size != prompt_tokens * expected_count:
            raise ValueError(f"batched oracle extent mismatch for {name}: {data.size} != {prompt_tokens}*{expected_count}")
        return data.reshape(prompt_tokens, expected_count)[logical_token]
    if data.size != expected_count:
        raise ValueError(f"oracle extent mismatch for {name}: {data.size} != {expected_count}")
    return data


def analyze_capture(
    ck_root: Path,
    oracle_root: Path,
    prompt_tokens: int,
    total_tokens: int,
    layer: int,
    state_size: int = 128,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for logical_token in range(total_tokens):
        for boundary in BOUNDARIES:
            ck_path = ck_root / f"tok_{logical_token:04d}_layer_{layer:03d}_{boundary}.f32"
            if not ck_path.is_file():
                continue
            ck = np.fromfile(ck_path, dtype=np.float32)
            if boundary == "new_state" and logical_token < prompt_tokens - 1:
                continue
            try:
                oracle = _load_oracle_row(
                    oracle_root, boundary, layer, logical_token, prompt_tokens, int(ck.size)
                )
            except (FileNotFoundError, ValueError) as exc:
                rows.append({
                    "logical_token": logical_token,
                    "layer": layer,
                    "boundary": boundary,
                    "status": "missing_or_incompatible",
                    "error": str(exc),
                })
                continue
            row = {"logical_token": logical_token, "layer": layer, "boundary": boundary}
            row.update(compare_arrays(boundary, ck, oracle, state_size))
            rows.append(row)
    schedules = {
        "ck": "sequential_decode",
        "oracle_prefix": "batched",
        "oracle_decode": "sequential",
    }
    first_value_divergence = next((
        {
            "logical_token": int(row["logical_token"]),
            "layer": int(row["layer"]),
            "boundary": str(row["boundary"]),
            "max_abs_diff": float(row.get("max_abs_diff", 0.0)),
        }
        for row in rows if row.get("status") == "different"
    ), None)
    material = classify(rows, schedules)
    return {
        "schema": "cke.xray.text-recurrent.v1",
        "schedules": schedules,
        "diagnostic_material_abs_floor": 1e-5,
        "acceptance_policy": "all value differences are reported; the material floor only prioritizes attribution",
        "rows": rows,
        "first_value_divergence": first_value_divergence,
        "first_material_divergence": material,
        "first_divergence": material or first_value_divergence,
    }


def capture_and_analyze(
    model_dir: Path,
    gguf: Path,
    parity_report: Path,
    capture_root: Path,
    layer: int,
    ctx_len: int,
    threads: int,
) -> dict[str, Any]:
    source = json.loads(parity_report.read_text(encoding="utf-8"))
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    state_size = int(config.get("ssm_state_size", 0))
    if state_size <= 0:
        raise ValueError("model config must declare a positive ssm_state_size")
    prompt = [int(token) for token in source["initial_tokens"]]
    full_prefix = [int(token) for token in source["final_prefix"]]
    if full_prefix[: len(prompt)] != prompt:
        raise ValueError("parity report final_prefix does not begin with initial_tokens")
    generated = full_prefix[len(prompt) :]
    ck_root = capture_root / "ck"
    oracle_root = capture_root / "llama"
    ck_root.mkdir(parents=True, exist_ok=True)
    oracle_names = ",".join(f"{name}-{layer}" for name in BOUNDARIES)

    with _temporary_environment({
        "CK_DEBUG_EXPORT_HIDDEN": str(ck_root),
        "CK_DEBUG_EXPORT_HIDDEN_LAYER": str(layer),
        "CK_DEBUG_EXPORT_HIDDEN_NAMES": ",".join(BOUNDARIES),
    }):
        load_ck_logits_segmented(model_dir, prompt, generated, ck_prefill_mode="sequential")

    llama = _run_llama_capture(
        gguf,
        generated,
        ctx_len,
        20,
        threads,
        tokens_before=prompt,
        prefix_decode_mode="batched",
        decode_mode="sequential",
        dump_dir=oracle_root,
        dump_names=oracle_names,
    )
    report = analyze_capture(ck_root, oracle_root, len(prompt), len(full_prefix), layer, state_size)
    report["source_parity_report"] = str(parity_report)
    report["capture_root"] = str(capture_root)
    report["llama_capture"] = {
        "token_count_before": int(llama["meta"].get("token_count_before", -1)),
        "token_count_after": int(llama["meta"].get("token_count_after", -1)),
        "decode_mode": str(llama["meta"].get("decode_mode", "")),
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--parity-report", required=True, type=Path)
    ap.add_argument("--capture-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--ctx-len", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    report = capture_and_analyze(
        args.model_dir.resolve(), args.gguf.resolve(), args.parity_report.resolve(),
        args.capture_root.resolve(), int(args.layer), int(args.ctx_len), int(args.threads),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report.get("first_divergence"), sort_keys=True))
    return 0 if report.get("first_divergence") is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
