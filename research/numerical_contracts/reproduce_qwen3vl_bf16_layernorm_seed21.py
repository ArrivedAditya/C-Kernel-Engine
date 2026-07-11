#!/usr/bin/env python3
"""Reproduce the public seed-21 BF16 LayerNorm reduction mismatch."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((Path(__file__).with_name("qwen3vl_bf16_layernorm_seed21.json")).read_text())


def to_bf16(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32, copy=False).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    config = FIXTURE["public_generator"]
    dim = int(config["dimension"])
    generator = torch.Generator().manual_seed(int(config["seed"]))
    x = torch.randn(1, dim, generator=generator).to(torch.bfloat16)
    gamma = torch.randn(dim, generator=generator).to(torch.bfloat16)
    beta = torch.randn(dim, generator=generator).to(torch.bfloat16)
    reference = torch.nn.functional.layer_norm(
        x, (dim,), gamma, beta, float(config["epsilon"])
    ).float().numpy()

    x_bf16 = to_bf16(x.float().numpy())
    output_bf16 = np.empty_like(x_bf16)
    gamma_f32 = gamma.float().numpy()
    beta_f32 = beta.float().numpy()
    mean = np.empty(1, np.float32)
    rstd = np.empty(1, np.float32)
    scratch_input = np.empty(dim, np.float32)
    scratch_output = np.empty(dim, np.float32)

    library = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
    function = library.layernorm_forward_unrolled_slice_bf16
    float_p = ctypes.POINTER(ctypes.c_float)
    bf16_p = ctypes.POINTER(ctypes.c_uint16)
    function.argtypes = [bf16_p, float_p, float_p, bf16_p, float_p, float_p,
                         ctypes.c_int, ctypes.c_int, ctypes.c_float, float_p, float_p]
    function(
        x_bf16.ctypes.data_as(bf16_p),
        gamma_f32.ctypes.data_as(float_p),
        beta_f32.ctypes.data_as(float_p),
        output_bf16.ctypes.data_as(bf16_p),
        mean.ctypes.data_as(float_p), rstd.ctypes.data_as(float_p),
        1, dim, ctypes.c_float(float(config["epsilon"])),
        scratch_input.ctypes.data_as(float_p), scratch_output.ctypes.data_as(float_p),
    )
    output = from_bf16(output_bf16)
    mismatch = np.flatnonzero(output != reference)
    if mismatch.size == 0:
        raise RuntimeError("fixture no longer diverges; update its validated contract status")
    diff = np.abs(output - reference)
    worst = int(np.argmax(diff))
    report = {
        "classification": "REDUCTION_CONTRACT_MISMATCH",
        "mismatch_count": int(mismatch.size),
        "index": worst,
        "ck": float(output.flat[worst]),
        "pytorch": float(reference.flat[worst]),
        "max_abs": float(diff.flat[worst]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
