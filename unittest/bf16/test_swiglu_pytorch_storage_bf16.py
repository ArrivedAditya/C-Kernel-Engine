#!/usr/bin/env python3
"""Exact PyTorch/SLEEF oracle for the BF16 SwiGLU storage contract."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]


def _bf16_values(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(values).to(torch.bfloat16).float().numpy()


def main() -> int:
    if not os.environ.get("CK_SLEEF_LIBRARY"):
        candidate = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cpu.so"
        if not candidate.is_file():
            raise RuntimeError(
                "CK_SLEEF_LIBRARY must identify a library exporting Sleef_expf16_u10"
            )
        os.environ["CK_SLEEF_LIBRARY"] = str(candidate)

    library = Path(os.environ.get("CK_ENGINE_SO", ROOT / "build" / "libckernel_engine.so"))
    kernel_library = ctypes.CDLL(str(library))
    kernel = kernel_library.swiglu_forward_pytorch_bf16_storage
    kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]

    rng = np.random.default_rng(43)
    finite_codes = np.arange(1 << 16, dtype=np.uint16)
    finite_values = (finite_codes.astype(np.uint32) << 16).view(np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    cases = [
        (
            "all_finite_bf16_gate",
            finite_values,
            np.resize(np.array([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=np.float32), finite_values.size),
        ),
        (
            "qwen3vl_intermediate_12288",
            rng.standard_normal(12288, dtype=np.float32) * 0.7,
            rng.standard_normal(12288, dtype=np.float32) * 0.7,
        ),
    ]

    for name, gate_values, up_values in cases:
        gate = _bf16_values(np.asarray(gate_values, dtype=np.float32))
        up = _bf16_values(np.asarray(up_values, dtype=np.float32))
        packed = np.concatenate((gate, up)).astype(np.float32, copy=False)
        actual = np.empty_like(gate)
        kernel(
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            1,
            gate.size,
        )
        expected = (F.silu(torch.from_numpy(gate).to(torch.bfloat16)) *
                    torch.from_numpy(up).to(torch.bfloat16)).float().numpy()
        mismatch = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
        if mismatch.size:
            index = int(mismatch[0])
            raise AssertionError(
                f"{name}: {mismatch.size}/{actual.size} mismatches; first={index} "
                f"gate={gate[index]!r} up={up[index]!r} "
                f"actual={actual[index]!r} expected={expected[index]!r}"
            )
        print(f"{name}: {actual.size}/{actual.size} byte-exact")

    print("BF16 PyTorch/SLEEF SwiGLU storage parity: 2/2 exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
