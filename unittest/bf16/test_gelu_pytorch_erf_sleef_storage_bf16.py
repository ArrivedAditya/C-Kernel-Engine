#!/usr/bin/env python3
"""Exact PyTorch x86 oracle for BF16 GELU (approximate='none')."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(os.environ.get("CK_ENGINE_SO", str(ROOT / "build" / "libckernel_engine.so")))
KERNEL = LIB.gelu_pytorch_erf_sleef_bf16_storage
KERNEL.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]


def main() -> int:
    if not os.environ.get("CK_SLEEF_LIBRARY"):
        candidate = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cpu.so"
        if not candidate.is_file():
            raise RuntimeError("CK_SLEEF_LIBRARY must identify a library exporting Sleef_expf16_u10")
        os.environ["CK_SLEEF_LIBRARY"] = str(candidate)

    rng = np.random.default_rng(29)
    codes = np.arange(1 << 16, dtype=np.uint16)
    exhaustive = (codes.astype(np.uint32) << 16).view(np.float32)
    exhaustive = exhaustive[np.isfinite(exhaustive)]
    cases = [
        ("all_finite_bf16", exhaustive),
        ("vision_projector_width", rng.standard_normal((3, 4608), dtype=np.float32) * 3.0),
    ]
    for name, values in cases:
        source = torch.from_numpy(values).to(torch.bfloat16).float().numpy()
        actual = source.copy()
        KERNEL(actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), actual.size)
        expected = torch.nn.functional.gelu(torch.from_numpy(source).to(torch.bfloat16)).float().numpy()
        mismatch = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
        if mismatch.size:
            i = int(mismatch[0])
            raise AssertionError(
                f"{name}: {mismatch.size}/{actual.size} mismatches; first={i} "
                f"input={source[i]!r} actual={actual[i]!r} expected={expected[i]!r}"
            )
        print(f"{name}: {actual.size}/{actual.size} byte-exact")
    print("BF16 PyTorch-erf SLEEF GELU storage parity: 2/2 exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
