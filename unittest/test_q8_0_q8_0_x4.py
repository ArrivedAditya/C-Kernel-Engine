import ctypes
import os
import struct
from pathlib import Path

import numpy as np


def _block(scale: float, values: np.ndarray) -> bytes:
    return struct.pack("<e", scale) + values.astype(np.int8).tobytes()


def test_q8_0_x4_matches_reference_provider_bit_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    lib = ctypes.CDLL(
        str(root / "build" / "libckernel_engine.so"),
        mode=getattr(os, "RTLD_LOCAL", 0),
    )
    signature = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemv_q8_0_q8_0.argtypes = signature
    lib.gemv_q8_0_q8_0_x4.argtypes = signature

    rng = np.random.default_rng(0x5136)
    rows, width = 19, 96
    blocks = width // 32
    weights = b"".join(
        _block(
            float(rng.uniform(0.001, 0.05)),
            rng.integers(-127, 128, size=32),
        )
        for _ in range(rows * blocks)
    )
    activation = b"".join(
        _block(
            float(rng.uniform(0.001, 0.05)),
            rng.integers(-127, 128, size=32),
        )
        for _ in range(blocks)
    )
    w_buf = ctypes.create_string_buffer(weights)
    x_buf = ctypes.create_string_buffer(activation)
    reference = (ctypes.c_float * rows)()
    candidate = (ctypes.c_float * rows)()

    lib.gemv_q8_0_q8_0(reference, w_buf, x_buf, rows, width)
    lib.gemv_q8_0_q8_0_x4(candidate, w_buf, x_buf, rows, width)

    assert bytes(reference) == bytes(candidate)
