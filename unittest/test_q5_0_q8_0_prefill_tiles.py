import ctypes
import os
import struct
from pathlib import Path

import numpy as np


def _q8_block(scale: float, values: np.ndarray) -> bytes:
    return struct.pack("<e", scale) + values.astype(np.int8).tobytes()


def _q5_block(scale: float, rng: np.random.Generator) -> bytes:
    return (
        struct.pack("<e", scale)
        + rng.integers(0, 256, size=4, dtype=np.uint8).tobytes()
        + rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
    )


def test_q5_0_prefill_tiles_match_established_gemm_bit_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    lib = ctypes.CDLL(
        str(root / "build" / "libckernel_engine.so"),
        mode=getattr(os, "RTLD_LOCAL", 0),
    )
    signature = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemm_nt_q5_0_q8_0.argtypes = signature
    lib.gemm_nt_q5_0_q8_0_m2n4.argtypes = signature
    lib.gemm_nt_q5_0_q8_0_m4n2.argtypes = signature
    lib.gemm_nt_q5_0_q8_0_m4n2_tile.argtypes = signature + [ctypes.c_int]

    rng = np.random.default_rng(0x5508)
    for tokens, rows, width in ((1, 1, 32), (3, 7, 96), (5, 19, 160), (8, 32, 256)):
        blocks = width // 32
        weights = b"".join(
            _q5_block(float(rng.uniform(0.001, 0.05)), rng)
            for _ in range(rows * blocks)
        )
        activations = b"".join(
            _q8_block(
                float(rng.uniform(0.001, 0.05)),
                rng.integers(-127, 128, size=32),
            )
            for _ in range(tokens * blocks)
        )
        bias_values = rng.uniform(-0.25, 0.25, size=rows).astype(np.float32)
        w_buf = ctypes.create_string_buffer(weights)
        a_buf = ctypes.create_string_buffer(activations)
        bias = (ctypes.c_float * rows)(*bias_values)
        reference = (ctypes.c_float * (tokens * rows))()
        m2n4 = (ctypes.c_float * (tokens * rows))()
        m4n2 = (ctypes.c_float * (tokens * rows))()
        tiled = (ctypes.c_float * (tokens * rows))()

        lib.gemm_nt_q5_0_q8_0(a_buf, w_buf, bias, reference, tokens, rows, width)
        lib.gemm_nt_q5_0_q8_0_m2n4(a_buf, w_buf, bias, m2n4, tokens, rows, width)
        lib.gemm_nt_q5_0_q8_0_m4n2(a_buf, w_buf, bias, m4n2, tokens, rows, width)

        weight_row_bytes = blocks * 22
        for n0 in range(0, rows, 8):
            tile_rows = min(8, rows - n0)
            lib.gemm_nt_q5_0_q8_0_m4n2_tile(
                a_buf,
                ctypes.c_void_p(ctypes.addressof(w_buf) + n0 * weight_row_bytes),
                ctypes.cast(ctypes.byref(bias, n0 * 4), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(ctypes.byref(tiled, n0 * 4), ctypes.POINTER(ctypes.c_float)),
                tokens,
                tile_rows,
                width,
                rows,
            )

        assert bytes(reference) == bytes(m2n4)
        assert bytes(reference) == bytes(m4n2)
        assert bytes(reference) == bytes(tiled)
