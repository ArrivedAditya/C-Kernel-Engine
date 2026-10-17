"""NEON parity tests for the Q6_K x Q8_K kernels on AArch64.

The NEON dot kernel (dot_q6_k_q8_k_neon in src/kernels/gemm_kernels_q6k_q8k.c)
is integer-exact per 256-value block but accumulates the FP32 block
contributions sequentially, whereas the scalar reference uses eight partial
lanes. These tests prove:

1. The default AArch64 dispatch actually selects the NEON provider (not a
   silent scalar fallback).
2. The NEON GEMV/GEMM results match an independent integer-exact NumPy oracle
   and the scalar C reference within a tight float-ordering tolerance.
3. CK_DEBUG_Q6K_Q8K_REF=1 forces the scalar reference (A/B escape hatch).

Runs only on aarch64/arm64; skipped elsewhere.
"""

from __future__ import annotations

import ctypes
import os
import platform
import struct
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "build" / "libckernel_engine.so"

QK_K = 256
BLOCK_Q6_K_SIZE = 210  # ql[128] + qh[64] + scales[16] + d[2]
BLOCK_Q8_K_SIZE = 292  # d[4] + qs[256] + bsums[32]

F32P = ctypes.POINTER(ctypes.c_float)

# NEON and the scalar reference differ only in FP32 block-accumulation order;
# the integer per-block dots are identical. Tolerances are therefore far below
# any quant-noise scale but above float reassociation noise.
RTOL = 1e-5
ATOL = 1e-4


def _fp32_to_fp16_bytes(f: float) -> bytes:
    return np.array([f], dtype=np.float16).tobytes()


def _random_q6k_blocks(rng: np.random.Generator, nb: int) -> bytes:
    """Random Q6_K super-blocks with realistic small FP16 scales."""
    out = bytearray()
    for _ in range(nb):
        out += rng.integers(0, 256, size=128, dtype=np.uint8).tobytes()  # ql
        out += rng.integers(0, 256, size=64, dtype=np.uint8).tobytes()   # qh
        out += rng.integers(-128, 128, size=16, dtype=np.int8).tobytes()  # scales
        out += _fp32_to_fp16_bytes(float(rng.uniform(0.005, 0.05)))       # d
    return bytes(out)


def _random_q8k_blocks(rng: np.random.Generator, nb: int) -> bytes:
    out = bytearray()
    for _ in range(nb):
        qs = rng.integers(-128, 128, size=QK_K, dtype=np.int8)
        out += struct.pack("<f", float(rng.uniform(0.005, 0.05)))         # d
        out += qs.tobytes()
        bsums = [int(qs[i * 16:(i + 1) * 16].sum()) for i in range(16)]
        out += np.array(bsums, dtype=np.int16).tobytes()
    return bytes(out)


def _fp16_to_fp32(b: bytes, off: int) -> float:
    return float(np.frombuffer(b[off:off + 2], dtype=np.float16)[0])


def _oracle_dot_q6k_q8k(w: bytes, x: bytes, K: int) -> float:
    """Independent integer-exact oracle: per-block int64 dot, then FP64 scale.

    Mirrors the Q6_K layout documented in ggml: within each 128-half, ql[l]
    holds the low nibbles of weights l and l+64, qh[l] holds the 2-bit high
    parts of weights l, l+32, l+64, l+96, and each int8 scale covers 16
    consecutive weights.
    """
    nb = K // QK_K
    total = np.float64(0.0)
    for i in range(nb):
        base_w = i * BLOCK_Q6_K_SIZE
        base_x = i * BLOCK_Q8_K_SIZE
        d = np.float64(_fp16_to_fp32(w, base_w + 208)) * np.float64(
            struct.unpack_from("<f", x, base_x)[0])
        ql = w[base_w:base_w + 128]
        qh = w[base_w + 128:base_w + 192]
        sc = np.frombuffer(w[base_w + 192:base_w + 208], dtype=np.int8)
        qs = np.frombuffer(x[base_x + 4:base_x + 260], dtype=np.int8)

        codes = np.empty(QK_K, dtype=np.int64)
        scales = np.empty(QK_K, dtype=np.int64)
        for n in range(2):  # two 128-weight halves
            for l in range(32):
                isub = l // 16
                b = n * 64  # ql offset for this half
                h = n * 32  # qh offset for this half
                idx = n * 128
                codes[idx + l] = (ql[b + l] & 0xF) | (((qh[h + l] >> 0) & 3) << 4)
                codes[idx + l + 32] = (ql[b + l + 32] & 0xF) | (((qh[h + l] >> 2) & 3) << 4)
                codes[idx + l + 64] = (ql[b + l] >> 4) | (((qh[h + l] >> 4) & 3) << 4)
                codes[idx + l + 96] = (ql[b + l + 32] >> 4) | (((qh[h + l] >> 6) & 3) << 4)
                scales[idx + l] = sc[n * 8 + isub]
                scales[idx + l + 32] = sc[n * 8 + isub + 2]
                scales[idx + l + 64] = sc[n * 8 + isub + 4]
                scales[idx + l + 96] = sc[n * 8 + isub + 6]
        q = codes - 32
        block_int = np.sum(q * scales * qs.astype(np.int64), dtype=np.int64)
        total += d * np.float64(block_int)
    return float(total)


def _buf(data: bytes) -> ctypes.Array:
    return ctypes.create_string_buffer(data, len(data))


@unittest.skipUnless(platform.machine() in ("aarch64", "arm64"),
                     "NEON parity path is AArch64-only")
@unittest.skipUnless(LIB_PATH.exists(), "engine library not built")
class TestQ6KQ8KNeonParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = ctypes.CDLL(str(LIB_PATH))
        cls.lib.gemv_q6_k_q8_k.argtypes = [F32P, ctypes.c_void_p,
                                           ctypes.c_void_p,
                                           ctypes.c_int, ctypes.c_int]
        cls.lib.gemv_q6_k_q8_k_ref.argtypes = cls.lib.gemv_q6_k_q8_k.argtypes
        cls.lib.gemv_q6_k_q8_k_parallel_simd.argtypes = [
            F32P, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        cls.lib.gemm_nt_q6_k_q8_k.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        cls.lib.ck_q6_k_q8_k_provider_name.argtypes = []
        cls.lib.ck_q6_k_q8_k_provider_name.restype = ctypes.c_char_p
        cls.rng = np.random.default_rng(20261011)

    def _make_case(self, M: int, K: int, N: int = 1):
        nb = K // QK_K
        rng = self.rng
        w = _random_q6k_blocks(rng, nb * N if N > 1 else nb * M)
        xs = [_random_q8k_blocks(rng, nb) for _ in range(max(M, N))]
        return w, xs

    def test_provider_name_reports_neon(self) -> None:
        name = self.lib.ck_q6_k_q8_k_provider_name().decode()
        self.assertEqual(name, "q6_k_q8_k_neon",
                         f"default AArch64 dispatch selected {name!r}, "
                         "expected the NEON provider")

    def test_forced_ref_escape_hatch(self) -> None:
        code = (
            "import ctypes;"
            "lib = ctypes.CDLL(%r);"
            "lib.ck_q6_k_q8_k_provider_name.restype = ctypes.c_char_p;"
            "print(lib.ck_q6_k_q8_k_provider_name().decode())"
            % str(LIB_PATH))
        env = dict(os.environ, CK_DEBUG_Q6K_Q8K_REF="1")
        out = subprocess.check_output([sys.executable, "-c", code],
                                      env=env, text=True).strip()
        self.assertEqual(out, "q6_k_q8_k_ref")

    def _check_gemv(self, M: int, K: int) -> None:
        nb = K // QK_K
        w, xs = self._make_case(M, K)
        wbuf = _buf(w)
        y_neon = np.zeros(M, dtype=np.float32)
        y_ref = np.zeros(M, dtype=np.float32)
        xbuf = _buf(xs[0])
        self.lib.gemv_q6_k_q8_k(
            y_neon.ctypes.data_as(F32P), wbuf, xbuf, M, K)
        self.lib.gemv_q6_k_q8_k_ref(
            y_ref.ctypes.data_as(F32P), wbuf, xbuf, M, K)
        np.testing.assert_allclose(
            y_neon, y_ref, rtol=RTOL, atol=ATOL,
            err_msg=f"NEON GEMV diverged from scalar ref (M={M}, K={K})")
        for row in range(M):
            w_row = w[row * nb * BLOCK_Q6_K_SIZE:
                      (row + 1) * nb * BLOCK_Q6_K_SIZE]
            oracle = _oracle_dot_q6k_q8k(w_row, xs[0], K)
            self.assertTrue(
                np.isclose(y_neon[row], oracle, rtol=1e-4, atol=ATOL),
                f"row {row}: NEON {y_neon[row]} vs oracle {oracle}")

    def test_gemv_neon_matches_ref_and_oracle(self) -> None:
        for M, K in ((1, QK_K), (4, 2 * QK_K), (5, 8 * QK_K)):
            with self.subTest(M=M, K=K):
                self._check_gemv(M, K)

    def test_parallel_simd_neon_matches_ref(self) -> None:
        M, K, nth = 9, 4 * QK_K, 4
        nb = K // QK_K
        w, xs = self._make_case(M, K)
        wbuf, xbuf = _buf(w), _buf(xs[0])
        y_neon = np.zeros(M, dtype=np.float32)
        y_ref = np.zeros(M, dtype=np.float32)
        for ith in range(nth):
            self.lib.gemv_q6_k_q8_k_parallel_simd(
                y_neon.ctypes.data_as(F32P), wbuf, xbuf, M, K, ith, nth)
        self.lib.gemv_q6_k_q8_k_ref(
            y_ref.ctypes.data_as(F32P), wbuf, xbuf, M, K)
        np.testing.assert_allclose(y_neon, y_ref, rtol=RTOL, atol=ATOL)

    def test_gemm_nt_neon_matches_ref(self) -> None:
        M, N, K = 3, 5, 2 * QK_K  # 3 activation rows, 5 weight rows
        nb = K // QK_K
        rng = self.rng
        w = _random_q6k_blocks(rng, nb * N)
        a = b"".join(_random_q8k_blocks(rng, nb) for _ in range(M))
        bias = np.linspace(-0.5, 0.5, N, dtype=np.float32)
        c = np.full((M, N), np.nan, dtype=np.float32)
        self.lib.gemm_nt_q6_k_q8_k(
            _buf(a), _buf(w),
            bias.ctypes.data_as(F32P),
            c.ctypes.data_as(F32P),
            M, N, K)
        self.assertFalse(np.isnan(c).any(), "GEMM left outputs unwritten")
        for m in range(M):
            a_row = a[m * nb * BLOCK_Q8_K_SIZE:(m + 1) * nb * BLOCK_Q8_K_SIZE]
            for n in range(N):
                w_row = w[n * nb * BLOCK_Q6_K_SIZE:(n + 1) * nb * BLOCK_Q6_K_SIZE]
                oracle = _oracle_dot_q6k_q8k(w_row, a_row, K) + float(bias[n])
                self.assertTrue(
                    np.isclose(c[m, n], oracle, rtol=1e-4, atol=ATOL),
                    f"C[{m},{n}]: NEON {c[m, n]} vs oracle {oracle}")


if __name__ == "__main__":
    unittest.main()
