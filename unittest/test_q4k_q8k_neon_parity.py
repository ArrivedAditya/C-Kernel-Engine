"""NEON parity tests for Q4_K x Q8_K GEMV/GEMM and Q8_K row quantization.

Covers the AArch64 NEON paths added in src/kernels/gemm_kernels_q4k_q8k.c:

1. Default AArch64 dispatch selects the NEON providers (no silent scalar
   fallback), and the CK_DEBUG_Q4K_Q8_REF / CK_DEBUG_Q8K_REF escape hatches
   report and use the scalar reference.
2. quantize_row_q8_k_neon is BYTE-EXACT against quantize_row_q8_k_ref,
   including adversarial rounding ties and the all-zero block case. Q8_K
   bytes are a numerical ABI (see quantize_batch_q8_k_4row_nearest_even), so
   this is a memcmp, not a tolerance check.
3. The NEON Q4_K dot matches an independent integer-exact NumPy oracle and
   the scalar C reference.

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
BLOCK_Q4_K_SIZE = 144  # d[2] + dmin[2] + scales[12] + qs[128]
BLOCK_Q8_K_SIZE = 292  # d[4] + qs[256] + bsums[32]

F32P = ctypes.POINTER(ctypes.c_float)

# Integer arithmetic is identical between NEON and the scalar reference; only
# FP32 reassociation noise is possible.
RTOL = 1e-5
ATOL = 1e-4


def _random_q4k_blocks(rng: np.random.Generator, nb: int) -> bytes:
    out = bytearray()
    for _ in range(nb):
        out += np.array([rng.uniform(0.005, 0.05)], dtype=np.float16).tobytes()  # d
        out += np.array([rng.uniform(0.001, 0.02)], dtype=np.float16).tobytes()  # dmin
        out += rng.integers(0, 256, size=12, dtype=np.uint8).tobytes()           # scales
        out += rng.integers(0, 256, size=128, dtype=np.uint8).tobytes()          # qs
    return bytes(out)


def _random_q8k_blocks(rng: np.random.Generator, nb: int) -> bytes:
    out = bytearray()
    for _ in range(nb):
        qs = rng.integers(-128, 128, size=QK_K, dtype=np.int8)
        out += struct.pack("<f", float(rng.uniform(0.005, 0.05)))
        out += qs.tobytes()
        bsums = [int(qs[i * 16:(i + 1) * 16].sum()) for i in range(16)]
        out += np.array(bsums, dtype=np.int16).tobytes()
    return bytes(out)


def _unpack_q4k_scales(scales: bytes):
    s = list(scales)
    sc = [0] * 8
    m = [0] * 8
    for j in range(4):
        sc[j] = s[j] & 0x3F
        m[j] = s[j + 4] & 0x3F
    for j in range(4):
        sc[j + 4] = (s[j + 8] & 0x0F) | ((s[j] >> 6) << 4)
        m[j + 4] = (s[j + 8] >> 4) | ((s[j + 4] >> 6) << 4)
    return sc, m


def _fp16_at(b: bytes, off: int) -> float:
    return float(np.frombuffer(b[off:off + 2], dtype=np.float16)[0])


def _oracle_dot_q4k_q8k(w: bytes, x: bytes, K: int) -> float:
    """Independent integer-exact oracle for one Q4_K x Q8_K dot."""
    nb = K // QK_K
    total = np.float64(0.0)
    for i in range(nb):
        bw = i * BLOCK_Q4_K_SIZE
        bx = i * BLOCK_Q8_K_SIZE
        sc, m = _unpack_q4k_scales(w[bw + 4:bw + 16])
        qs_w = w[bw + 16:bw + 144]
        d = np.float64(_fp16_at(w, bw)) * np.float64(struct.unpack_from("<f", x, bx)[0])
        dmin = np.float64(_fp16_at(w, bw + 2)) * np.float64(struct.unpack_from("<f", x, bx)[0])
        bsums = np.frombuffer(x[bx + 260:bx + 292], dtype=np.int16).astype(np.int64)
        sumi = int(sum(int(bsums[j]) * m[j // 2] for j in range(16)))
        q8 = np.frombuffer(x[bx + 4:bx + 260], dtype=np.int8).astype(np.int64)
        scaled_sum = 0
        for group in range(4):
            qs = qs_w[group * 32:(group + 1) * 32]
            lo = sum((qs[l] & 0x0F) * int(q8[group * 64 + l]) for l in range(32))
            hi = sum((qs[l] >> 4) * int(q8[group * 64 + 32 + l]) for l in range(32))
            scaled_sum += sc[2 * group] * lo + sc[2 * group + 1] * hi
        total += d * np.float64(scaled_sum) - dmin * np.float64(sumi)
    return float(total)


def _buf(data: bytes) -> ctypes.Array:
    return ctypes.create_string_buffer(data, len(data))


def _provider_name_via_subprocess(symbol: str, env_var: str) -> str:
    code = (
        "import ctypes;"
        "lib = ctypes.CDLL(%r);"
        "f = lib.%s; f.restype = ctypes.c_char_p;"
        "print(f().decode())" % (str(LIB_PATH), symbol))
    env = dict(os.environ)
    env[env_var] = "1"
    return subprocess.check_output([sys.executable, "-c", code],
                                   env=env, text=True).strip()


@unittest.skipUnless(platform.machine() in ("aarch64", "arm64"),
                     "NEON parity path is AArch64-only")
@unittest.skipUnless(LIB_PATH.exists(), "engine library not built")
class TestQ4KQ8KNeonParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = ctypes.CDLL(str(LIB_PATH))
        cls.lib.gemv_q4_k_q8_k.argtypes = [F32P, ctypes.c_void_p,
                                           ctypes.c_void_p,
                                           ctypes.c_int, ctypes.c_int]
        cls.lib.gemv_q4_k_q8_k_ref.argtypes = cls.lib.gemv_q4_k_q8_k.argtypes
        cls.lib.gemm_nt_q4_k_q8_k.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        cls.lib.quantize_row_q8_k.argtypes = [F32P, ctypes.c_void_p,
                                              ctypes.c_int]
        cls.lib.quantize_row_q8_k_ref.argtypes = cls.lib.quantize_row_q8_k.argtypes
        cls.lib.quantize_row_q8_k_neon.argtypes = cls.lib.quantize_row_q8_k.argtypes
        cls.lib.ck_q4_k_q8_k_provider_name.restype = ctypes.c_char_p
        cls.lib.ck_quantize_row_q8_k_provider_name.restype = ctypes.c_char_p
        cls.rng = np.random.default_rng(20261017)
        # The GEMV NEON dot needs the dot-product extension (SDOT); baseline
        # armv8-a engine builds keep the scalar reference instead. The
        # quantize NEON path only needs plain NEON and is always present on
        # AArch64.
        cls.has_neon_gemv = (
            cls.lib.ck_q4_k_q8_k_provider_name().decode() == "q4_k_q8_k_neon")

    # ---- provider selection -------------------------------------------

    def test_provider_names_are_truthful(self) -> None:
        # Quantize only needs plain NEON: always NEON on AArch64.
        self.assertEqual(self.lib.ck_quantize_row_q8_k_provider_name().decode(),
                         "q8_k_quantize_neon")
        # GEMV reports NEON only when built with the dot-product extension
        # (e.g. CK_ARM_NATIVE=1 on GB10); otherwise it must honestly report
        # the scalar reference rather than claiming an unused kernel.
        name = self.lib.ck_q4_k_q8_k_provider_name().decode()
        self.assertIn(name, ("q4_k_q8_k_neon", "q4_k_q8_k_ref"))

    def test_forced_ref_escape_hatches(self) -> None:
        if self.has_neon_gemv:
            self.assertEqual(
                _provider_name_via_subprocess("ck_q4_k_q8_k_provider_name",
                                              "CK_DEBUG_Q4K_Q8_REF"),
                "q4_k_q8_k_ref")
        self.assertEqual(
            _provider_name_via_subprocess("ck_quantize_row_q8_k_provider_name",
                                          "CK_DEBUG_Q8K_REF"),
            "q8_k_quantize_ref")

    # ---- quantize_row_q8_k: byte-exact ---------------------------------

    def _check_quantize_bytes(self, x: np.ndarray, label: str) -> None:
        k = x.size
        nb = k // QK_K
        y_neon = ctypes.create_string_buffer(nb * BLOCK_Q8_K_SIZE)
        y_ref = ctypes.create_string_buffer(nb * BLOCK_Q8_K_SIZE)
        xp = x.ctypes.data_as(F32P)
        self.lib.quantize_row_q8_k(xp, y_neon, k)       # default == NEON here
        self.lib.quantize_row_q8_k_ref(xp, y_ref, k)
        self.assertEqual(y_neon.raw, y_ref.raw,
                         f"NEON quantize not byte-exact vs ref ({label})")

    def test_quantize_random_rows_byte_exact(self) -> None:
        rng = self.rng
        for k in (QK_K, 4 * QK_K):
            for _ in range(8):
                x = rng.standard_normal(k).astype(np.float32) * 0.1
                self._check_quantize_bytes(x, f"random k={k}")

    def test_quantize_adversarial_byte_exact(self) -> None:
        rng = self.rng
        # max = -127 forces iscale == 1.0, exposing exact .5 rounding ties
        ties = np.full(QK_K, 0.0, dtype=np.float32)
        ties[0] = -127.0
        ties[1:9] = [2.5, -2.5, 3.5, -3.5, 0.5, -0.5, 126.5, -126.5]
        self._check_quantize_bytes(ties, "rounding ties")
        # exact +/- amax pair in both orders (first-occurrence tie-break)
        for sign in (1.0, -1.0):
            x = rng.standard_normal(QK_K).astype(np.float32) * 0.01
            x[3] = 5.0 * sign
            x[200] = -5.0 * sign
            self._check_quantize_bytes(x, f"amax pair sign={sign}")
        # all-zero block
        self._check_quantize_bytes(np.zeros(QK_K, dtype=np.float32), "zeros")
        # tiny denormal-ish magnitudes
        x = (rng.standard_normal(2 * QK_K) * 1e-30).astype(np.float32)
        self._check_quantize_bytes(x, "tiny magnitudes")

    # ---- gemv / gemm_nt: integer-exact oracle --------------------------

    def _check_gemv(self, M: int, K: int) -> None:
        rng = self.rng
        nb = K // QK_K
        w = _random_q4k_blocks(rng, nb * M)
        x = _random_q8k_blocks(rng, nb)
        y_neon = np.zeros(M, dtype=np.float32)
        y_ref = np.zeros(M, dtype=np.float32)
        self.lib.gemv_q4_k_q8_k(y_neon.ctypes.data_as(F32P), _buf(w),
                                _buf(x), M, K)
        self.lib.gemv_q4_k_q8_k_ref(y_ref.ctypes.data_as(F32P), _buf(w),
                                    _buf(x), M, K)
        np.testing.assert_allclose(y_neon, y_ref, rtol=RTOL, atol=ATOL,
                                   err_msg=f"NEON GEMV vs ref (M={M}, K={K})")
        for row in range(M):
            w_row = w[row * nb * BLOCK_Q4_K_SIZE:
                      (row + 1) * nb * BLOCK_Q4_K_SIZE]
            oracle = _oracle_dot_q4k_q8k(w_row, x, K)
            self.assertTrue(
                np.isclose(y_neon[row], oracle, rtol=1e-4, atol=ATOL),
                f"row {row}: NEON {y_neon[row]} vs oracle {oracle}")

    def test_gemv_neon_matches_ref_and_oracle(self) -> None:
        if not self.has_neon_gemv:
            self.skipTest("engine built without __ARM_FEATURE_DOTPROD; "
                          "rebuild with CK_ARM_NATIVE=1 to exercise NEON")
        for M, K in ((1, QK_K), (4, 2 * QK_K), (5, 8 * QK_K)):
            with self.subTest(M=M, K=K):
                self._check_gemv(M, K)

    def test_gemm_nt_neon_matches_oracle(self) -> None:
        if not self.has_neon_gemv:
            self.skipTest("engine built without __ARM_FEATURE_DOTPROD; "
                          "rebuild with CK_ARM_NATIVE=1 to exercise NEON")
        M, N, K = 3, 5, 2 * QK_K  # 3 activation rows, 5 weight rows
        rng = self.rng
        nb = K // QK_K
        w = _random_q4k_blocks(rng, nb * N)
        a = b"".join(_random_q8k_blocks(rng, nb) for _ in range(M))
        bias = np.linspace(-0.5, 0.5, N, dtype=np.float32)
        c = np.full((M, N), np.nan, dtype=np.float32)
        self.lib.gemm_nt_q4_k_q8_k(_buf(a), _buf(w),
                                   bias.ctypes.data_as(F32P),
                                   c.ctypes.data_as(F32P),
                                   M, N, K)
        self.assertFalse(np.isnan(c).any(), "GEMM left outputs unwritten")
        for m in range(M):
            a_row = a[m * nb * BLOCK_Q8_K_SIZE:(m + 1) * nb * BLOCK_Q8_K_SIZE]
            for n in range(N):
                w_row = w[n * nb * BLOCK_Q4_K_SIZE:(n + 1) * nb * BLOCK_Q4_K_SIZE]
                oracle = _oracle_dot_q4k_q8k(w_row, a_row, K) + float(bias[n])
                self.assertTrue(
                    np.isclose(c[m, n], oracle, rtol=1e-4, atol=ATOL),
                    f"C[{m},{n}]: NEON {c[m, n]} vs oracle {oracle}")


if __name__ == "__main__":
    unittest.main()
