"""NEON parity tests for the Q5_K x Q8_K dot on AArch64.

Covers dot_q5_k_q8_k_row_neon_sdot in src/kernels/gemm_kernels_q5_k.c
(gated on __ARM_FEATURE_DOTPROD):

1. Provider truthfulness: ck_q5_k_provider_name reports
   "q5_k_q8_k_neon_sdot" on dot-product builds and "q5_k_q8_k_ref" on
   baseline armv8-a builds; the existing CK_DEBUG_Q5K_GENERIC_DOT=1 and
   CK_DEBUG_Q5K_FP32_FALLBACK=1 hatches are honored and reported.
2. The dispatched GEMV/GEMM match an independent integer-exact NumPy oracle
   (super-block scale/min math included) at model-realistic K (Qwen3.5-0.8B:
   1024, 3584).
3. The SDOT path matches the generic scalar path (CK_DEBUG_Q5K_GENERIC_DOT=1
   subprocess) tightly: integer math is exact per super-block; only the
   FP32 lane grouping differs (8 live consecutive-position lanes vs the
   scalar's strided positions - documented in the kernel comment).
4. Adversarial 5th-bit patterns (all-zeros/all-ones qh) exercise the high-bit
   spread.

Runs only on aarch64/arm64; skipped elsewhere. CK_REQUIRE_Q5K_NEON=1 (set by
the nightly dotprod step) fails closed when the SDOT provider is not selected.
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
BLOCK_Q5_K_SIZE = 176  # d[2] + dmin[2] + scales[12] + qh[32] + qs[128]
BLOCK_Q8_K_SIZE = 292  # d[4] + qs[256] + bsums[32]

F32P = ctypes.POINTER(ctypes.c_float)

# Integer arithmetic is exact; only FP32 lane grouping can differ between the
# SDOT and generic-scalar paths.
RTOL = 1e-5
ATOL = 1e-4


def _random_q5k_blocks(rng: np.random.Generator, nb: int,
                       qh_pattern: str | None = None) -> bytes:
    out = bytearray()
    for _ in range(nb):
        out += np.array([rng.uniform(0.005, 0.05)], dtype=np.float16).tobytes()  # d
        out += np.array([rng.uniform(0.001, 0.02)], dtype=np.float16).tobytes()  # dmin
        out += rng.integers(0, 256, size=12, dtype=np.uint8).tobytes()           # scales
        if qh_pattern == "zeros":
            out += bytes(32)
        elif qh_pattern == "ones":
            out += b"\xff" * 32
        else:
            out += rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()       # qh
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


def _unpack_q5k_scales(scales: bytes):
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


def _oracle_dot_q5k_q8k(w: bytes, x: bytes, K: int) -> float:
    """Independent integer-exact oracle for one Q5_K x Q8_K dot.

    Q5_K layout (ggml-compatible): sub-block sb (32 weights) uses the low
    nibble of qs[(sb//2)*32 + l] when sb is even, the high nibble when odd,
    plus bit sb of qh[l] as the 5th bit. Codes are unsigned [0,31]; the min
    term is sum_j bsums[j] * mins[j//2].
    """
    nb = K // QK_K
    total = np.float64(0.0)
    for i in range(nb):
        bw = i * BLOCK_Q5_K_SIZE
        bx = i * BLOCK_Q8_K_SIZE
        sc, mins = _unpack_q5k_scales(w[bw + 4:bw + 16])
        qh = w[bw + 16:bw + 48]
        qs_w = w[bw + 48:bw + 176]
        d = np.float64(_fp16_at(w, bw)) * np.float64(struct.unpack_from("<f", x, bx)[0])
        dmin = np.float64(_fp16_at(w, bw + 2)) * np.float64(struct.unpack_from("<f", x, bx)[0])
        bsums = np.frombuffer(x[bx + 260:bx + 292], dtype=np.int16).astype(np.int64)
        sumi_mins = int(sum(int(bsums[j]) * mins[j // 2] for j in range(16)))
        q8 = np.frombuffer(x[bx + 4:bx + 260], dtype=np.int8).astype(np.int64)
        sumi = 0
        for sb in range(8):
            base = (sb // 2) * 32
            bit = 1 << sb
            acc = 0
            for l in range(32):
                nib = (qs_w[base + l] >> 4) if (sb & 1) else (qs_w[base + l] & 0x0F)
                code = nib + (16 if (qh[l] & bit) else 0)
                acc += code * int(q8[sb * 32 + l])
            sumi += sc[sb] * acc
        total += d * np.float64(sumi) - dmin * np.float64(sumi_mins)
    return float(total)


def _buf(data: bytes) -> ctypes.Array:
    return ctypes.create_string_buffer(data, len(data))


def _provider_name_via_subprocess(env_var: str) -> str:
    code = (
        "import ctypes;"
        "lib = ctypes.CDLL(%r);"
        "f = lib.ck_q5_k_provider_name; f.restype = ctypes.c_char_p;"
        "print(f().decode())" % (str(LIB_PATH),))
    env = dict(os.environ)
    env[env_var] = "1"
    return subprocess.check_output([sys.executable, "-c", code],
                                   env=env, text=True).strip()


def _gemv_generic_dot_via_subprocess(w: bytes, x: bytes, M: int, K: int) -> np.ndarray:
    """Run gemv_q5_k_q8_k with CK_DEBUG_Q5K_GENERIC_DOT=1 in a subprocess."""
    code = (
        "import ctypes, sys;"
        "import numpy as np;"
        "lib = ctypes.CDLL(%r);"
        "F32P = ctypes.POINTER(ctypes.c_float);"
        "lib.gemv_q5_k_q8_k.argtypes = [F32P, ctypes.c_void_p, ctypes.c_void_p,"
        " ctypes.c_int, ctypes.c_int];"
        "w = open(sys.argv[1], 'rb').read(); x = open(sys.argv[2], 'rb').read();"
        "y = np.zeros(%d, dtype=np.float32);"
        "lib.gemv_q5_k_q8_k(y.ctypes.data_as(F32P),"
        " ctypes.create_string_buffer(w, len(w)),"
        " ctypes.create_string_buffer(x, len(x)), %d, %d);"
        "sys.stdout.write(y.tobytes().hex())"
        % (str(LIB_PATH), M, M, K))
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as fw:
        fw.write(w)
        wpath = fw.name
    with tempfile.NamedTemporaryFile(delete=False) as fx:
        fx.write(x)
        xpath = fx.name
    try:
        env = dict(os.environ, CK_DEBUG_Q5K_GENERIC_DOT="1")
        out = subprocess.check_output([sys.executable, "-c", code, wpath, xpath],
                                      env=env)
        return np.frombuffer(bytes.fromhex(out.decode()), dtype=np.float32).copy()
    finally:
        os.unlink(wpath)
        os.unlink(xpath)


@unittest.skipUnless(platform.machine() in ("aarch64", "arm64"),
                     "NEON parity path is AArch64-only")
@unittest.skipUnless(LIB_PATH.exists(), "engine library not built")
class TestQ5KQ8KNeonParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = ctypes.CDLL(str(LIB_PATH))
        cls.lib.gemv_q5_k_q8_k.argtypes = [F32P, ctypes.c_void_p,
                                           ctypes.c_void_p,
                                           ctypes.c_int, ctypes.c_int]
        cls.lib.gemm_nt_q5_k_q8_k.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        cls.lib.ck_q5_k_provider_name.restype = ctypes.c_char_p
        cls.rng = np.random.default_rng(20261024)
        cls.provider = cls.lib.ck_q5_k_provider_name().decode()
        cls.has_sdot = cls.provider == "q5_k_q8_k_neon_sdot"
        # Optimized-lane enforcement: when CK_REQUIRE_Q5K_NEON=1 (set by the
        # nightly dotprod build step), failing to select the SDOT provider is
        # a hard failure, not a skip.
        cls.require_neon = os.environ.get("CK_REQUIRE_Q5K_NEON", "") not in (
            "", "0")

    def _require_or_skip_sdot(self) -> None:
        if self.has_sdot:
            return
        if self.require_neon:
            self.fail("CK_REQUIRE_Q5K_NEON=1 but the engine dispatches "
                      "gemv_q5_k_q8_k to %r - the optimized lane did not "
                      "execute the intended SDOT NEON provider" % self.provider)
        self.skipTest("engine built without __ARM_FEATURE_DOTPROD; "
                      "rebuild with CK_ARM_ARCH or CK_ARM_NATIVE=1 to "
                      "exercise the SDOT provider")

    # ---- provider selection -------------------------------------------

    def test_provider_names_are_truthful(self) -> None:
        # Dispatch selects SDOT on dot-product builds and the generic scalar
        # dot otherwise; it must honestly report which one.
        self.assertIn(self.provider, ("q5_k_q8_k_neon_sdot", "q5_k_q8_k_ref"))

    def test_forced_ref_escape_hatches(self) -> None:
        self.assertEqual(
            _provider_name_via_subprocess("CK_DEBUG_Q5K_GENERIC_DOT"),
            "q5_k_q8_k_ref")
        self.assertEqual(
            _provider_name_via_subprocess("CK_DEBUG_Q5K_FP32_FALLBACK"),
            "q5_k_fp32_ref")

    def test_require_neon_fail_closed(self) -> None:
        if not self.require_neon:
            self.skipTest("CK_REQUIRE_Q5K_NEON not set")
        self.assertEqual(self.provider, "q5_k_q8_k_neon_sdot",
                         "optimized lane must select the SDOT provider")

    # ---- gemv / gemm_nt: integer-exact oracle --------------------------

    def _check_gemv(self, M: int, K: int, qh_pattern: str | None = None) -> None:
        rng = self.rng
        nb = K // QK_K
        w = _random_q5k_blocks(rng, nb * M, qh_pattern)
        x = _random_q8k_blocks(rng, nb)
        y = np.zeros(M, dtype=np.float32)
        self.lib.gemv_q5_k_q8_k(y.ctypes.data_as(F32P), _buf(w), _buf(x), M, K)
        label = f"M={M} K={K} qh={qh_pattern or 'random'}"
        for row in range(M):
            w_row = w[row * nb * BLOCK_Q5_K_SIZE:(row + 1) * nb * BLOCK_Q5_K_SIZE]
            oracle = _oracle_dot_q5k_q8k(w_row, x, K)
            self.assertTrue(
                np.isclose(y[row], oracle, rtol=1e-4, atol=ATOL),
                f"{label} row {row}: dispatch {y[row]} vs oracle {oracle}")
        # A/B against the generic scalar dot (subprocess, same inputs):
        # integer math is exact; only FP32 lane grouping can differ.
        y_generic = _gemv_generic_dot_via_subprocess(w, x, M, K)
        np.testing.assert_allclose(y, y_generic, rtol=RTOL, atol=ATOL,
                                   err_msg=f"dispatch vs generic dot ({label})")

    def test_gemv_matches_oracle_and_generic(self) -> None:
        # Block counts 1, 2 and Qwen3.5-0.8B-realistic K = 1024/2048/3584.
        for M, K in ((1, QK_K), (4, 2 * QK_K), (5, 1024), (3, 2048), (2, 3584)):
            with self.subTest(M=M, K=K):
                self._check_gemv(M, K)

    def test_gemv_high_bit_patterns(self) -> None:
        for pattern in ("zeros", "ones"):
            with self.subTest(qh=pattern):
                self._check_gemv(3, 2 * QK_K, pattern)

    def test_gemm_nt_matches_oracle(self) -> None:
        self._require_or_skip_sdot()
        M, N, K = 3, 5, 2 * QK_K  # 3 activation rows, 5 weight rows
        rng = self.rng
        nb = K // QK_K
        w = _random_q5k_blocks(rng, nb * N)
        a = b"".join(_random_q8k_blocks(rng, nb) for _ in range(M))
        bias = np.linspace(-0.5, 0.5, N, dtype=np.float32)
        c = np.full((M, N), np.nan, dtype=np.float32)
        self.lib.gemm_nt_q5_k_q8_k(_buf(a), _buf(w),
                                   bias.ctypes.data_as(F32P),
                                   c.ctypes.data_as(F32P), M, N, K)
        self.assertFalse(np.isnan(c).any(), "GEMM left outputs unwritten")
        for m in range(M):
            a_row = a[m * nb * BLOCK_Q8_K_SIZE:(m + 1) * nb * BLOCK_Q8_K_SIZE]
            for n in range(N):
                w_row = w[n * nb * BLOCK_Q5_K_SIZE:(n + 1) * nb * BLOCK_Q5_K_SIZE]
                oracle = _oracle_dot_q5k_q8k(w_row, a_row, K) + float(bias[n])
                self.assertTrue(
                    np.isclose(c[m, n], oracle, rtol=1e-4, atol=ATOL),
                    f"C[{m},{n}]: GEMM {c[m, n]} vs oracle {oracle}")


if __name__ == "__main__":
    unittest.main()
