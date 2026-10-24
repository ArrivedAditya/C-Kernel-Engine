"""NEON parity tests for Q5_1 x Q8_1 on AArch64.

Covers the NEON paths in src/kernels/gemm_kernels_q5_1_q8_1.c:

1. quantize_row_q8_1_neon (AArch64) is BYTE-EXACT against the scalar Q8_1
   contract (independent NumPy model with roundf ties-away semantics);
   Q8_1 bytes are a numerical ABI on this path, so this is a memcmp.
2. dot_q5_1_q8_1_block_neon_sdot (gated on __ARM_FEATURE_DOTPROD) matches an
   independent integer-exact NumPy oracle and the forced-scalar A/B
   (CK_DEBUG_Q5_1_Q8_1_REF=1 subprocess) - the int8 x int8 dot is exact in
   int32 and the FP32 epilogue is identical, so the A/B is bit-exact.
3. Provider truthfulness: ck_q5_1_q8_1_provider_name /
   ck_quantize_row_q8_1_provider_name report what the dispatchers actually
   select, including the CK_DEBUG_Q5_1_Q8_1_REF hatch.
4. Adversarial 5th-bit patterns (all-zeros/all-ones/alternating qh) exercise
   the high-bit spread; model-realistic K for Gemma3-270M (640, 2048).

Runs only on aarch64/arm64; skipped elsewhere. CK_REQUIRE_Q5_1_NEON=1 (set by
the nightly dotprod step) fails closed when the SDOT provider is not selected.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "build" / "libckernel_engine.so"

QK = 32
BLOCK_Q5_1_SIZE = 24  # d[2] + m[2] + qh[4] + qs[16]
BLOCK_Q8_1_SIZE = 36  # d[2] + s[2] + qs[32]

F32P = ctypes.POINTER(ctypes.c_float)

RTOL = 1e-5
ATOL = 1e-4


def _fp16_bytes(f: float) -> bytes:
    return np.array([f], dtype=np.float16).tobytes()


def _fp16_at(b: bytes, off: int) -> float:
    return float(np.frombuffer(b[off:off + 2], dtype=np.float16)[0])


def _q5_1_block(rng: np.random.Generator, qh_pattern: str | None = None) -> bytes:
    out = bytearray()
    out += _fp16_bytes(float(rng.uniform(0.005, 0.05)))                    # d
    out += _fp16_bytes(float(rng.uniform(0.0, 0.02)))                      # m
    if qh_pattern == "zeros":
        out += bytes(4)
    elif qh_pattern == "ones":
        out += b"\xff" * 4
    elif qh_pattern == "alt_lo":
        out += (0x55555555).to_bytes(4, "little")
    elif qh_pattern == "alt_hi":
        out += (0xAAAAAAAA).to_bytes(4, "little")
    else:
        out += rng.integers(0, 256, size=4, dtype=np.uint8).tobytes()      # qh
    out += rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()         # qs
    return bytes(out)


def _random_q5_1_blocks(rng: np.random.Generator, nb: int,
                        qh_pattern: str | None = None) -> bytes:
    return b"".join(_q5_1_block(rng, qh_pattern) for _ in range(nb))


def _random_q8_1_blocks(rng: np.random.Generator, nb: int) -> bytes:
    out = bytearray()
    for _ in range(nb):
        qs = rng.integers(-128, 128, size=QK, dtype=np.int8)
        d = float(rng.uniform(0.005, 0.05))
        out += _fp16_bytes(d)
        out += _fp16_bytes(float(qs.sum()) * d)                            # s
        out += qs.tobytes()
    return bytes(out)


def _oracle_dot_q5_1_q8_1(w: bytes, x: bytes, K: int) -> float:
    """Independent integer-exact oracle for one Q5_1 x Q8_1 dot.

    Q5_1 layout (ggml-compatible): weight j uses the low nibble of qs[j] plus
    bit j of qh; weight j+16 uses the high nibble of qs[j] plus bit (j+16) of
    qh. Codes are unsigned [0,31]; block result is (wd*xd)*sumi + wm*xs.
    """
    nb = K // QK
    total = np.float64(0.0)
    for i in range(nb):
        bw = i * BLOCK_Q5_1_SIZE
        bx = i * BLOCK_Q8_1_SIZE
        wd = np.float64(_fp16_at(w, bw))
        wm = np.float64(_fp16_at(w, bw + 2))
        xd = np.float64(_fp16_at(x, bx))
        xs = np.float64(_fp16_at(x, bx + 2))
        qh = int.from_bytes(w[bw + 4:bw + 8], "little")
        qs = w[bw + 8:bw + 24]
        q8 = np.frombuffer(x[bx + 4:bx + 36], dtype=np.int8).astype(np.int64)
        sumi = 0
        for j in range(QK // 2):
            q0 = (qs[j] & 0x0F) | (((qh >> j) & 1) << 4)
            q1 = (qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)
            sumi += q0 * int(q8[j]) + q1 * int(q8[j + QK // 2])
        total += wd * xd * np.float64(sumi) + wm * xs
    return float(total)


def _oracle_quantize_q8_1(x: np.ndarray) -> bytes:
    """NumPy model of the scalar Q8_1 contract (byte-exact target).

    roundf is round-half-away-from-zero on fp32; |x*id| <= 127 so
    sign*floor(|v|+0.5) in fp32 is exact.
    """
    x = np.asarray(x, dtype=np.float32)
    nb = x.size // QK
    out = bytearray()
    for b in range(nb):
        xb = x[b * QK:(b + 1) * QK]
        amax = np.float32(0.0)
        for v in xb:
            av = v if v >= 0.0 else -v
            if av > amax:
                amax = av
        d = np.float32(amax / np.float32(127.0))
        idv = np.float32(1.0) / d if d != 0.0 else np.float32(0.0)
        out += np.array([d], dtype=np.float16).tobytes()
        v = xb * idv
        q = (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.int32)
        qs = q.astype(np.int8)
        out += np.array([np.float32(qs.sum()) * d], dtype=np.float16).tobytes()
        out += qs.tobytes()
    return bytes(out)


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


def _gemv_forced_ref_via_subprocess(w: bytes, x: bytes, M: int, K: int) -> np.ndarray:
    """Run gemv_q5_1_q8_1_ref with CK_DEBUG_Q5_1_Q8_1_REF=1 (scalar dot)."""
    code = (
        "import ctypes, sys;"
        "import numpy as np;"
        "lib = ctypes.CDLL(%r);"
        "F32P = ctypes.POINTER(ctypes.c_float);"
        "lib.gemv_q5_1_q8_1_ref.argtypes = [F32P, ctypes.c_void_p, ctypes.c_void_p,"
        " ctypes.c_int, ctypes.c_int];"
        "w = open(sys.argv[1], 'rb').read(); x = open(sys.argv[2], 'rb').read();"
        "y = np.zeros(%d, dtype=np.float32);"
        "lib.gemv_q5_1_q8_1_ref(y.ctypes.data_as(F32P),"
        " ctypes.create_string_buffer(w, len(w)),"
        " ctypes.create_string_buffer(x, len(x)), %d, %d);"
        "sys.stdout.write(y.tobytes().hex())"
        % (str(LIB_PATH), M, M, K))
    with tempfile.NamedTemporaryFile(delete=False) as fw:
        fw.write(w)
        wpath = fw.name
    with tempfile.NamedTemporaryFile(delete=False) as fx:
        fx.write(x)
        xpath = fx.name
    try:
        env = dict(os.environ, CK_DEBUG_Q5_1_Q8_1_REF="1")
        out = subprocess.check_output([sys.executable, "-c", code, wpath, xpath],
                                      env=env)
        return np.frombuffer(bytes.fromhex(out.decode()), dtype=np.float32).copy()
    finally:
        os.unlink(wpath)
        os.unlink(xpath)


@unittest.skipUnless(platform.machine() in ("aarch64", "arm64"),
                     "NEON parity path is AArch64-only")
@unittest.skipUnless(LIB_PATH.exists(), "engine library not built")
class TestQ51Q81NeonParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = ctypes.CDLL(str(LIB_PATH))
        cls.lib.quantize_row_q8_1.argtypes = [F32P, ctypes.c_void_p,
                                              ctypes.c_int]
        cls.lib.gemv_q5_1_q8_1.argtypes = [F32P, ctypes.c_void_p, F32P,
                                           ctypes.c_int, ctypes.c_int]
        cls.lib.gemv_q5_1_q8_1_ref.argtypes = [F32P, ctypes.c_void_p,
                                               ctypes.c_void_p,
                                               ctypes.c_int, ctypes.c_int]
        cls.lib.gemm_nt_q5_1_q8_1_m2.argtypes = [
            F32P, ctypes.c_void_p, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        cls.lib.ck_q5_1_q8_1_provider_name.restype = ctypes.c_char_p
        cls.lib.ck_quantize_row_q8_1_provider_name.restype = ctypes.c_char_p
        cls.rng = np.random.default_rng(20261024)
        cls.provider = cls.lib.ck_q5_1_q8_1_provider_name().decode()
        cls.has_sdot = cls.provider == "q5_1_q8_1_neon_sdot"
        # Optimized-lane enforcement: when CK_REQUIRE_Q5_1_NEON=1 (set by the
        # nightly dotprod build step), failing to select the SDOT provider is
        # a hard failure, not a skip.
        cls.require_neon = os.environ.get("CK_REQUIRE_Q5_1_NEON", "") not in (
            "", "0")

    def _require_or_skip_sdot(self) -> None:
        if self.has_sdot:
            return
        if self.require_neon:
            self.fail("CK_REQUIRE_Q5_1_NEON=1 but the engine dispatches "
                      "gemv_q5_1_q8_1 to %r - the optimized lane did not "
                      "execute the intended SDOT NEON provider" % self.provider)
        self.skipTest("engine built without __ARM_FEATURE_DOTPROD; "
                      "rebuild with CK_ARM_ARCH or CK_ARM_NATIVE=1 to "
                      "exercise the SDOT provider")

    # ---- provider selection -------------------------------------------

    def test_provider_names_are_truthful(self) -> None:
        # Quantize is NEON on every AArch64 build.
        self.assertEqual(
            self.lib.ck_quantize_row_q8_1_provider_name().decode(),
            "q8_1_quantize_neon")
        # The block dot reports SDOT on dot-product builds, scalar otherwise.
        self.assertIn(self.provider, ("q5_1_q8_1_neon_sdot", "q5_1_q8_1_ref"))

    def test_forced_ref_escape_hatch(self) -> None:
        self.assertEqual(
            _provider_name_via_subprocess("ck_q5_1_q8_1_provider_name",
                                          "CK_DEBUG_Q5_1_Q8_1_REF"),
            "q5_1_q8_1_ref")
        self.assertEqual(
            _provider_name_via_subprocess("ck_quantize_row_q8_1_provider_name",
                                          "CK_DEBUG_Q5_1_Q8_1_REF"),
            "q8_1_quantize_ref")

    def test_require_neon_fail_closed(self) -> None:
        if not self.require_neon:
            self.skipTest("CK_REQUIRE_Q5_1_NEON not set")
        self.assertEqual(self.provider, "q5_1_q8_1_neon_sdot",
                         "optimized lane must select the SDOT provider")

    # ---- quantize_row_q8_1: byte-exact ---------------------------------

    def _check_quantize_bytes(self, x: np.ndarray, label: str) -> None:
        k = x.size
        nb = k // QK
        y = ctypes.create_string_buffer(nb * BLOCK_Q8_1_SIZE)
        self.lib.quantize_row_q8_1(x.ctypes.data_as(F32P), y, k)
        oracle = _oracle_quantize_q8_1(x)
        self.assertEqual(y.raw, oracle,
                         f"quantize_row_q8_1 not byte-exact vs scalar model ({label})")

    def test_quantize_random_rows_byte_exact(self) -> None:
        rng = self.rng
        for k in (QK, 4 * QK, 640, 2048):
            for _ in range(4):
                x = rng.standard_normal(k).astype(np.float32) * 0.1
                self._check_quantize_bytes(x, f"random k={k}")

    def test_quantize_adversarial_byte_exact(self) -> None:
        # exact .5 ties (roundf half-away), amax at +/- extremes, all-zero
        ties = np.full(QK, 0.0, dtype=np.float32)
        ties[0] = 127.0
        ties[1:9] = [2.5, -2.5, 3.5, -3.5, 0.5, -0.5, 126.5, -126.5]
        self._check_quantize_bytes(ties, "rounding ties")
        self._check_quantize_bytes(np.zeros(QK, dtype=np.float32), "zeros")
        x = (self.rng.standard_normal(2 * QK) * 1e-30).astype(np.float32)
        self._check_quantize_bytes(x, "tiny magnitudes")

    # ---- dot: integer-exact oracle + forced-scalar A/B -----------------

    def _check_gemv(self, M: int, K: int, qh_pattern: str | None = None) -> None:
        rng = self.rng
        nb = K // QK
        w = _random_q5_1_blocks(rng, nb * M, qh_pattern)
        x = _random_q8_1_blocks(rng, nb)
        y = np.zeros(M, dtype=np.float32)
        self.lib.gemv_q5_1_q8_1_ref(y.ctypes.data_as(F32P), _buf(w), _buf(x),
                                    M, K)
        label = f"M={M} K={K} qh={qh_pattern or 'random'}"
        for row in range(M):
            w_row = w[row * nb * BLOCK_Q5_1_SIZE:(row + 1) * nb * BLOCK_Q5_1_SIZE]
            oracle = _oracle_dot_q5_1_q8_1(w_row, x, K)
            self.assertTrue(
                np.isclose(y[row], oracle, rtol=RTOL, atol=ATOL),
                f"{label} row {row}: dispatch {y[row]} vs oracle {oracle}")
        # A/B against the forced-scalar dot (subprocess, same inputs): the
        # int8 x int8 dot is exact in int32 and the FP32 epilogue is
        # identical, so this must be bit-exact.
        y_ref = _gemv_forced_ref_via_subprocess(w, x, M, K)
        np.testing.assert_array_equal(
            y, y_ref, err_msg=f"dispatch not bit-exact vs scalar ({label})")

    def test_gemv_matches_oracle_and_scalar(self) -> None:
        # Block counts 1, 2, 8 and Gemma3-270M-realistic K = 640/2048.
        for M, K in ((1, QK), (4, 2 * QK), (5, 8 * QK), (3, 640), (2, 2048)):
            with self.subTest(M=M, K=K):
                self._check_gemv(M, K)

    def test_gemv_high_bit_patterns(self) -> None:
        for pattern in ("zeros", "ones", "alt_lo", "alt_hi"):
            with self.subTest(qh=pattern):
                self._check_gemv(3, 4 * QK, pattern)

    def test_gemv_fp32_path_matches_oracle(self) -> None:
        # Production entry: gemv_q5_1_q8_1 quantizes FP32 activations to Q8_1
        # (NEON, byte-exact) then dots. Cross-check against the composed
        # oracle: quantize in NumPy, dot integer-exact.
        self._require_or_skip_sdot()
        rng = self.rng
        for M, K in ((3, 640), (2, 2048)):
            with self.subTest(M=M, K=K):
                nb = K // QK
                w = _random_q5_1_blocks(rng, nb * M)
                x = rng.standard_normal(K).astype(np.float32) * 0.1
                y = np.zeros(M, dtype=np.float32)
                self.lib.gemv_q5_1_q8_1(y.ctypes.data_as(F32P), _buf(w),
                                        x.ctypes.data_as(F32P), M, K)
                x_q8 = _oracle_quantize_q8_1(x)
                for row in range(M):
                    w_row = w[row * nb * BLOCK_Q5_1_SIZE:
                              (row + 1) * nb * BLOCK_Q5_1_SIZE]
                    oracle = _oracle_dot_q5_1_q8_1(w_row, x_q8, K)
                    self.assertTrue(
                        np.isclose(y[row], oracle, rtol=RTOL, atol=ATOL),
                        f"row {row}: gemv {y[row]} vs composed oracle {oracle}")

    def test_gemm_nt_m2_matches_oracle(self) -> None:
        # Gemma3 batch-decode two-row path.
        self._require_or_skip_sdot()
        rng = self.rng
        N, K = 5, 640
        nb = K // QK
        w = _random_q5_1_blocks(rng, nb * N)
        a = rng.standard_normal((2, K)).astype(np.float32) * 0.1
        bias = np.linspace(-0.5, 0.5, N, dtype=np.float32)
        c = np.full((2, N), np.nan, dtype=np.float32)
        self.lib.gemm_nt_q5_1_q8_1_m2(
            a.ctypes.data_as(F32P), _buf(w), bias.ctypes.data_as(F32P),
            c.ctypes.data_as(F32P), 2, N, K)
        self.assertFalse(np.isnan(c).any(), "m2 left outputs unwritten")
        for m in range(2):
            a_q8 = _oracle_quantize_q8_1(a[m])
            for n in range(N):
                w_row = w[n * nb * BLOCK_Q5_1_SIZE:(n + 1) * nb * BLOCK_Q5_1_SIZE]
                oracle = _oracle_dot_q5_1_q8_1(w_row, a_q8, K) + float(bias[n])
                self.assertTrue(
                    np.isclose(c[m, n], oracle, rtol=1e-4, atol=ATOL),
                    f"C[{m},{n}]: m2 {c[m, n]} vs oracle {oracle}")


if __name__ == "__main__":
    unittest.main()
