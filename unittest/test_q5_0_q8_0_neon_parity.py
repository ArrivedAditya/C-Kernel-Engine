"""NEON parity tests for the Q5_0 x Q8_0 quantized dot on AArch64.

Covers the NEON paths in src/kernels/gemm_kernels_q5_0.c:

1. Default AArch64 dispatch selects the SDOT (vdotq_s32) provider when built
   with the dot-product extension (e.g. CK_ARM_ARCH="armv8.2-a+dotprod"), and
   honestly reports the scalar reference on baseline armv8-a builds (the
   plain-NEON kernel measured 2.5-3.3x slower than gcc-autovectorized scalar
   on Cortex-X925, so it is compiled but not dispatched - same finding as
   Q4_K).
2. CK_DEBUG_Q5_0_Q8_0_REF=1 forces the scalar reference and the provider
   name reports it truthfully (A/B escape hatch, mirrors
   CK_DEBUG_Q4K_Q8_REF).
3. The dispatched dot is BIT-EXACT against vec_dot_q5_0_q8_0_ref: the int8
   x int8 products are exact in int32 and every provider accumulates block
   contributions in the same FP32 order. An independent integer-exact NumPy
   oracle cross-checks the arithmetic (FP64 accumulation, tight tolerance).
4. High-bit (5th bit) reconstruction is exercised with adversarial qh
   patterns: all-zeros, all-ones, and alternating bit patterns.
5. gemv_q5_0_q8_0 and gemv_q5_0_q8_0_parallel_simd route through the same
   dispatcher and match the oracle at model-realistic K (Qwen2-0.5B: 896,
   1536, 2048).

Runs only on aarch64/arm64; skipped elsewhere. When CK_REQUIRE_Q5_0_NEON=1
(set by the nightly dotprod build step), failing to select the SDOT provider
is a hard failure, not a skip.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "build" / "libckernel_engine.so"

QK5_0 = 32
BLOCK_Q5_0_SIZE = 22  # d[2] + qh[4] + qs[16]
BLOCK_Q8_0_SIZE = 34  # d[2] (fp16) + qs[32]

F32P = ctypes.POINTER(ctypes.c_float)

# Integer arithmetic is identical between NEON and the scalar reference; the
# FP32 per-block epilogue is also identical, so dispatch must be bit-exact
# against the C reference. The NumPy oracle accumulates in FP64, so its
# comparison uses a tight float tolerance only.
RTOL = 1e-5
ATOL = 1e-4


def _fp16_bytes(f: float) -> bytes:
    return np.array([f], dtype=np.float16).tobytes()


def _fp16_at(b: bytes, off: int) -> float:
    return float(np.frombuffer(b[off:off + 2], dtype=np.float16)[0])


def _q5_0_block(rng: np.random.Generator, qh_pattern: str | None = None) -> bytes:
    out = bytearray()
    out += _fp16_bytes(float(rng.uniform(0.005, 0.05)))                    # d
    if qh_pattern == "zeros":
        out += bytes(4)
    elif qh_pattern == "ones":
        out += b"\xff" * 4
    elif qh_pattern == "alt_lo":
        out += (0x55555555).to_bytes(4, "little")                          # even bits set
    elif qh_pattern == "alt_hi":
        out += (0xAAAAAAAA).to_bytes(4, "little")                          # odd bits set
    else:
        out += rng.integers(0, 256, size=4, dtype=np.uint8).tobytes()      # qh
    out += rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()         # qs
    return bytes(out)


def _random_q5_0_blocks(rng: np.random.Generator, nb: int,
                        qh_pattern: str | None = None) -> bytes:
    return b"".join(_q5_0_block(rng, qh_pattern) for _ in range(nb))


def _random_q8_0_blocks(rng: np.random.Generator, nb: int) -> bytes:
    out = bytearray()
    for _ in range(nb):
        out += _fp16_bytes(float(rng.uniform(0.005, 0.05)))                # d
        out += rng.integers(-128, 128, size=QK5_0, dtype=np.int8).tobytes()
    return bytes(out)


def _oracle_dot_q5_0_q8_0(w: bytes, x: bytes, K: int) -> float:
    """Independent integer-exact oracle for one Q5_0 x Q8_0 dot.

    Q5_0 layout (ggml-compatible): weight j uses the low nibble of qs[j]
    plus bit j of qh; weight j+16 uses the high nibble of qs[j] plus bit
    (j+16) of qh. Values are (nibble | (bit << 4)) - 16, i.e. signed
    int8 in [-16, 15].
    """
    nb = K // QK5_0
    total = np.float64(0.0)
    for i in range(nb):
        bw = i * BLOCK_Q5_0_SIZE
        bx = i * BLOCK_Q8_0_SIZE
        d = np.float64(_fp16_at(w, bw)) * np.float64(_fp16_at(x, bx))
        qh = int.from_bytes(w[bw + 2:bw + 6], "little")
        qs = w[bw + 6:bw + 22]
        q8 = np.frombuffer(x[bx + 2:bx + 34], dtype=np.int8).astype(np.int64)
        sumi = 0
        for j in range(QK5_0 // 2):
            x0 = ((qs[j] & 0x0F) | (((qh >> j) & 1) << 4)) - 16
            x1 = ((qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16
            sumi += x0 * int(q8[j]) + x1 * int(q8[j + QK5_0 // 2])
        total += d * np.float64(sumi)
    return float(total)


def _buf(data: bytes) -> ctypes.Array:
    return ctypes.create_string_buffer(data, len(data))


def _provider_name_via_subprocess(env_var: str) -> str:
    code = (
        "import ctypes;"
        "lib = ctypes.CDLL(%r);"
        "f = lib.ck_q5_0_q8_0_provider_name; f.restype = ctypes.c_char_p;"
        "print(f().decode())" % (str(LIB_PATH),))
    env = dict(os.environ)
    env[env_var] = "1"
    return subprocess.check_output([sys.executable, "-c", code],
                                   env=env, text=True).strip()


@unittest.skipUnless(platform.machine() in ("aarch64", "arm64"),
                     "NEON parity path is AArch64-only")
@unittest.skipUnless(LIB_PATH.exists(), "engine library not built")
class TestQ50Q80NeonParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = ctypes.CDLL(str(LIB_PATH))
        cls.lib.vec_dot_q5_0_q8_0.argtypes = [
            ctypes.c_int, F32P, ctypes.c_void_p, ctypes.c_void_p]
        cls.lib.vec_dot_q5_0_q8_0_ref.argtypes = \
            cls.lib.vec_dot_q5_0_q8_0.argtypes
        cls.lib.vec_dot_q5_0_q8_0_neon.argtypes = \
            cls.lib.vec_dot_q5_0_q8_0.argtypes
        cls.lib.gemv_q5_0_q8_0.argtypes = [F32P, ctypes.c_void_p,
                                           ctypes.c_void_p,
                                           ctypes.c_int, ctypes.c_int]
        cls.lib.gemv_q5_0_q8_0_parallel_simd.argtypes = [
            F32P, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        cls.lib.ck_q5_0_q8_0_provider_name.restype = ctypes.c_char_p
        cls.rng = np.random.default_rng(20261024)
        cls.provider = cls.lib.ck_q5_0_q8_0_provider_name().decode()
        # The GEMV SDOT dot needs the dot-product extension; baseline armv8-a
        # engine builds dispatch to the scalar reference instead (plain NEON
        # measured slower than autovectorized scalar on Cortex-X925).
        cls.has_sdot = cls.provider == "q5_0_q8_0_neon_sdot"
        # Optimized-lane enforcement: when CK_REQUIRE_Q5_0_NEON=1 (set by the
        # nightly dotprod build step), failing to select the SDOT provider is
        # a hard failure, not a skip - the optimized lane must actually
        # execute the optimized kernel.
        cls.require_neon = os.environ.get("CK_REQUIRE_Q5_0_NEON", "") not in (
            "", "0")

    def _require_or_skip_sdot(self) -> None:
        if self.has_sdot:
            return
        if self.require_neon:
            self.fail("CK_REQUIRE_Q5_0_NEON=1 but the engine dispatches "
                      "vec_dot_q5_0_q8_0 to %r - the optimized lane did not "
                      "execute the intended SDOT NEON provider" % self.provider)
        self.skipTest("engine built without __ARM_FEATURE_DOTPROD; "
                      "rebuild with CK_ARM_ARCH or CK_ARM_NATIVE=1 to "
                      "exercise the SDOT provider")

    # ---- provider selection -------------------------------------------

    def test_provider_names_are_truthful(self) -> None:
        # AArch64 dispatch selects SDOT when the dot-product extension is
        # compiled in, and the scalar reference otherwise (plain NEON is not
        # dispatched; see the SDOT kernel comment). It must honestly report
        # which one.
        self.assertIn(self.provider,
                      ("q5_0_q8_0_neon_sdot", "q5_0_q8_0_ref"))

    def test_forced_ref_escape_hatch(self) -> None:
        self.assertEqual(
            _provider_name_via_subprocess("CK_DEBUG_Q5_0_Q8_0_REF"),
            "q5_0_q8_0_ref")

    def test_require_neon_fail_closed(self) -> None:
        if not self.require_neon:
            self.skipTest("CK_REQUIRE_Q5_0_NEON not set")
        self.assertEqual(self.provider, "q5_0_q8_0_neon_sdot",
                         "optimized lane must select the SDOT provider")

    # ---- dot: bit-exact vs C ref + integer-exact oracle ----------------

    def _check_dot(self, K: int, qh_pattern: str | None = None) -> None:
        rng = self.rng
        nb = K // QK5_0
        w = _random_q5_0_blocks(rng, nb, qh_pattern)
        x = _random_q8_0_blocks(rng, nb)
        s_dispatch = np.zeros(1, dtype=np.float32)
        s_ref = np.zeros(1, dtype=np.float32)
        s_neon = np.zeros(1, dtype=np.float32)
        self.lib.vec_dot_q5_0_q8_0(K, s_dispatch.ctypes.data_as(F32P),
                                   _buf(w), _buf(x))
        self.lib.vec_dot_q5_0_q8_0_ref(K, s_ref.ctypes.data_as(F32P),
                                       _buf(w), _buf(x))
        # The plain-NEON kernel exists on every AArch64 build and must also
        # be bit-exact against the reference.
        self.lib.vec_dot_q5_0_q8_0_neon(K, s_neon.ctypes.data_as(F32P),
                                        _buf(w), _buf(x))
        label = f"K={K} qh={qh_pattern or 'random'}"
        np.testing.assert_array_equal(
            s_dispatch, s_ref,
            err_msg=f"dispatched dot not bit-exact vs ref ({label})")
        np.testing.assert_array_equal(
            s_neon, s_ref,
            err_msg=f"plain-NEON dot not bit-exact vs ref ({label})")
        oracle = _oracle_dot_q5_0_q8_0(w, x, K)
        self.assertTrue(
            np.isclose(s_dispatch[0], oracle, rtol=RTOL, atol=ATOL),
            f"{label}: dispatch {s_dispatch[0]} vs oracle {oracle}")

    def test_dot_matches_ref_and_oracle(self) -> None:
        # Block counts: 1, 2, 8 and Qwen2-0.5B-realistic K = 896/1536/2048.
        for K in (QK5_0, 2 * QK5_0, 8 * QK5_0, 896, 1536, 2048):
            with self.subTest(K=K):
                self._check_dot(K)

    def test_dot_high_bit_patterns(self) -> None:
        # Adversarial qh patterns exercise the 5th-bit reconstruction.
        for pattern in ("zeros", "ones", "alt_lo", "alt_hi"):
            with self.subTest(qh=pattern):
                self._check_dot(4 * QK5_0, pattern)

    def test_dot_sdot_provider_when_required(self) -> None:
        # Dispatch evidence comes ONLY from the provider name, whose
        # compile-time branch is the same preprocessor condition the
        # dispatcher uses: equal outputs can never prove SDOT executed (the
        # kernels are integer-exact against each other by design).
        self._require_or_skip_sdot()
        self._check_dot(2 * QK5_0)

    # ---- gemv / parallel gemv entry points -----------------------------

    def test_gemv_matches_oracle(self) -> None:
        rng = self.rng
        for M, K in ((1, QK5_0), (4, 896), (5, 2048)):
            with self.subTest(M=M, K=K):
                nb = K // QK5_0
                w = _random_q5_0_blocks(rng, nb * M)
                x = _random_q8_0_blocks(rng, nb)
                y = np.zeros(M, dtype=np.float32)
                self.lib.gemv_q5_0_q8_0(y.ctypes.data_as(F32P),
                                        _buf(w), _buf(x), M, K)
                for row in range(M):
                    w_row = w[row * nb * BLOCK_Q5_0_SIZE:
                              (row + 1) * nb * BLOCK_Q5_0_SIZE]
                    oracle = _oracle_dot_q5_0_q8_0(w_row, x, K)
                    self.assertTrue(
                        np.isclose(y[row], oracle, rtol=RTOL, atol=ATOL),
                        f"row {row}: GEMV {y[row]} vs oracle {oracle}")

    def test_parallel_simd_matches_oracle(self) -> None:
        rng = self.rng
        M, K, nth = 9, 1536, 4
        nb = K // QK5_0
        w = _random_q5_0_blocks(rng, nb * M)
        x = _random_q8_0_blocks(rng, nb)
        y = np.full(M, np.nan, dtype=np.float32)
        for ith in range(nth):
            self.lib.gemv_q5_0_q8_0_parallel_simd(
                y.ctypes.data_as(F32P), _buf(w), _buf(x), M, K, ith, nth)
        self.assertFalse(np.isnan(y).any(), "parallel GEMV left rows unwritten")
        for row in range(M):
            w_row = w[row * nb * BLOCK_Q5_0_SIZE:
                      (row + 1) * nb * BLOCK_Q5_0_SIZE]
            oracle = _oracle_dot_q5_0_q8_0(w_row, x, K)
            self.assertTrue(
                np.isclose(y[row], oracle, rtol=RTOL, atol=ATOL),
                f"row {row}: parallel GEMV {y[row]} vs oracle {oracle}")


if __name__ == "__main__":
    unittest.main()
