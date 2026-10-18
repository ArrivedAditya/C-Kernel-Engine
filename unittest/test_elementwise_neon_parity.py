"""NEON parity tests for the elementwise kernels (ARM Phase 3).

Covers the AArch64 NEON paths in:

1. ck_residual_add_token_major (src/ckernel_orchestration.c) -- pure FP32
   elementwise add. There is no hand-written NEON provider here: gcc -O3
   already autovectorizes the scalar loop to NEON (verified in the build).
   This test pins the numerical contract regardless of how the loop is
   compiled: FP32 add is elementwise with no reduction, so the result must
   be BIT-EXACT against an independent NumPy fp32 oracle.
2. rmsnorm_forward / rmsnorm_forward_strided_f32 (src/kernels/rmsnorm_kernels.c)
   -- the NEON branch reassociates the sum-of-squares through four lanes,
   exactly like the existing AVX/AVX-512 branches (tolerance-based contract,
   not bit-exact). The CK_RMSNORM_EXACT / strict-parity escape hatch keeps
   the ascending scalar reference; the test compares NEON against that
   reference and against a float64 NumPy oracle.
3. swiglu_forward_ggml / swiglu_forward_ggml_split (src/kernels/swiglu_kernels.c)
   -- the NEON branch uses ck_ggml_expf_neon, llama.cpp's own AArch64
   ggml_v_expf polynomial (the same approximation family as the AVX2 path).
   Contract is the AVX path's: approximation with tolerance, not libm-exact.
   The test compares against the libm-exact swiglu_forward_exact provider
   and a float64 NumPy oracle, plus the CK_DEBUG_SWIGLU_GGML_REF escape
   hatch and provider-name truthfulness.

Runs only on aarch64/arm64; skipped elsewhere.
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

F32P = ctypes.POINTER(ctypes.c_float)

# Tolerances mirror the existing AVX contracts: the AVX rmsnorm path is
# validated at max_diff <= 1e-6 against PyTorch (tolerance-based, not
# bit-exact), and the fast SwiGLU path at 5e-5..1e-4. NEON reassociation
# noise is the same order; we use rtol=1e-5/atol=1e-6 against both the
# scalar references and the float64 NumPy oracles.
RTOL = 1e-5
ATOL = 1e-6


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


def _f32p(a: np.ndarray) -> F32P:
    return a.ctypes.data_as(F32P)


@unittest.skipUnless(platform.machine() in ("aarch64", "arm64"),
                     "NEON parity path is AArch64-only")
@unittest.skipUnless(LIB_PATH.exists(), "engine library not built")
class TestElementwiseNeonParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = ctypes.CDLL(str(LIB_PATH))
        cls.lib.ck_residual_add_token_major.argtypes = [
            F32P, F32P, F32P, ctypes.c_int, ctypes.c_int]
        cls.lib.rmsnorm_forward.argtypes = [
            F32P, F32P, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        cls.lib.swiglu_forward_ggml.argtypes = [
            F32P, F32P, ctypes.c_int, ctypes.c_int]
        cls.lib.swiglu_forward_ggml_split.argtypes = [
            F32P, F32P, F32P, ctypes.c_int, ctypes.c_int]
        cls.lib.swiglu_forward_exact.argtypes = [
            F32P, F32P, ctypes.c_int, ctypes.c_int]
        cls.lib.ck_rmsnorm_provider_name.restype = ctypes.c_char_p
        cls.lib.ck_swiglu_ggml_provider_name.restype = ctypes.c_char_p
        cls.lib.ck_set_num_threads.argtypes = [ctypes.c_int]
        cls.lib.ck_threadpool_global_destroy.argtypes = []
        cls.rng = np.random.default_rng(20261017)
        cls.rmsnorm_provider = cls.lib.ck_rmsnorm_provider_name().decode()
        cls.swiglu_provider = cls.lib.ck_swiglu_ggml_provider_name().decode()

    # ---- provider selection -------------------------------------------

    def test_provider_names_are_truthful(self) -> None:
        # On AArch64 the default (non-strict) providers must be the NEON
        # paths added in this change, not a silent scalar fallback.
        self.assertEqual(self.rmsnorm_provider, "rmsnorm_neon")
        self.assertEqual(self.swiglu_provider, "swiglu_ggml_neon")

    def test_forced_ref_escape_hatches(self) -> None:
        self.assertEqual(
            _provider_name_via_subprocess("ck_rmsnorm_provider_name",
                                          "CK_RMSNORM_EXACT"),
            "rmsnorm_strict_scalar")
        self.assertEqual(
            _provider_name_via_subprocess("ck_swiglu_ggml_provider_name",
                                          "CK_DEBUG_SWIGLU_GGML_REF"),
            "swiglu_ggml_ref")

    # ---- residual add: bit-exact ---------------------------------------

    def test_residual_add_bit_exact_vs_numpy(self) -> None:
        rng = self.rng
        for tokens, dim in ((1, 2048), (1, 4096), (7, 4096), (3, 130)):
            with self.subTest(tokens=tokens, dim=dim):
                a = rng.standard_normal((tokens, dim)).astype(np.float32)
                b = rng.standard_normal((tokens, dim)).astype(np.float32)
                out = np.full((tokens, dim), np.nan, dtype=np.float32)
                self.lib.ck_residual_add_token_major(
                    _f32p(a), _f32p(b), _f32p(out), tokens, dim)
                expected = (a.astype(np.float32) + b.astype(np.float32))
                self.assertTrue(
                    np.array_equal(out.view(np.uint32),
                                   expected.view(np.uint32)),
                    f"residual add not bit-exact ({tokens}x{dim})")

    # ---- rmsnorm: tolerance vs strict scalar and float64 oracle ---------

    def _rmsnorm(self, x: np.ndarray, gamma: np.ndarray,
                 d_model: int, out_stride: int, eps: float,
                 exact: bool) -> tuple[np.ndarray, np.ndarray]:
        tokens = x.shape[0]
        out = np.full((tokens, out_stride), np.nan, dtype=np.float32)
        rstd = np.full(tokens, np.nan, dtype=np.float32)
        prev = os.environ.get("CK_RMSNORM_EXACT")
        try:
            if exact:
                os.environ["CK_RMSNORM_EXACT"] = "1"
            elif prev is not None:
                del os.environ["CK_RMSNORM_EXACT"]
            self.lib.rmsnorm_forward(
                _f32p(x), _f32p(gamma), _f32p(out), _f32p(rstd),
                tokens, d_model, out_stride, ctypes.c_float(eps))
        finally:
            if prev is None:
                os.environ.pop("CK_RMSNORM_EXACT", None)
            else:
                os.environ["CK_RMSNORM_EXACT"] = prev
        return out, rstd

    def _check_rmsnorm(self, tokens: int, d_model: int,
                       out_stride: int) -> None:
        rng = self.rng
        eps = 1e-5
        x = rng.standard_normal((tokens, out_stride)).astype(np.float32)
        gamma = rng.standard_normal(out_stride).astype(np.float32)
        neon, rstd_neon = self._rmsnorm(x, gamma, d_model, out_stride, eps,
                                        exact=False)
        ref, rstd_ref = self._rmsnorm(x, gamma, d_model, out_stride, eps,
                                      exact=True)
        # Padding region must be zeroed identically.
        if out_stride > d_model:
            self.assertTrue((neon[:, d_model:] == 0.0).all())
        np.testing.assert_allclose(
            neon[:, :d_model], ref[:, :d_model], rtol=RTOL, atol=ATOL,
            err_msg=f"NEON rmsnorm vs strict scalar (T={tokens}, D={d_model})")
        np.testing.assert_allclose(
            rstd_neon, rstd_ref, rtol=RTOL, atol=ATOL,
            err_msg=f"NEON rstd vs strict scalar (T={tokens}, D={d_model})")
        # Independent float64 NumPy oracle.
        x64 = x[:, :d_model].astype(np.float64)
        rstd64 = 1.0 / np.sqrt((x64 * x64).mean(axis=1) + eps)
        y64 = x64 * rstd64[:, None] * gamma[:d_model].astype(np.float64)
        np.testing.assert_allclose(
            neon[:, :d_model], y64, rtol=RTOL, atol=ATOL,
            err_msg=f"NEON rmsnorm vs float64 oracle (T={tokens}, D={d_model})")

    def test_rmsnorm_neon_matches_ref_and_oracle(self) -> None:
        for tokens, d_model, out_stride in (
                (1, 2048, 2048), (1, 4096, 4096),
                (5, 4096, 4096), (3, 130, 136)):
            with self.subTest(tokens=tokens, d_model=d_model,
                              out_stride=out_stride):
                self._check_rmsnorm(tokens, d_model, out_stride)

    # ---- swiglu ggml: tolerance vs libm-exact and float64 oracle --------

    def _check_swiglu(self, interleaved: np.ndarray) -> None:
        tokens, two_d = interleaved.shape
        dim = two_d // 2
        neon = np.full((tokens, dim), np.nan, dtype=np.float32)
        exact = np.full((tokens, dim), np.nan, dtype=np.float32)
        self.lib.swiglu_forward_ggml(_f32p(interleaved), _f32p(neon),
                                     tokens, dim)
        self.lib.swiglu_forward_exact(_f32p(interleaved), _f32p(exact),
                                      tokens, dim)
        np.testing.assert_allclose(
            neon, exact, rtol=RTOL, atol=ATOL,
            err_msg=f"NEON ggml swiglu vs libm-exact (T={tokens}, D={dim})")
        # Independent float64 NumPy oracle: silu(g) * u.
        g = interleaved[:, :dim].astype(np.float64)
        u = interleaved[:, dim:].astype(np.float64)
        y64 = (g / (1.0 + np.exp(-g))) * u
        np.testing.assert_allclose(
            neon, y64, rtol=RTOL, atol=ATOL,
            err_msg=f"NEON ggml swiglu vs float64 oracle (T={tokens}, D={dim})")

    def test_swiglu_ggml_neon_matches_ref_and_oracle(self) -> None:
        rng = self.rng
        for tokens, dim in ((1, 2048), (1, 4096), (4, 4096), (2, 2050)):
            with self.subTest(tokens=tokens, dim=dim):
                x = rng.standard_normal((tokens, 2 * dim)).astype(np.float32)
                self._check_swiglu(x)

    def test_swiglu_ggml_neon_extreme_gates(self) -> None:
        # Exercise the overflow/underflow mask paths of the polynomial exp:
        # |n| > 126 (split scaling), n <= 0 (underflow bias), |n| > 192
        # (flush to inf / zero).
        dim = 64
        gates = np.linspace(-300.0, 300.0, dim, dtype=np.float32)
        ups = np.linspace(-4.0, 4.0, dim, dtype=np.float32)
        x = np.concatenate([gates, ups]).reshape(1, 2 * dim)
        self._check_swiglu(np.ascontiguousarray(x))

    def test_swiglu_ggml_split_neon_matches_oracle(self) -> None:
        rng = self.rng
        tokens, dim = 3, 4096
        gate = rng.standard_normal((tokens, dim)).astype(np.float32)
        up = rng.standard_normal((tokens, dim)).astype(np.float32)
        out = np.full((tokens, dim), np.nan, dtype=np.float32)
        self.lib.swiglu_forward_ggml_split(_f32p(gate), _f32p(up),
                                           _f32p(out), tokens, dim)
        g64 = gate.astype(np.float64)
        u64 = up.astype(np.float64)
        y64 = (g64 / (1.0 + np.exp(-g64))) * u64
        np.testing.assert_allclose(out, y64, rtol=RTOL, atol=ATOL,
                                   err_msg="NEON ggml split swiglu vs oracle")

    def test_swiglu_ggml_threading_is_deterministic(self) -> None:
        # Row decomposition must not change results (bit-exact across thread
        # counts), as pinned by the existing swiglu parallel-consistency test.
        rng = self.rng
        tokens, dim = 64, 4096  # 64*4096 >= 2*65536: row parallelism engages
        x = rng.standard_normal((tokens, 2 * dim)).astype(np.float32)
        serial = np.empty((tokens, dim), dtype=np.float32)
        parallel = np.empty_like(serial)
        try:
            self.lib.ck_threadpool_global_destroy()
            self.lib.ck_set_num_threads(1)
            self.lib.swiglu_forward_ggml(_f32p(x), _f32p(serial), tokens, dim)
            self.lib.ck_threadpool_global_destroy()
            self.lib.ck_set_num_threads(4)
            self.lib.swiglu_forward_ggml(_f32p(x), _f32p(parallel), tokens, dim)
        finally:
            self.lib.ck_threadpool_global_destroy()
        self.assertTrue(np.array_equal(serial.view(np.uint32),
                                       parallel.view(np.uint32)),
                        "swiglu ggml NEON not deterministic across threads")


if __name__ == "__main__":
    unittest.main()
