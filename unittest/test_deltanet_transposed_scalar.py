#!/usr/bin/env python3
"""Regression for the portable (non-AVX2) DeltaNet recurrent core.

PR #577 fixed a silent no-op: on non-AVX2 targets the transposed llama-layout
grouped kernel (gated_deltanet_llama_avx2_grouped_forward_transposed_impl in
src/kernels/deltanet_kernels.c) had an empty #else branch, so recurrent state
never updated and qwen3.5-class models emitted garbage on ARM.

This test compiles the same production source WITHOUT AVX2 enabled, which
activates the portable scalar branch, and checks it against an independent
float64 NumPy implementation of the recurrence:

    gate   = exp(g[h]);  beta_s = sigmoid(beta[h])
    per head h (group = h % group_count), per state column:
        cur    = prev_col * gate
        memory = cur . k_group
        delta  = (v_h[col] - memory) * beta_s
        cur   += k_group * delta
        out_h[col] = (cur . q_group) / sqrt(state_dim)

It exercises multi-step state chaining with changing inputs, grouped heads,
head-range partitioning, and argument validation. The no-op guard (sentinel
buffers) fails against the pre-#577 implementation.

Portable: no model downloads, no GPU; needs only gcc + numpy. Runs on any
host (x86 CI included) because the scalar branch is selected at compile time.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "kernels" / "deltanet_kernels.c"
F32P = ctypes.POINTER(ctypes.c_float)

SENTINEL = -777.0


def _build_scalar_lib(tmp: Path) -> ctypes.CDLL:
    """Compile deltanet_kernels.c with __AVX2__ undefined (scalar branch).

    Links against the built engine for shared helpers (strict-parity flag,
    threadpool symbols used by entry points this test does not call).
    -Bsymbolic keeps this library's own scalar definitions in charge of its
    internal calls, so the engine's AVX2 symbols cannot interpose.
    """
    engine_dir = ROOT / "build"
    engine = engine_dir / "libckernel_engine.so"
    if not engine.is_file():
        raise unittest.SkipTest(
            "build/libckernel_engine.so required (make build/libckernel_engine.so)"
        )
    lib = tmp / "libdeltanet_scalar.so"
    cmd = [
        "gcc", "-O2", "-fPIC", "-shared", "-Wl,-Bsymbolic",
        "-I", str(ROOT / "include"), "-I", str(ROOT / "src"),
        str(SRC), "-o", str(lib),
        "-L", str(engine_dir), "-lckernel_engine",
        "-Wl,-rpath," + str(engine_dir),
        "-lm", "-lpthread", "-ldl",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    cdll = ctypes.CDLL(str(lib))
    for name in (
        "gated_deltanet_llama_avx2_forward",
        "gated_deltanet_llama_avx2_forward_head_range",
    ):
        getattr(cdll, name).argtypes = [
            F32P, F32P, F32P, F32P, F32P, F32P, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ]
    cdll.gated_deltanet_llama_avx2_forward_head_range.argtypes += [
        ctypes.c_int, ctypes.c_int,
    ]
    return cdll


def _reference_step(q, k, v, g, beta, state, num_heads, group_count, state_dim):
    """Independent float64 recurrence; state shape (num_heads, state_dim, state_dim)
    uses the transposed (column-contiguous) llama layout."""
    scale = 1.0 / np.sqrt(state_dim)
    state = state.astype(np.float64).copy()
    out = np.zeros((num_heads, state_dim), dtype=np.float64)
    for h in range(num_heads):
        grp = h % group_count
        q_h = q[grp].astype(np.float64)
        k_h = k[grp].astype(np.float64)
        v_h = v[h].astype(np.float64)
        gate = np.exp(g[h])
        beta_s = 1.0 / (1.0 + np.exp(-beta[h]))
        for col in range(state_dim):
            cur = state[h, col] * gate          # gate row of column
            memory = cur @ k_h
            delta = (v_h[col] - memory) * beta_s
            cur = cur + k_h * delta
            state[h, col] = cur
            out[h, col] = (cur @ q_h) * scale
    return state, out


def _run_c(cdll, entry, q, k, v, g, beta, state_in, num_heads, group_count,
           state_dim, head_range=None):
    state_out = np.full_like(state_in, SENTINEL)
    out = np.full((num_heads, state_dim), SENTINEL, dtype=np.float32)
    args = [
        q.ctypes.data_as(F32P), k.ctypes.data_as(F32P), v.ctypes.data_as(F32P),
        g.ctypes.data_as(F32P), beta.ctypes.data_as(F32P),
        state_in.ctypes.data_as(F32P), state_out.ctypes.data_as(F32P),
        out.ctypes.data_as(F32P),
        num_heads, group_count, state_dim, ctypes.c_float(1e-5),
    ]
    if head_range is not None:
        entry(*args, head_range[0], head_range[1])
    else:
        entry(*args)
    return state_out, out


@unittest.skipUnless(shutil.which("gcc"), "gcc required to build scalar path")
class DeltaNetTransposedScalarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="ck-deltanet-scalar-")
        cls.lib = _build_scalar_lib(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_multistep_grouped_state_and_output(self):
        rng = np.random.default_rng(20261003)
        num_heads, group_count, state_dim, steps = 4, 2, 64, 5
        state_c = np.full((num_heads, state_dim, state_dim), SENTINEL,
                          dtype=np.float32)
        state_ref = np.zeros((num_heads, state_dim, state_dim),
                             dtype=np.float64)
        state_in = np.zeros_like(state_c)
        for step in range(steps):
            with self.subTest(step=step):
                q = rng.standard_normal((group_count, state_dim)).astype(np.float32)
                k = rng.standard_normal((group_count, state_dim)).astype(np.float32)
                v = rng.standard_normal((num_heads, state_dim)).astype(np.float32)
                g = (-0.5 * rng.random(num_heads)).astype(np.float32)
                beta = rng.standard_normal(num_heads).astype(np.float32)
                state_out, out = _run_c(
                    self.lib, self.lib.gated_deltanet_llama_avx2_forward,
                    q, k, v, g, beta, state_in, num_heads, group_count, state_dim)
                state_ref, out_ref = _reference_step(
                    q, k, v, g, beta, state_ref, num_heads, group_count, state_dim)
                # No-op guard: the pre-#577 branch leaves both sentinels.
                self.assertFalse(np.any(state_out == SENTINEL),
                                 "state_out untouched (no-op regression)")
                self.assertFalse(np.any(out == SENTINEL),
                                 "out untouched (no-op regression)")
                np.testing.assert_allclose(state_out, state_ref, rtol=2e-4,
                                           atol=2e-4)
                np.testing.assert_allclose(out, out_ref, rtol=2e-4, atol=2e-4)
                state_in = state_out

    def test_head_range_partition_matches_full(self):
        rng = np.random.default_rng(7)
        num_heads, group_count, state_dim = 4, 2, 32
        q = rng.standard_normal((group_count, state_dim)).astype(np.float32)
        k = rng.standard_normal((group_count, state_dim)).astype(np.float32)
        v = rng.standard_normal((num_heads, state_dim)).astype(np.float32)
        g = (-0.5 * rng.random(num_heads)).astype(np.float32)
        beta = rng.standard_normal(num_heads).astype(np.float32)
        state_in = rng.standard_normal((num_heads, state_dim, state_dim)).astype(np.float32)
        full_s, full_o = _run_c(
            self.lib, self.lib.gated_deltanet_llama_avx2_forward,
            q, k, v, g, beta, state_in, num_heads, group_count, state_dim)
        part_s = np.full_like(state_in, SENTINEL)
        part_o = np.full((num_heads, state_dim), SENTINEL, dtype=np.float32)
        hr = self.lib.gated_deltanet_llama_avx2_forward_head_range
        # Drive each disjoint slice through the range entry point.
        for begin, end in ((0, 2), (2, 4)):
            tmp_s, tmp_o = _run_c(
                self.lib, hr, q, k, v, g, beta, state_in,
                num_heads, group_count, state_dim, head_range=(begin, end))
            part_s[begin:end] = tmp_s[begin:end]
            part_o[begin:end] = tmp_o[begin:end]
        np.testing.assert_array_equal(part_s, full_s)
        np.testing.assert_array_equal(part_o, full_o)

    def test_ungrouped_heads(self):
        rng = np.random.default_rng(11)
        num_heads = group_count = 3
        state_dim = 48
        q = rng.standard_normal((group_count, state_dim)).astype(np.float32)
        k = rng.standard_normal((group_count, state_dim)).astype(np.float32)
        v = rng.standard_normal((num_heads, state_dim)).astype(np.float32)
        g = (-0.5 * rng.random(num_heads)).astype(np.float32)
        beta = rng.standard_normal(num_heads).astype(np.float32)
        state_in = rng.standard_normal((num_heads, state_dim, state_dim)).astype(np.float32)
        state_out, out = _run_c(
            self.lib, self.lib.gated_deltanet_llama_avx2_forward,
            q, k, v, g, beta, state_in, num_heads, group_count, state_dim)
        state_ref, out_ref = _reference_step(
            q, k, v, g, beta, state_in.astype(np.float64),
            num_heads, group_count, state_dim)
        np.testing.assert_allclose(state_out, state_ref, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(out, out_ref, rtol=2e-4, atol=2e-4)

    def test_invalid_args_leave_buffers_untouched(self):
        num_heads, group_count, state_dim = 4, 3, 32  # 4 % 3 != 0
        q = np.zeros((group_count, state_dim), dtype=np.float32)
        k = np.zeros((group_count, state_dim), dtype=np.float32)
        v = np.zeros((num_heads, state_dim), dtype=np.float32)
        g = np.zeros(num_heads, dtype=np.float32)
        beta = np.zeros(num_heads, dtype=np.float32)
        state_in = np.zeros((num_heads, state_dim, state_dim), dtype=np.float32)
        state_out, out = _run_c(
            self.lib, self.lib.gated_deltanet_llama_avx2_forward,
            q, k, v, g, beta, state_in, num_heads, group_count, state_dim)
        self.assertTrue(np.all(state_out == SENTINEL))
        self.assertTrue(np.all(out == SENTINEL))


if __name__ == "__main__":
    unittest.main()
