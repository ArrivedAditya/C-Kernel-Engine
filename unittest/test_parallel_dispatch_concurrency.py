"""Concurrency and dispatch-init tests for the v8 parallel decode/prefill
entry points.

The production decode path (version/v8/src/ck_parallel_decode_v8.c) shares one
global threadpool across callers; the provider/env caches in the quantized dot
dispatchers are initialized lazily on first use. These tests exercise the real
multithreaded entry points:

1. Concurrent calls from multiple Python threads (ctypes releases the GIL, so
   the C entry points genuinely run in parallel) must produce results
   bit-identical to the same inputs run sequentially. Per-row arithmetic does
   not depend on the thread partition, so parallel dispatch must match the
   serial GEMV bit-exactly.
2. Concurrent FIRST use in a fresh process (subprocess): provider-name query
   and parallel dispatch race on the lazy env-cache init; outputs must match
   the serial reference.
3. Repeated sequential requests must be deterministic.

Runs on x86 and AArch64 (the entry points exist on both); skipped only when
the engine library is not built.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "build" / "libckernel_engine.so"

# Pool size is fixed at first use; ask for 4 threads before loading the lib.
os.environ.setdefault("CK_NUM_THREADS", "4")

F32P = ctypes.POINTER(ctypes.c_float)

QK5_0 = 32
BLOCK_Q5_0_SIZE = 22   # d[2] + qh[4] + qs[16]
BLOCK_Q8_0_SIZE = 34   # d[2] + qs[32]

QK_K = 256
BLOCK_Q4_K_SIZE = 144  # d[2] + dmin[2] + scales[12] + qs[128]
BLOCK_Q8_K_SIZE = 292  # d[4] + qs[256] + bsums[32]


def _random_q5_0(rng: np.random.Generator, nblocks: int) -> bytes:
    out = bytearray()
    for _ in range(nblocks):
        out += np.array([rng.uniform(0.005, 0.05)], dtype=np.float16).tobytes()
        out += rng.integers(0, 256, size=4, dtype=np.uint8).tobytes()
        out += rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
    return bytes(out)


def _random_q8_0(rng: np.random.Generator, nblocks: int) -> bytes:
    out = bytearray()
    for _ in range(nblocks):
        out += np.array([rng.uniform(0.005, 0.05)], dtype=np.float16).tobytes()
        out += rng.integers(-128, 128, size=QK5_0, dtype=np.int8).tobytes()
    return bytes(out)


def _random_q4k(rng: np.random.Generator, nblocks: int) -> bytes:
    out = bytearray()
    for _ in range(nblocks):
        out += np.array([rng.uniform(0.005, 0.05)], dtype=np.float16).tobytes()
        out += np.array([rng.uniform(0.001, 0.02)], dtype=np.float16).tobytes()
        out += rng.integers(0, 256, size=12, dtype=np.uint8).tobytes()
        out += rng.integers(0, 256, size=128, dtype=np.uint8).tobytes()
    return bytes(out)


def _random_q8k(rng: np.random.Generator, nblocks: int) -> bytes:
    import struct
    out = bytearray()
    for _ in range(nblocks):
        qs = rng.integers(-128, 128, size=QK_K, dtype=np.int8)
        out += struct.pack("<f", float(rng.uniform(0.005, 0.05)))
        out += qs.tobytes()
        bsums = [int(qs[i * 16:(i + 1) * 16].sum()) for i in range(16)]
        out += np.array(bsums, dtype=np.int16).tobytes()
    return bytes(out)


def _buf(data: bytes) -> ctypes.Array:
    return ctypes.create_string_buffer(data, len(data))


@unittest.skipUnless(LIB_PATH.exists(), "engine library not built")
class TestParallelDispatchConcurrency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = ctypes.CDLL(str(LIB_PATH))
        cls.lib.gemv_q5_0_q8_0.argtypes = [F32P, ctypes.c_void_p,
                                           ctypes.c_void_p,
                                           ctypes.c_int, ctypes.c_int]
        cls.lib.gemv_q5_0_q8_0_parallel_dispatch.argtypes = \
            cls.lib.gemv_q5_0_q8_0.argtypes
        cls.lib.gemv_q4_k_q8_k.argtypes = cls.lib.gemv_q5_0_q8_0.argtypes
        cls.lib.gemv_q4_k_q8_k_parallel_dispatch.argtypes = \
            cls.lib.gemv_q5_0_q8_0.argtypes
        cls.lib.gemv_fused_q5_0_bias_dispatch.argtypes = [
            F32P, ctypes.c_void_p, F32P, F32P, ctypes.c_int, ctypes.c_int]
        cls.lib.gemv_fused_q5_0_bias_parallel_dispatch.argtypes = \
            cls.lib.gemv_fused_q5_0_bias_dispatch.argtypes
        cls.lib.gemm_nt_q4_k_q8_k.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        cls.lib.gemm_nt_q4_k_q8_k_parallel_dispatch.argtypes = \
            cls.lib.gemm_nt_q4_k_q8_k.argtypes
        # Warm the pool once so setUpClass failures surface here, not in a
        # worker thread.
        w = _buf(_random_q5_0(np.random.default_rng(0), QK5_0))
        x = _buf(_random_q8_0(np.random.default_rng(1), QK5_0))
        y = np.zeros(QK5_0, dtype=np.float32)
        cls.lib.gemv_q5_0_q8_0_parallel_dispatch(
            y.ctypes.data_as(F32P), w, x, QK5_0, 32 * QK5_0)

    # ---- helpers -------------------------------------------------------

    def _q5_0_case(self, seed: int, M: int, K: int):
        rng = np.random.default_rng(seed)
        nb = K // QK5_0
        w = _random_q5_0(rng, nb * M)
        x = _random_q8_0(rng, nb)
        return w, x

    def _q4k_case(self, seed: int, M: int, K: int):
        rng = np.random.default_rng(seed)
        nb = K // QK_K
        w = _random_q4k(rng, nb * M)
        x = _random_q8k(rng, nb)
        return w, x

    # ---- 1. concurrent calls, mixed entry points ------------------------

    def test_concurrent_mixed_dispatch_matches_sequential(self) -> None:
        n_threads = 4
        n_iters = 15
        M, K5, K4 = 768, 32 * 16, 512  # K5 for Q5_0/Q8_0, K4 for Q4_K/Q8_K
        barrier = threading.Barrier(n_threads)
        errors: list[str] = []

        def worker(tid: int) -> None:
            try:
                barrier.wait()
                for it in range(n_iters):
                    seed = tid * 1000 + it
                    # Q5_0 x Q8_0 GEMV
                    w, x = self._q5_0_case(seed, M, K5)
                    y_seq = np.zeros(M, dtype=np.float32)
                    self.lib.gemv_q5_0_q8_0(
                        y_seq.ctypes.data_as(F32P), _buf(w), _buf(x), M, K5)
                    y_par = np.zeros(M, dtype=np.float32)
                    self.lib.gemv_q5_0_q8_0_parallel_dispatch(
                        y_par.ctypes.data_as(F32P), _buf(w), _buf(x), M, K5)
                    if not np.array_equal(y_seq, y_par):
                        errors.append(f"t{tid}i{it}: q5_0 gemv mismatch")
                        return
                    # Q4_K x Q8_K GEMV
                    w4, x4 = self._q4k_case(seed, M, K4)
                    y4_seq = np.zeros(M, dtype=np.float32)
                    self.lib.gemv_q4_k_q8_k(
                        y4_seq.ctypes.data_as(F32P), _buf(w4), _buf(x4), M, K4)
                    y4_par = np.zeros(M, dtype=np.float32)
                    self.lib.gemv_q4_k_q8_k_parallel_dispatch(
                        y4_par.ctypes.data_as(F32P), _buf(w4), _buf(x4), M, K4)
                    if not np.array_equal(y4_seq, y4_par):
                        errors.append(f"t{tid}i{it}: q4_k gemv mismatch")
                        return
                    # Prefill GEMM entry (2 batch rows, distinct)
                    a4 = b"".join(self._q4k_case(seed + 1 + r, 1, K4)[1]
                                  for r in range(2))
                    c_seq = np.full((2, M), np.nan, dtype=np.float32)
                    self.lib.gemm_nt_q4_k_q8_k(
                        _buf(a4), _buf(w4), None, c_seq.ctypes.data_as(F32P),
                        2, M, K4)
                    c_par = np.full((2, M), np.nan, dtype=np.float32)
                    self.lib.gemm_nt_q4_k_q8_k_parallel_dispatch(
                        _buf(a4), _buf(w4), None, c_par.ctypes.data_as(F32P),
                        2, M, K4)
                    if np.isnan(c_par).any() or not np.array_equal(c_seq, c_par):
                        errors.append(f"t{tid}i{it}: q4_k gemm mismatch")
                        return
            except Exception as exc:  # noqa: BLE001 - report in main thread
                errors.append(f"t{tid}: exception {exc!r}")

        threads = [threading.Thread(target=worker, args=(t,))
                   for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_concurrent_fused_fp32_dispatch_matches_sequential(self) -> None:
        # The serial fused entry (gemv_fused_q5_0_bias_dispatch) uses a
        # different (FP32-dequant) arithmetic by design, so the concurrency
        # check compares parallel dispatch against itself run sequentially:
        # any cross-thread state corruption shows up as a mismatch.
        n_threads = 4
        n_iters = 15
        M, K = 768, 32 * 16
        bias = np.linspace(-0.25, 0.25, M, dtype=np.float32)
        barrier = threading.Barrier(n_threads)
        errors: list[str] = []

        def reference(seed: int) -> np.ndarray:
            rng = np.random.default_rng(seed)
            w = _random_q5_0(rng, (K // QK5_0) * M)
            x = rng.standard_normal(K).astype(np.float32) * 0.1
            y = np.zeros(M, dtype=np.float32)
            self.lib.gemv_fused_q5_0_bias_parallel_dispatch(
                y.ctypes.data_as(F32P), _buf(w), x.ctypes.data_as(F32P),
                bias.ctypes.data_as(F32P), M, K)
            return y

        def worker(tid: int) -> None:
            try:
                barrier.wait()
                for it in range(n_iters):
                    seed = tid * 1000 + it
                    rng = np.random.default_rng(seed)
                    w = _random_q5_0(rng, (K // QK5_0) * M)
                    x = rng.standard_normal(K).astype(np.float32) * 0.1
                    y = np.zeros(M, dtype=np.float32)
                    self.lib.gemv_fused_q5_0_bias_parallel_dispatch(
                        y.ctypes.data_as(F32P), _buf(w),
                        x.ctypes.data_as(F32P),
                        bias.ctypes.data_as(F32P), M, K)
                    if not np.array_equal(reference(seed), y):
                        errors.append(f"t{tid}i{it}: fused mismatch")
                        return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"t{tid}: exception {exc!r}")

        refs = {}
        for t in range(n_threads):
            for it in range(n_iters):
                refs[t * 1000 + it] = reference(t * 1000 + it)

        def worker(tid: int) -> None:
            try:
                barrier.wait()
                for it in range(n_iters):
                    seed = tid * 1000 + it
                    rng = np.random.default_rng(seed)
                    w = _random_q5_0(rng, (K // QK5_0) * M)
                    x = rng.standard_normal(K).astype(np.float32) * 0.1
                    y = np.zeros(M, dtype=np.float32)
                    self.lib.gemv_fused_q5_0_bias_parallel_dispatch(
                        y.ctypes.data_as(F32P), _buf(w),
                        x.ctypes.data_as(F32P),
                        bias.ctypes.data_as(F32P), M, K)
                    if not np.array_equal(refs[seed], y):
                        errors.append(f"t{tid}i{it}: fused mismatch")
                        return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"t{tid}: exception {exc!r}")

        threads = [threading.Thread(target=worker, args=(t,))
                   for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    # ---- 2. concurrent first use (fresh process) ------------------------

    def test_concurrent_first_use_dispatch_init(self) -> None:
        """Provider-name query + parallel dispatch racing on first use must
        agree with the serial reference (lazy env-cache init is atomic)."""
        code = r"""
import ctypes, threading
import numpy as np
lib = ctypes.CDLL(%r)
F32P = ctypes.POINTER(ctypes.c_float)
lib.ck_q5_0_q8_0_provider_name.restype = ctypes.c_char_p
M, K = 768, 512
rng = np.random.default_rng(7)
nb = K // 32
w = bytearray()
for _ in range(nb * M):
    w += np.array([rng.uniform(0.005, 0.05)], dtype=np.float16).tobytes()
    w += rng.integers(0, 256, size=4, dtype=np.uint8).tobytes()
    w += rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
x = bytearray()
for _ in range(nb):
    x += np.array([rng.uniform(0.005, 0.05)], dtype=np.float16).tobytes()
    x += rng.integers(-128, 128, size=32, dtype=np.int8).tobytes()
w = bytes(w); x = bytes(x)

names = []
errors = []
def worker():
    try:
        names.append(lib.ck_q5_0_q8_0_provider_name())
        y = np.zeros(M, dtype=np.float32)
        lib.gemv_q5_0_q8_0_parallel_dispatch(
            y.ctypes.data_as(F32P),
            ctypes.create_string_buffer(w, len(w)),
            ctypes.create_string_buffer(x, len(x)), M, K)
        worker_results.append(y)
    except Exception as exc:
        errors.append(repr(exc))
worker_results = []
threads = [threading.Thread(target=worker) for _ in range(8)]
for t in threads: t.start()
for t in threads: t.join()
assert not errors, errors
assert len(set(names)) == 1, names
y_seq = np.zeros(M, dtype=np.float32)
lib.gemv_q5_0_q8_0(y_seq.ctypes.data_as(F32P),
    ctypes.create_string_buffer(w, len(w)),
    ctypes.create_string_buffer(x, len(x)), M, K)
for y in worker_results:
    assert np.array_equal(y, y_seq)
print("OK", names[0].decode())
""" % (str(LIB_PATH),)
        env = dict(os.environ)
        env["CK_NUM_THREADS"] = "4"
        out = subprocess.check_output([sys.executable, "-c", code],
                                      env=env, text=True).strip()
        self.assertTrue(out.startswith("OK "), out)

    # ---- 3. repeated sequential requests are deterministic --------------

    def test_repeated_sequential_requests_deterministic(self) -> None:
        M, K = 768, 512
        w, x = self._q5_0_case(1234, M, K)
        first = None
        for _ in range(20):
            y = np.zeros(M, dtype=np.float32)
            self.lib.gemv_q5_0_q8_0_parallel_dispatch(
                y.ctypes.data_as(F32P), _buf(w), _buf(x), M, K)
            if first is None:
                first = y
            else:
                np.testing.assert_array_equal(first, y)


if __name__ == "__main__":
    unittest.main()
