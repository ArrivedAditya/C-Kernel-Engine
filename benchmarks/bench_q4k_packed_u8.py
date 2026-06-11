#!/usr/bin/env python3
"""Benchmark an experimental packed Q4_K x Q8_K prefill GEMM path.

The packed path expands Q4 nibbles and Q4_K scale/min metadata once, then
measures whether the GEMM hot loop is faster than the current in-place Q4_K
kernel. It is intentionally standalone and not wired into production dispatch.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libckernel_engine.so"
QK_K = 256
Q4K_BLOCK = 144
Q8K_BLOCK = 292


def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(LIB))
    lib.gemm_nt_q4_k_q8_k.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemm_nt_q4_k_q8_k.restype = None
    lib.q4_k_packed_u8_block_size.argtypes = []
    lib.q4_k_packed_u8_block_size.restype = ctypes.c_size_t
    lib.q4_k_packed_meta_block_size.argtypes = []
    lib.q4_k_packed_meta_block_size.restype = ctypes.c_size_t
    lib.pack_q4_k_to_packed_u8.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.pack_q4_k_to_packed_u8.restype = None
    lib.pack_q4_k_to_packed_meta.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.pack_q4_k_to_packed_meta.restype = None
    lib.gemm_nt_q4_k_packed_u8_q8_k.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemm_nt_q4_k_packed_u8_q8_k.restype = None
    lib.gemm_nt_q4_k_packed_meta_q8_k.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemm_nt_q4_k_packed_meta_q8_k.restype = None
    lib.gemm_nt_q4_k_packed_meta_q8_k_threaded.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemm_nt_q4_k_packed_meta_q8_k_threaded.restype = None
    lib.gemm_nt_q4_k_packed_meta_q8_k_threaded_nsplit.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemm_nt_q4_k_packed_meta_q8_k_threaded_nsplit.restype = None
    return lib


def _ptr(a: np.ndarray) -> ctypes.c_void_p:
    return ctypes.c_void_p(a.ctypes.data)


def _f16_bytes(value: float) -> bytes:
    return np.float16(value).tobytes()


def make_q4k_weights(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    blocks = n * (k // QK_K)
    data = np.zeros(blocks * Q4K_BLOCK, dtype=np.uint8)
    for b in range(blocks):
        off = b * Q4K_BLOCK
        data[off : off + 2] = np.frombuffer(_f16_bytes(0.03125), dtype=np.uint8)
        data[off + 2 : off + 4] = np.frombuffer(_f16_bytes(0.00390625), dtype=np.uint8)
        data[off + 4 : off + 16] = rng.integers(0, 64, size=12, dtype=np.uint8)
        data[off + 16 : off + 144] = rng.integers(0, 256, size=128, dtype=np.uint8)
    return data


def make_q8k_acts(m: int, k: int, rng: np.random.Generator) -> np.ndarray:
    blocks = m * (k // QK_K)
    data = np.zeros(blocks * Q8K_BLOCK, dtype=np.uint8)
    for b in range(blocks):
        off = b * Q8K_BLOCK
        data[off : off + 4] = np.frombuffer(np.float32(0.03125).tobytes(), dtype=np.uint8)
        qs = rng.integers(-32, 33, size=QK_K, dtype=np.int16).astype(np.int8)
        data[off + 4 : off + 260] = qs.view(np.uint8)
        bsums = np.array([int(qs[i : i + 16].sum()) for i in range(0, QK_K, 16)], dtype=np.int16)
        data[off + 260 : off + 292] = bsums.view(np.uint8)
    return data


def median_ms(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1.0e6)
    return statistics.median(times)


def run_shape(lib: ctypes.CDLL, m: int, n: int, k: int, repeats: int, seed: int, threads: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = make_q8k_acts(m, k, rng)
    w = make_q4k_weights(n, k, rng)
    bias = rng.normal(0.0, 0.01, size=n).astype(np.float32)
    c_base = np.zeros((m, n), dtype=np.float32)
    c_packed = np.zeros((m, n), dtype=np.float32)
    c_meta = np.zeros((m, n), dtype=np.float32)
    c_meta_threaded = np.zeros((m, n), dtype=np.float32)
    c_meta_nsplit = np.zeros((m, n), dtype=np.float32)

    packed_block = int(lib.q4_k_packed_u8_block_size())
    packed = np.zeros(n * (k // QK_K) * packed_block, dtype=np.uint8)
    meta_block = int(lib.q4_k_packed_meta_block_size())
    meta = np.zeros(n * (k // QK_K) * meta_block, dtype=np.uint8)

    def pack_u8_once() -> None:
        lib.pack_q4_k_to_packed_u8(_ptr(w), _ptr(packed), n, k)

    def pack_meta_once() -> None:
        lib.pack_q4_k_to_packed_meta(_ptr(w), _ptr(meta), n, k)

    pack_u8_once()
    pack_meta_once()
    lib.gemm_nt_q4_k_q8_k(_ptr(a), _ptr(w), _ptr(bias), _ptr(c_base), m, n, k)
    lib.gemm_nt_q4_k_packed_u8_q8_k(_ptr(a), _ptr(packed), _ptr(bias), _ptr(c_packed), m, n, k)
    lib.gemm_nt_q4_k_packed_meta_q8_k(_ptr(a), _ptr(meta), _ptr(bias), _ptr(c_meta), m, n, k)
    lib.gemm_nt_q4_k_packed_meta_q8_k_threaded(_ptr(a), _ptr(meta), _ptr(bias), _ptr(c_meta_threaded), m, n, k, threads)
    lib.gemm_nt_q4_k_packed_meta_q8_k_threaded_nsplit(_ptr(a), _ptr(meta), _ptr(bias), _ptr(c_meta_nsplit), m, n, k, threads)

    diff_u8 = np.abs(c_base - c_packed)
    diff_meta = np.abs(c_base - c_meta)
    diff_meta_threaded = np.abs(c_base - c_meta_threaded)
    diff_meta_nsplit = np.abs(c_base - c_meta_nsplit)

    base_ms = median_ms(
        lambda: lib.gemm_nt_q4_k_q8_k(_ptr(a), _ptr(w), _ptr(bias), _ptr(c_base), m, n, k),
        repeats,
    )
    pack_u8_ms = median_ms(pack_u8_once, repeats)
    packed_u8_ms = median_ms(
        lambda: lib.gemm_nt_q4_k_packed_u8_q8_k(_ptr(a), _ptr(packed), _ptr(bias), _ptr(c_packed), m, n, k),
        repeats,
    )
    pack_meta_ms = median_ms(pack_meta_once, repeats)
    packed_meta_ms = median_ms(
        lambda: lib.gemm_nt_q4_k_packed_meta_q8_k(_ptr(a), _ptr(meta), _ptr(bias), _ptr(c_meta), m, n, k),
        repeats,
    )
    packed_meta_threaded_ms = median_ms(
        lambda: lib.gemm_nt_q4_k_packed_meta_q8_k_threaded(_ptr(a), _ptr(meta), _ptr(bias), _ptr(c_meta_threaded), m, n, k, threads),
        repeats,
    )
    packed_meta_nsplit_ms = median_ms(
        lambda: lib.gemm_nt_q4_k_packed_meta_q8_k_threaded_nsplit(_ptr(a), _ptr(meta), _ptr(bias), _ptr(c_meta_nsplit), m, n, k, threads),
        repeats,
    )
    return {
        "M": float(m),
        "N": float(n),
        "K": float(k),
        "max_abs_u8": float(diff_u8.max()),
        "max_abs_meta": float(diff_meta.max()),
        "max_abs_meta_threaded": float(diff_meta_threaded.max()),
        "max_abs_meta_nsplit": float(diff_meta_nsplit.max()),
        "base_ms": base_ms,
        "pack_u8_ms": pack_u8_ms,
        "packed_u8_ms": packed_u8_ms,
        "pack_meta_ms": pack_meta_ms,
        "packed_meta_ms": packed_meta_ms,
        "packed_meta_threaded_ms": packed_meta_threaded_ms,
        "packed_meta_nsplit_ms": packed_meta_nsplit_ms,
        "u8_speedup": base_ms / packed_u8_ms if packed_u8_ms > 0 else 0.0,
        "meta_speedup": base_ms / packed_meta_ms if packed_meta_ms > 0 else 0.0,
        "meta_threaded_speedup": base_ms / packed_meta_threaded_ms if packed_meta_threaded_ms > 0 else 0.0,
        "meta_nsplit_speedup": base_ms / packed_meta_nsplit_ms if packed_meta_nsplit_ms > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=0, help="Active CK threads for packed-meta threaded wrapper; 0 uses CK pool size")
    parser.add_argument(
        "--shapes",
        default="4x128x1024,16x512x2048,32x896x4864,64x1024x2560",
        help="Comma-separated MxNxK shapes. K must be divisible by 256.",
    )
    args = parser.parse_args()

    if not LIB.exists():
        raise SystemExit(f"missing {LIB}; run make build/libckernel_engine.so first")

    os.environ.setdefault("CK_ENABLE_Q4K_Q8K_VNNI_FAST", "1")
    lib = _load_lib()
    print("Q4_K packing experiments; lower ms is better")
    print(
        f"{'M':>5} {'N':>6} {'K':>6} {'M/th':>6} {'N/th':>6} {'base':>9} "
        f"{'M-split':>9} {'M x':>7} {'N-split':>9} {'N x':>7} {'max abs':>9}"
    )
    for shape in args.shapes.split(","):
        m_s, n_s, k_s = shape.lower().split("x")
        m = int(m_s)
        n = int(n_s)
        k = int(k_s)
        row = run_shape(lib, m, n, k, args.repeats, args.seed, args.threads)
        display_threads = args.threads if args.threads > 0 else int(os.environ.get("CK_NUM_THREADS", "24"))
        m_per_thread = (m + max(display_threads, 1) - 1) // max(display_threads, 1)
        n_per_thread = (n + max(display_threads, 1) - 1) // max(display_threads, 1)
        print(
            f"{m:5d} {n:6d} {k:6d} {m_per_thread:6d} {n_per_thread:6d} "
            f"{row['base_ms']:9.3f} {row['packed_meta_threaded_ms']:9.3f} {row['meta_threaded_speedup']:7.3f} "
            f"{row['packed_meta_nsplit_ms']:9.3f} {row['meta_nsplit_speedup']:7.3f} "
            f"{max(row['max_abs_u8'], row['max_abs_meta'], row['max_abs_meta_threaded'], row['max_abs_meta_nsplit']):9.3g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
