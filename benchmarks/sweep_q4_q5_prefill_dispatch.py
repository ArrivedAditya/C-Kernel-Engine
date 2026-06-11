#!/usr/bin/env python3
"""Sweep Q4_K/Q5_0 prefill dispatch choices for local hardware.

This is a performance characterization runner, not a correctness gate. It
compares the raw serial prefill GEMM against the v8 threadpool dispatch wrapper
for representative gate-up shapes that dominate current Qwen2/Qwen3.5 prefill
profiles.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QK5_0 = 32
QK8_0 = 32
QK_K = 256
BLOCK_Q5_0_SIZE = 22
BLOCK_Q8_0_SIZE = 34
BLOCK_Q4_K_SIZE = 144
BLOCK_Q8_K_SIZE = 292
DEFAULT_DISPATCH_LIB = ROOT / "build" / "bench_q4_q5_prefill_dispatch.so"

SHAPES: dict[str, dict[str, Any]] = {
    "qwen2_gate_up_q5_0": {"kind": "q5_0", "N": 4864, "K": 896},
    "qwen35_gate_up_q4_k_smoke": {"kind": "q4_k", "N": 512, "K": 1024},
    "qwen35_gate_up_q4_k": {"kind": "q4_k", "N": 3584, "K": 1024},
    "nanbeige_gate_up_q4_k": {"kind": "q4_k", "N": 10496, "K": 2560},
}


def _compiler() -> str:
    cc = os.getenv("CC")
    if cc:
        return cc
    for candidate in ("icx", "clang", "gcc"):
        if subprocess.run(["sh", "-c", f"command -v {candidate} >/dev/null 2>&1"]).returncode == 0:
            return candidate
    return "cc"


def _ensure_dispatch_lib(engine_lib: Path, model_lib: Path | None) -> Path:
    if model_lib is not None:
        if not model_lib.exists():
            raise FileNotFoundError(model_lib)
        return model_lib

    engine_dir = engine_lib.resolve().parent
    suffix = engine_dir.name.replace("/", "_").replace(" ", "_")
    out = DEFAULT_DISPATCH_LIB.with_name(f"bench_q4_q5_prefill_dispatch_{suffix}.so")
    src = ROOT / "version" / "v8" / "src" / "ck_parallel_prefill_v8.c"
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _compiler(),
        "-O3",
        "-fPIC",
        "-shared",
        "-march=native",
        "-Iinclude",
        "-Iversion/v8/src",
        str(src),
        f"-L{engine_dir}",
        "-lckernel_engine",
        "-lm",
        "-lpthread",
        "-Wl,-rpath,$ORIGIN",
        "-o",
        str(out),
    ]
    print("building local v8 prefill dispatch shim:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out


def _make_q8_0_rows(rows: int, blocks: int, rng: random.Random) -> bytearray:
    buf = bytearray(rows * blocks * BLOCK_Q8_0_SIZE)
    for r in range(rows):
        for b in range(blocks):
            off = (r * blocks + b) * BLOCK_Q8_0_SIZE
            struct.pack_into("<H", buf, off, 0x3C00)  # fp16 1.0
            vals = [rng.randint(-8, 8) for _ in range(QK8_0)]
            struct.pack_into("32b", buf, off + 2, *vals)
    return buf


def _make_q5_0_rows(rows: int, blocks: int, rng: random.Random) -> bytearray:
    buf = bytearray(rows * blocks * BLOCK_Q5_0_SIZE)
    for r in range(rows):
        for b in range(blocks):
            off = (r * blocks + b) * BLOCK_Q5_0_SIZE
            struct.pack_into("<H", buf, off, 0x3C00)  # fp16 1.0
            for i in range(4):
                buf[off + 2 + i] = rng.randrange(256)
            for i in range(16):
                buf[off + 6 + i] = rng.randrange(256)
    return buf


def _make_q8_k_rows(rows: int, blocks: int, rng: random.Random) -> bytearray:
    buf = bytearray(rows * blocks * BLOCK_Q8_K_SIZE)
    for r in range(rows):
        for b in range(blocks):
            off = (r * blocks + b) * BLOCK_Q8_K_SIZE
            struct.pack_into("<f", buf, off, 0.01)
            vals = [rng.randint(-8, 8) for _ in range(QK_K)]
            struct.pack_into("256b", buf, off + 4, *vals)
            sums = [sum(vals[i:i + 16]) for i in range(0, QK_K, 16)]
            struct.pack_into("<16h", buf, off + 4 + QK_K, *sums)
    return buf


def _make_q4_k_rows(rows: int, blocks: int, rng: random.Random) -> bytearray:
    buf = bytearray(rows * blocks * BLOCK_Q4_K_SIZE)
    for r in range(rows):
        for b in range(blocks):
            off = (r * blocks + b) * BLOCK_Q4_K_SIZE
            struct.pack_into("<H", buf, off, 0x3C00)      # d = 1.0
            struct.pack_into("<H", buf, off + 2, 0x0000)  # dmin = 0.0
            for i in range(12):
                buf[off + 4 + i] = rng.randrange(64)
            for i in range(128):
                buf[off + 16 + i] = rng.randrange(256)
    return buf


def _as_ubyte_buffer(data: bytearray) -> ctypes.Array[ctypes.c_ubyte]:
    return (ctypes.c_ubyte * len(data)).from_buffer(data)


def _bind(engine: ctypes.CDLL, model: ctypes.CDLL, kind: str) -> tuple[Any, Any]:
    if kind == "q5_0":
        serial = engine.gemm_nt_q5_0_q8_0
        dispatch = model.gemm_nt_q5_0_q8_0_parallel_dispatch
    elif kind == "q4_k":
        serial = engine.gemm_nt_q4_k_q8_k
        dispatch = model.gemm_nt_q4_k_q8_k_parallel_dispatch
    else:
        raise ValueError(kind)

    for fn in (serial, dispatch):
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        fn.restype = None
    return serial, dispatch


def _run_kernel(fn: Any, a_buf: Any, b_buf: Any, c: Any, m: int, n: int, k: int, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn(a_buf, b_buf, None, c, m, n, k)
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(a_buf, b_buf, None, c, m, n, k)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _checksum(c: Any, size: int) -> float:
    return float(c[0]) + float(c[size // 2]) + float(c[size - 1])


def _run_one(engine: ctypes.CDLL, model: ctypes.CDLL, *, shape: str, kind: str, m: int, n: int, k: int,
             threads: int, warmup: int, iters: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    if kind == "q5_0":
        if k % QK5_0 != 0:
            raise ValueError(f"Q5_0 K must be divisible by {QK5_0}: {k}")
        a = _make_q8_0_rows(m, k // QK8_0, rng)
        b = _make_q5_0_rows(n, k // QK5_0, rng)
    else:
        if k % QK_K != 0:
            raise ValueError(f"Q4_K K must be divisible by {QK_K}: {k}")
        a = _make_q8_k_rows(m, k // QK_K, rng)
        b = _make_q4_k_rows(n, k // QK_K, rng)

    a_buf = _as_ubyte_buffer(a)
    b_buf = _as_ubyte_buffer(b)
    c_serial = (ctypes.c_float * (m * n))()
    c_pool = (ctypes.c_float * (m * n))()
    serial, dispatch = _bind(engine, model, kind)

    if hasattr(engine, "ck_set_num_threads"):
        engine.ck_set_num_threads.argtypes = [ctypes.c_int]
        engine.ck_set_num_threads(threads)
    if hasattr(model, "ck_parallel_prefill_init"):
        model.ck_parallel_prefill_init()

    os.environ["CK_NUM_THREADS"] = str(threads)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if kind == "q4_k":
        os.environ["CK_ENABLE_Q4K_Q8K_PREFILL_POOL"] = "1"

    serial_ms = _run_kernel(serial, a_buf, b_buf, c_serial, m, n, k, warmup, iters)
    pool_ms = _run_kernel(dispatch, a_buf, b_buf, c_pool, m, n, k, warmup, iters)
    serial_sum = _checksum(c_serial, m * n)
    pool_sum = _checksum(c_pool, m * n)
    diff = abs(serial_sum - pool_sum)
    best_serial = min(serial_ms)
    best_pool = min(pool_ms)
    speedup = best_serial / best_pool if best_pool > 0 else 0.0
    return {
        "shape": shape,
        "kind": kind,
        "M": m,
        "N": n,
        "K": k,
        "threads": threads,
        "serial_best_ms": best_serial,
        "pool_best_ms": best_pool,
        "speedup": speedup,
        "serial_avg_ms": sum(serial_ms) / len(serial_ms),
        "pool_avg_ms": sum(pool_ms) / len(pool_ms),
        "checksum_abs_diff": diff,
        "checksum_match": diff <= 1e-3,
    }


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", choices=sorted(SHAPES), help="Shape id to sweep; repeatable")
    parser.add_argument("--m-values", default="16,32,64,128,256,512")
    parser.add_argument("--threads", type=int, default=int(os.getenv("CK_NUM_THREADS", "12")))
    parser.add_argument("--thread-values", default=None, help="Comma-separated thread counts to sweep; overrides --threads")
    parser.add_argument("--engine-lib", type=Path, default=ROOT / "build" / "libckernel_engine.so")
    parser.add_argument("--model-lib", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    selected = args.shape or (
        ["qwen2_gate_up_q5_0", "qwen35_gate_up_q4_k_smoke"]
        if args.quick
        else ["qwen2_gate_up_q5_0", "qwen35_gate_up_q4_k"]
    )
    m_values = [16, 128] if args.quick else _parse_csv_ints(args.m_values)
    if args.quick:
        args.warmup = min(args.warmup, 1)
        args.iters = min(args.iters, 1)
    thread_values = _parse_csv_ints(args.thread_values) if args.thread_values else [args.threads]

    engine = ctypes.CDLL(str(args.engine_lib), mode=ctypes.RTLD_GLOBAL)
    model_path = _ensure_dispatch_lib(args.engine_lib, args.model_lib)
    model = ctypes.CDLL(str(model_path))

    results: list[dict[str, Any]] = []
    print(f"Q4_K/Q5_0 prefill dispatch sweep threads={','.join(str(t) for t in thread_values)}", flush=True)
    for threads in thread_values:
        print(f"\nthreads={threads}", flush=True)
        for shape in selected:
            spec = SHAPES[shape]
            kind = str(spec["kind"])
            n = int(spec["N"])
            k = int(spec["K"])
            print(f"\nshape={shape} kind={kind} N={n} K={k}", flush=True)
            for m in m_values:
                row = _run_one(
                    engine,
                    model,
                    shape=shape,
                    kind=kind,
                    m=m,
                    n=n,
                    k=k,
                    threads=threads,
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed,
                )
                results.append(row)
                delta = (row["serial_best_ms"] - row["pool_best_ms"]) / row["serial_best_ms"] * 100.0
                print(
                    f"T={threads:2d} M={m:4d} serial={row['serial_best_ms']:8.2f}ms "
                    f"pool={row['pool_best_ms']:8.2f}ms speed={row['speedup']:5.3f} "
                    f"delta={delta:+6.2f}% checksum={'ok' if row['checksum_match'] else 'DIFF'} "
                    f"diff={row['checksum_abs_diff']:.3e}",
                    flush=True,
                )

    report = {
        "kind": "q4_q5_prefill_dispatch_sweep",
        "threads": thread_values,
        "engine_lib": str(args.engine_lib),
        "dispatch_lib": str(model_path),
        "warmup": args.warmup,
        "iters": args.iters,
        "shapes": {name: SHAPES[name] for name in selected},
        "results": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}", flush=True)
    return 0 if all(r["checksum_match"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
