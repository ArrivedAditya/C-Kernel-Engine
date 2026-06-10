#!/usr/bin/env python3
"""Benchmark v8 Q6_K x Q8_K prefill dispatch scheduling.

This targets the Gemma4/Qwen-family MLP-down prefill shape where activations
are already Q8_K and weights are Q6_K.  The important comparison is not a new
math kernel versus old math kernel; it is orchestrator scheduling:

    row-split: split only token rows M across workers
    2d-tile:   split M x N tiles across workers, kernel owns tile math only

The script uses valid finite synthetic Q6_K/Q8_K blocks and the v8 prefill
``gemm_nt_q6_k_q8_k_parallel_dispatch`` symbol.  If a generated model
``libmodel.so`` is not provided, it builds a small local dispatch shared object
from ``version/v8/src/ck_parallel_prefill_v8.c``.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import random
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QK_K = 256
BLOCK_Q6_K_SIZE = 210
BLOCK_Q8_K_SIZE = 292
DEFAULT_DISPATCH_LIB = ROOT / 'build' / 'bench_q6k_prefill_dispatch.so'


def _make_q8_rows(rows: int, blocks: int, rng: random.Random) -> bytearray:
    buf = bytearray(rows * blocks * BLOCK_Q8_K_SIZE)
    for r in range(rows):
        for b in range(blocks):
            off = (r * blocks + b) * BLOCK_Q8_K_SIZE
            struct.pack_into('<f', buf, off, 0.01)
            vals = [rng.randint(-8, 8) for _ in range(QK_K)]
            struct.pack_into('256b', buf, off + 4, *vals)
            sums = [sum(vals[i:i + 16]) for i in range(0, QK_K, 16)]
            struct.pack_into('<16h', buf, off + 4 + QK_K, *sums)
    return buf


def _make_q6_rows(rows: int, blocks: int, rng: random.Random) -> bytearray:
    buf = bytearray(rows * blocks * BLOCK_Q6_K_SIZE)
    for r in range(rows):
        for b in range(blocks):
            off = (r * blocks + b) * BLOCK_Q6_K_SIZE
            for i in range(128):
                buf[off + i] = rng.randrange(256)
            for i in range(64):
                buf[off + 128 + i] = rng.randrange(256)
            for i in range(16):
                buf[off + 192 + i] = rng.randint(-3, 3) & 0xff
            buf[off + 208:off + 210] = b'\x00<'  # fp16 1.0
    return buf


def _compiler() -> str:
    for env_name in ('CC',):
        cc = os.getenv(env_name)
        if cc:
            return cc
    for cc in ('icx', 'clang', 'gcc'):
        if subprocess.run(['sh', '-c', f'command -v {cc} >/dev/null 2>&1']).returncode == 0:
            return cc
    return 'cc'


def _ensure_dispatch_lib(engine_lib: Path, model_lib: Path | None) -> Path:
    if model_lib is not None:
        if not model_lib.exists():
            raise FileNotFoundError(model_lib)
        return model_lib

    engine_dir = engine_lib.resolve().parent
    suffix = engine_dir.name.replace('/', '_').replace(' ', '_')
    out = DEFAULT_DISPATCH_LIB.with_name(f'bench_q6k_prefill_dispatch_{suffix}.so')
    src = ROOT / 'version' / 'v8' / 'src' / 'ck_parallel_prefill_v8.c'
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _compiler(),
        '-O3',
        '-fPIC',
        '-shared',
        '-march=native',
        '-Iinclude',
        '-Iversion/v8/src',
        str(src),
        f'-L{engine_dir}',
        '-lckernel_engine',
        '-lm',
        '-lpthread',
        '-Wl,-rpath,$ORIGIN',
        '-o',
        str(out),
    ]
    print('building local v8 prefill dispatch shim:')
    print('  ' + ' '.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out


def _run_one(args: argparse.Namespace) -> int:
    if args.k % QK_K != 0:
        raise ValueError(f'--k must be divisible by {QK_K}')
    blocks = args.k // QK_K
    rng = random.Random(args.seed)

    a = _make_q8_rows(args.m, blocks, rng)
    b = _make_q6_rows(args.n, blocks, rng)
    c = (ctypes.c_float * (args.m * args.n))()
    a_buf = (ctypes.c_ubyte * len(a)).from_buffer(a)
    b_buf = (ctypes.c_ubyte * len(b)).from_buffer(b)

    engine = ctypes.CDLL(str(args.engine_lib), mode=ctypes.RTLD_GLOBAL)
    model_path = _ensure_dispatch_lib(args.engine_lib, args.model_lib)
    model = ctypes.CDLL(str(model_path))
    if hasattr(engine, 'ck_set_num_threads'):
        engine.ck_set_num_threads.argtypes = [ctypes.c_int]
        engine.ck_set_num_threads(args.threads)
    if hasattr(model, 'ck_parallel_prefill_init'):
        model.ck_parallel_prefill_init()

    fn = model.gemm_nt_q6_k_q8_k_parallel_dispatch
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

    for _ in range(args.warmup):
        fn(a_buf, b_buf, None, c, args.m, args.n, args.k)

    times: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        fn(a_buf, b_buf, None, c, args.m, args.n, args.k)
        times.append(time.perf_counter() - t0)

    checksum = float(c[0]) + float(c[(args.m * args.n) // 2]) + float(c[args.m * args.n - 1])
    print(
        f'mode={args.mode} threads={args.threads} M={args.m} N={args.n} K={args.k} '
        f'tile_m={os.getenv("CK_PREFILL_TILE_M", "")} tile_n={os.getenv("CK_PREFILL_TILE_N", "")}'
    )
    print('times_ms=' + ','.join(f'{t * 1000.0:.2f}' for t in times))
    print(f'best_ms={min(times) * 1000.0:.2f} avg_ms={sum(times) * 1000.0 / len(times):.2f}')
    print(f'checksum={checksum:.6f}')
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    base = [
        sys.executable,
        str(script),
        '--mode', 'row',
        '--m', str(args.m), '--n', str(args.n), '--k', str(args.k),
        '--threads', str(args.threads), '--warmup', str(args.warmup), '--iters', str(args.iters),
        '--seed', str(args.seed), '--engine-lib', str(args.engine_lib),
    ]
    if args.model_lib is not None:
        base.extend(['--model-lib', str(args.model_lib)])
    env = os.environ.copy()
    env.setdefault('CK_Q6K_Q8K_SIMD', '1')
    env.setdefault('OMP_NUM_THREADS', '1')
    env['CK_NUM_THREADS'] = str(args.threads)

    print('row-split baseline:')
    row = subprocess.run(base, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    print(row.stdout.rstrip())

    env2 = env.copy()
    env2['CK_ENABLE_Q6K_Q8K_2D_PREFILL'] = '1'
    env2.setdefault('CK_PREFILL_TILE_M', str(args.tile_m))
    env2.setdefault('CK_PREFILL_TILE_N', str(args.tile_n))
    tiled_cmd = base.copy()
    tiled_cmd[tiled_cmd.index('row')] = '2d'
    print('\n2d-tile candidate:')
    tiled = subprocess.run(tiled_cmd, env=env2, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    print(tiled.stdout.rstrip())
    return 0 if row.returncode == 0 and tiled.returncode == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--m', type=int, default=1024, help='token rows')
    ap.add_argument('--n', type=int, default=2560, help='output rows/features')
    ap.add_argument('--k', type=int, default=10240, help='input dimension')
    ap.add_argument('--threads', type=int, default=int(os.getenv('CK_NUM_THREADS', '24')))
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--iters', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--tile-m', type=int, default=16)
    ap.add_argument('--tile-n', type=int, default=256)
    ap.add_argument('--engine-lib', type=Path, default=ROOT / 'build' / 'libckernel_engine.so')
    ap.add_argument('--model-lib', type=Path, default=None)
    ap.add_argument('--mode', choices=('compare', 'row', '2d'), default='compare')
    args = ap.parse_args()

    if args.mode == 'compare':
        return _run_compare(args)
    return _run_one(args)


if __name__ == '__main__':
    raise SystemExit(main())
