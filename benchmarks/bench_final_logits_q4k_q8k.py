#!/usr/bin/env python3
"""Benchmark Gemma4-style quantized decode hot spots.

This is intentionally focused on the expensive final-head shape:

    logits[vocab] = token_embedding_q4_k[vocab, hidden] @ hidden_q8_k[hidden]

For Gemma4 E4B Q4_K_M final logits is typically vocab=262144, hidden=2560.
The MLP hot spots are also included:

    mlp_gate/up: Q4_K x Q8_K, M=10240, K=2560, called twice per layer
    mlp_down:    Q6_K x Q8_K, M=2560,  K=10240

The script times CK's exported kernels on synthetic Q4_K/Q8_K buffers with the
same byte layout and shape as the generated model.  It can also run llama-bench
for a model-level tokens/sec baseline, but llama.cpp does not expose a standalone
final-head kernel through the CLI, so that row is not an apples-to-apples kernel
measurement.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
QK_K = 256
BLOCK_Q4_K_SIZE = 144
BLOCK_Q6_K_SIZE = 210
BLOCK_Q8_K_SIZE = 292


def _load_lib(path: Path) -> ctypes.CDLL:
    if not path.exists():
        raise FileNotFoundError(path)
    return ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def _bind_gemv(lib: ctypes.CDLL, name: str):
    fn = getattr(lib, name, None)
    if fn is None:
        return None
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    fn.restype = None
    return fn


def _bind_gemma4_embed(lib: ctypes.CDLL):
    fn = getattr(lib, "gemma4_per_layer_embed_forward", None)
    if fn is None:
        return None
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    fn.restype = None
    return fn


def _bind_gemma4_prepare(lib: ctypes.CDLL):
    fn = getattr(lib, "gemma4_per_layer_prepare_forward", None)
    if fn is None:
        return None
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    fn.restype = None
    return fn


def _make_bytes(size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, size=size, dtype=np.uint8)


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / max(1, iters)


def _run_llama_bench(llama_bench: Path, model: Path, threads: int, prompt: int, gen: int) -> None:
    if not llama_bench.exists() or not model.exists():
        return
    cmd = [
        str(llama_bench),
        "-m",
        str(model),
        "-t",
        str(threads),
        "-ngl",
        "0",
        "-p",
        str(prompt),
        "-n",
        str(gen),
        "-r",
        "1",
        "-o",
        "md",
    ]
    print("\nllama.cpp model-level baseline:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=262144)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--threads", type=int, default=int(os.getenv("CK_NUM_THREADS", "24")))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--engine-lib", type=Path, default=ROOT / "build" / "libckernel_engine.so")
    ap.add_argument("--model-lib", type=Path, default=None, help="Optional generated libmodel.so for v8 parallel dispatch symbols")
    ap.add_argument("--llama-bench", type=Path, default=Path("/opt/app-root/src/Software/llama.cpp/build/bin/llama-bench"))
    ap.add_argument("--llama-model", type=Path, default=None)
    ap.add_argument("--llama-prompt", type=int, default=32)
    ap.add_argument("--llama-gen", type=int, default=32)
    args = ap.parse_args()

    if args.hidden % QK_K != 0:
        raise ValueError(f"--hidden must be divisible by {QK_K}")

    blocks = args.hidden // QK_K
    w_bytes = args.vocab * blocks * BLOCK_Q4_K_SIZE
    x_bytes = blocks * BLOCK_Q8_K_SIZE

    print("CK final logits Q4_K x Q8_K benchmark")
    print(f"  shape: M/vocab={args.vocab} K/hidden={args.hidden} blocks={blocks}")
    print(f"  buffers: W={w_bytes / (1024 ** 2):.1f} MiB x_q8={x_bytes} bytes y={args.vocab * 4 / (1024 ** 2):.1f} MiB")
    print(f"  threads env: CK_NUM_THREADS={os.getenv('CK_NUM_THREADS', '<unset>')} OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS', '<unset>')}")

    rng = np.random.default_rng(args.seed)
    w = _make_bytes(w_bytes, rng)
    x = _make_bytes(x_bytes, rng)
    y = np.zeros(args.vocab, dtype=np.float32)

    engine = _load_lib(args.engine_lib)
    model_lib = None
    if args.model_lib is not None and args.model_lib.exists():
        model_lib = _load_lib(args.model_lib)

    kernels: list[tuple[str, object]] = []
    for name in ("gemv_q4_k_q8_k", "gemv_q4_k_q8_k_ref", "gemv_q4_k_q8_k_vnni"):
        fn = _bind_gemv(engine, name)
        if fn is not None:
            kernels.append((f"engine.{name}", fn))

    if model_lib is not None:
        fn = _bind_gemv(model_lib, "gemv_q4_k_q8_k_parallel_dispatch")
        if fn is not None:
            kernels.append(("model.gemv_q4_k_q8_k_parallel_dispatch", fn))

    if not kernels:
        raise RuntimeError("no Q4_K/Q8_K kernels found")

    w_ptr = w.ctypes.data_as(ctypes.c_void_p)
    x_ptr = x.ctypes.data_as(ctypes.c_void_p)
    y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    print("\nCK isolated final-head timings:")
    print("  note: lower is better; token/s-equivalent is 1000/ms for this one GEMV")
    for label, fn in kernels:
        def call() -> None:
            fn(y_ptr, w_ptr, x_ptr, ctypes.c_int(args.vocab), ctypes.c_int(args.hidden))

        ms = _time_ms(call, args.warmup, args.iters)
        print(f"  {label:<46} {ms:10.3f} ms  {1000.0 / ms:8.2f} calls/s")


    if model_lib is not None:
        q4_dispatch = _bind_gemv(model_lib, "gemv_q4_k_q8_k_parallel_dispatch")
        q6_dispatch = _bind_gemv(model_lib, "gemv_q6_k_q8_k_parallel_dispatch")
        q6_engine = _bind_gemv(engine, "gemv_q6_k_q8_k")

        print("\nCK Gemma4 decode hot-shape timings:")

        def bench_shape(label: str, fn, weight_block_size: int, m: int, k: int, calls_per_layer: int = 1) -> float:
            shape_blocks = k // QK_K
            ww = _make_bytes(m * shape_blocks * weight_block_size, rng)
            xx = _make_bytes(shape_blocks * BLOCK_Q8_K_SIZE, rng)
            yy = np.zeros(m, dtype=np.float32)
            ww_ptr = ww.ctypes.data_as(ctypes.c_void_p)
            xx_ptr = xx.ctypes.data_as(ctypes.c_void_p)
            yy_ptr = yy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            def call() -> None:
                fn(yy_ptr, ww_ptr, xx_ptr, ctypes.c_int(m), ctypes.c_int(k))

            ms = _time_ms(call, args.warmup, args.iters)
            total = ms * calls_per_layer
            print(f"  {label:<46} {ms:10.3f} ms/call  layer-cost~{total:8.3f} ms")
            return total

        if q4_dispatch is not None:
            bench_shape("mlp_gate/up q4 dispatch M=10240 K=2560", q4_dispatch, BLOCK_Q4_K_SIZE, 10240, 2560, 2)
        if q6_dispatch is not None:
            bench_shape("mlp_down q6 dispatch M=2560 K=10240", q6_dispatch, BLOCK_Q6_K_SIZE, 2560, 10240, 1)
        if q6_engine is not None:
            bench_shape("mlp_down q6 direct engine M=2560 K=10240", q6_engine, BLOCK_Q6_K_SIZE, 2560, 10240, 1)


        embed_fn = _bind_gemma4_embed(engine)
        prepare_fn = _bind_gemma4_prepare(engine)
        if embed_fn is not None or prepare_fn is not None:
            print("\nCK Gemma4 per-layer embedding timings:")
            tokens = 1
            num_layers = 42
            embed_dim = 2560
            per_layer_dim = 256
            eps = 1.0e-6
            hidden = rng.standard_normal(tokens * embed_dim, dtype=np.float32) * np.float32(0.01)
            per_layer_input = rng.standard_normal(tokens * num_layers * per_layer_dim, dtype=np.float32) * np.float32(0.01)
            inp_gate = rng.standard_normal(per_layer_dim * embed_dim, dtype=np.float32) * np.float32(0.01)
            proj = rng.standard_normal(embed_dim * per_layer_dim, dtype=np.float32) * np.float32(0.01)
            post_norm = np.ones(embed_dim, dtype=np.float32)
            out_scale = np.ones(1, dtype=np.float32)

            if embed_fn is not None:
                def call_embed_one() -> None:
                    embed_fn(
                        hidden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        per_layer_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        inp_gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        post_norm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        out_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.c_int(tokens),
                        ctypes.c_int(0),
                        ctypes.c_int(num_layers),
                        ctypes.c_int(embed_dim),
                        ctypes.c_int(per_layer_dim),
                        ctypes.c_float(eps),
                    )

                ms = _time_ms(call_embed_one, args.warmup, args.iters)
                print(f"  gemma4_per_layer_embed_forward one layer      {ms:10.3f} ms/call  42-layer~{ms * num_layers:8.3f} ms")

            if prepare_fn is not None:
                out = np.zeros(tokens * num_layers * per_layer_dim, dtype=np.float32)
                token_ids = np.zeros(tokens, dtype=np.int32)
                q5_token_blocks = _make_bytes(num_layers * 176, rng)
                model_proj_bf16 = np.full(num_layers * per_layer_dim * embed_dim, 0x3c23, dtype=np.uint16)
                proj_norm = np.ones(per_layer_dim, dtype=np.float32)

                def call_prepare() -> None:
                    prepare_fn(
                        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        hidden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        token_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        q5_token_blocks.ctypes.data_as(ctypes.c_void_p),
                        model_proj_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                        proj_norm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.c_int(tokens),
                        ctypes.c_int(num_layers),
                        ctypes.c_int(embed_dim),
                        ctypes.c_int(per_layer_dim),
                        ctypes.c_int(1),
                        ctypes.c_float(eps),
                    )

                ms = _time_ms(call_prepare, args.warmup, args.iters)
                print(f"  gemma4_per_layer_prepare_forward token        {ms:10.3f} ms/call")

    if args.llama_model is not None:
        _run_llama_bench(args.llama_bench, args.llama_model, args.threads, args.llama_prompt, args.llama_gen)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
