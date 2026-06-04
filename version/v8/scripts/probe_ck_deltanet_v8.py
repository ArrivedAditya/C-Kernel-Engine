#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np


def _load_f32(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size == 0:
        raise ValueError(f"empty float32 file: {path}")
    return np.ascontiguousarray(data, dtype=np.float32)


def _stats(actual: np.ndarray, expected: np.ndarray) -> str:
    a = actual.astype(np.float64, copy=False).ravel()
    b = expected.astype(np.float64, copy=False).ravel()
    n = min(a.size, b.size)
    if n == 0:
        raise ValueError("cannot compare empty arrays")
    a = a[:n]
    b = b[:n]
    diff = a - b
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    cosine = float(np.dot(a, b) / (na * nb)) if na and nb else 0.0
    return (
        f"sizes={actual.size}/{expected.size} cosine={cosine:.9f} "
        f"rmse={float(np.sqrt(np.mean(diff * diff))):.9f} "
        f"mean_abs={float(np.mean(np.abs(diff))):.9f} "
        f"max_abs={float(np.max(np.abs(diff))):.9f} "
        f"max_idx={int(np.argmax(np.abs(diff)))}"
    )


def _ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe CK DeltaNet with external float32 tensors")
    ap.add_argument("--lib", type=Path, default=Path("build/libckernel_engine.so"))
    ap.add_argument("--q", required=True, type=Path)
    ap.add_argument("--k", required=True, type=Path)
    ap.add_argument("--v", required=True, type=Path)
    ap.add_argument("--g", required=True, type=Path)
    ap.add_argument("--beta", required=True, type=Path)
    ap.add_argument("--state-in", required=True, type=Path)
    ap.add_argument("--expected-out", required=True, type=Path)
    ap.add_argument("--expected-state", required=True, type=Path)
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--state-dim", type=int, default=128)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument(
        "--llama-state-layout",
        action="store_true",
        help="treat state files as llama [head,col,row] dumps and transpose to CK layout",
    )
    args = ap.parse_args()

    lib = ctypes.CDLL(str(args.lib.resolve()))
    fn = lib.gated_deltanet_autoregressive_forward
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    fn.restype = None

    h = int(args.num_heads)
    d = int(args.state_dim)
    vec_size = h * d
    state_size = h * d * d

    q = _load_f32(args.q)
    k = _load_f32(args.k)
    v = _load_f32(args.v)
    g = _load_f32(args.g)
    beta = _load_f32(args.beta)
    state_in = _load_f32(args.state_in)
    expected_out = _load_f32(args.expected_out)
    expected_state = _load_f32(args.expected_state)

    if q.size != vec_size or k.size != vec_size or v.size != vec_size:
        raise ValueError(f"q/k/v must have {vec_size} floats")
    if g.size != h or beta.size != h:
        raise ValueError(f"g/beta must have {h} floats")
    if state_in.size != state_size or expected_state.size != state_size:
        raise ValueError(f"state tensors must have {state_size} floats")
    if expected_out.size != vec_size:
        raise ValueError(f"expected output must have {vec_size} floats")

    if args.llama_state_layout:
        state_in = np.ascontiguousarray(state_in.reshape(h, d, d).transpose(0, 2, 1).ravel(), dtype=np.float32)
        expected_state_ck = np.ascontiguousarray(
            expected_state.reshape(h, d, d).transpose(0, 2, 1).ravel(),
            dtype=np.float32,
        )
    else:
        expected_state_ck = expected_state

    state_out = np.empty(state_size, dtype=np.float32)
    out = np.empty(vec_size, dtype=np.float32)
    fn(
        _ptr(q),
        _ptr(k),
        _ptr(v),
        _ptr(g),
        _ptr(beta),
        _ptr(state_in),
        _ptr(state_out),
        _ptr(out),
        h,
        d,
        ctypes.c_float(float(args.eps)),
    )

    print("out", _stats(out, expected_out))
    print("state_ck_layout", _stats(state_out, expected_state_ck))
    if args.llama_state_layout:
        state_out_llama = np.ascontiguousarray(state_out.reshape(h, d, d).transpose(0, 2, 1).ravel(), dtype=np.float32)
        print("state_llama_layout", _stats(state_out_llama, expected_state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
