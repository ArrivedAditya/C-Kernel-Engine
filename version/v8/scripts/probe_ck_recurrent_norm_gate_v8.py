#!/usr/bin/env python3
from __future__ import annotations

"""Probe CK recurrent_norm_gate_forward with external FP32 tensors."""

import argparse
import ctypes
import re
from pathlib import Path

import numpy as np

from compare_first_token_logits_v8 import discover_ck_model_dir  # type: ignore


def _parse_define(c_path: Path, name: str) -> int:
    pattern = re.compile(rf"^\s*#define\s+{re.escape(name)}\s+([0-9]+)\s*$")
    for line in c_path.read_text(errors="replace").splitlines():
        m = pattern.match(line)
        if m:
            return int(m.group(1))
    raise KeyError(f"missing define {name} in {c_path}")


def _compare(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(int(a.size), int(b.size))
    af = a[:n].astype(np.float64)
    bf = b[:n].astype(np.float64)
    diff = af - bf
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    cosine = float(np.dot(af, bf) / denom) if denom else float("nan")
    max_idx = int(np.argmax(np.abs(diff)))
    return {
        "compared": n,
        "cosine": cosine,
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "max_idx": max_idx,
        "actual_at_max": float(af[max_idx]),
        "expected_at_max": float(bf[max_idx]),
    }


def _init_model(lib: ctypes.CDLL, model_dir: Path) -> None:
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    tried: list[str] = []
    for init_path in (model_dir / "weights.bump", model_dir):
        rc = lib.ck_model_init(str(init_path.resolve()).encode("utf-8"))
        tried.append(f"{init_path}:rc={rc}")
        if rc == 0:
            return
    raise RuntimeError(f"ck_model_init failed: {', '.join(tried)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--weight-symbol", required=True, help="generated C define, e.g. W_LAYER_0_SSM_NORM")
    ap.add_argument("--x", required=True, type=Path, help="float32 recurrent attention output")
    ap.add_argument("--gate", required=True, type=Path, help="float32 z/gate vector")
    ap.add_argument("--expected", type=Path, help="optional float32 expected final_output vector")
    ap.add_argument("--output", type=Path, help="optional float32 output path")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--num-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--eps", type=float, default=1.0e-6)
    args = ap.parse_args()

    model_dir = discover_ck_model_dir(args.model_dir)
    c_path = model_dir / "model_v8.c"
    weight_offset = _parse_define(c_path, args.weight_symbol)

    engine_path = Path("build/libckernel_engine.so").resolve()
    ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str((model_dir / "libmodel.so").resolve()), mode=ctypes.RTLD_GLOBAL)
    _init_model(lib, model_dir)

    try:
        lib.ck_model_get_base_ptr.restype = ctypes.c_uint64
        base = int(lib.ck_model_get_base_ptr())
        if base == 0:
            raise RuntimeError("ck_model_get_base_ptr returned null")

        kernel = lib.recurrent_norm_gate_forward
        kernel.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        kernel.restype = None

        x = np.ascontiguousarray(np.fromfile(args.x, dtype=np.float32), dtype=np.float32)
        gate = np.ascontiguousarray(np.fromfile(args.gate, dtype=np.float32), dtype=np.float32)
        expected_size = int(args.rows) * int(args.num_heads) * int(args.head_dim)
        if int(x.size) != expected_size:
            raise ValueError(f"x has {x.size} floats, expected {expected_size}")
        if int(gate.size) != expected_size:
            raise ValueError(f"gate has {gate.size} floats, expected {expected_size}")

        y = np.zeros(expected_size, dtype=np.float32)
        kernel(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(ctypes.c_void_p(base + weight_offset), ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(args.rows),
            int(args.num_heads),
            int(args.head_dim),
            ctypes.c_float(float(args.eps)),
        )

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            y.tofile(args.output)

        print(
            f"kernel=recurrent_norm_gate_forward weight_symbol={args.weight_symbol} "
            f"weight_offset={weight_offset} rows={args.rows} num_heads={args.num_heads} "
            f"head_dim={args.head_dim} eps={args.eps:g} output_norm={float(np.linalg.norm(y)):.9f}"
        )
        if args.expected:
            expected = np.fromfile(args.expected, dtype=np.float32)
            result = _compare(y, expected)
            print(
                f"sizes={y.size}/{expected.size} cosine={result['cosine']:.9f} "
                f"rmse={result['rmse']:.9f} mean_abs={result['mean_abs']:.9f} "
                f"max_abs={result['max_abs']:.9f} max_idx={result['max_idx']} "
                f"actual_at_max={result['actual_at_max']:.9f} "
                f"expected_at_max={result['expected_at_max']:.9f}"
            )
    finally:
        if hasattr(lib, "ck_model_free"):
            lib.ck_model_free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
