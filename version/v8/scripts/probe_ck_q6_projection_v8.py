#!/usr/bin/env python3
from __future__ import annotations

"""Probe a generated CK quantized projection with an external FP32 input vector."""

import argparse
import ctypes
import re
from pathlib import Path

import numpy as np

from compare_first_token_logits_v8 import discover_ck_model_dir  # type: ignore


QK_K = 256
BLOCK_Q8_K_SIZE = 292


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
    ap.add_argument("--weight-symbol", required=True, help="generated C define, e.g. W_LAYER_23_FFN_DOWN")
    ap.add_argument("--input", required=True, type=Path, help="float32 input vector")
    ap.add_argument("--expected", type=Path, help="optional float32 expected output vector")
    ap.add_argument("--output", type=Path, help="optional float32 output path")
    ap.add_argument("--rows", type=int, required=True, help="projection output rows")
    ap.add_argument("--cols", type=int, required=True, help="projection input cols")
    ap.add_argument(
        "--kernel",
        choices=("q6_k_q8_k", "q4_k_q8_k", "q5_k", "q5_k_q8_k"),
        default="q6_k_q8_k",
        help="CK quantized projection kernel to call",
    )
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

        lib.quantize_row_q8_k.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int]
        lib.quantize_row_q8_k.restype = None
        kernel_name_by_id = {
            "q6_k_q8_k": "gemv_q6_k_q8_k",
            "q4_k_q8_k": "gemv_q4_k_q8_k",
            "q5_k": "gemv_q5_k",
            "q5_k_q8_k": "gemv_q5_k_q8_k",
        }
        kernel_name = kernel_name_by_id[args.kernel]
        kernel = getattr(lib, kernel_name)
        input_is_q8 = args.kernel in ("q6_k_q8_k", "q4_k_q8_k", "q5_k_q8_k")
        input_arg_type = ctypes.c_void_p if input_is_q8 else ctypes.POINTER(ctypes.c_float)
        kernel.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, input_arg_type, ctypes.c_int, ctypes.c_int]
        kernel.restype = None

        x = np.fromfile(args.input, dtype=np.float32)
        if int(x.size) != int(args.cols):
            raise ValueError(f"input has {x.size} floats, expected {args.cols}")
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.zeros(int(args.rows), dtype=np.float32)
        if input_is_q8:
            if int(args.cols) % QK_K != 0:
                raise ValueError(f"Q8_K probe input cols must be divisible by {QK_K}, got {args.cols}")
            q8_bytes = (int(args.cols) // QK_K) * BLOCK_Q8_K_SIZE
            q8 = (ctypes.c_uint8 * q8_bytes)()
            lib.quantize_row_q8_k(
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(q8, ctypes.c_void_p),
                int(args.cols),
            )
            kernel(
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_void_p(base + weight_offset),
                ctypes.cast(q8, ctypes.c_void_p),
                int(args.rows),
                int(args.cols),
            )
        else:
            kernel(
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_void_p(base + weight_offset),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(args.rows),
                int(args.cols),
            )

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            y.tofile(args.output)

        print(
            f"kernel={args.kernel} weight_symbol={args.weight_symbol} weight_offset={weight_offset} "
            f"rows={args.rows} cols={args.cols} output_norm={float(np.linalg.norm(y)):.9f}"
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
