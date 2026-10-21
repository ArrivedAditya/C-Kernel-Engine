#!/usr/bin/env python3
"""Replay a selected Q4_K/Q8_K prefill projection on captured FP32 rows.

This is an exact-input diagnostic, not an alternative model executor. The
generated model supplies the converted weights; the engine supplies the same
quantizer and segmented projection provider used by generated prefill.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
from pathlib import Path

import numpy as np

QK_K = 256
Q8_K_ROW_BYTES = 292
Q4_K_BLOCK_BYTES = 144


def _offset(source: Path, symbol: str) -> int:
    if not re.fullmatch(r"W_LAYER_[0-9]+_[A-Z0-9_]+", symbol):
        raise ValueError("weight symbol must name a generated layer weight")
    matches = re.findall(rf"^#define {re.escape(symbol)} ([0-9]+)$", source.read_text(), re.M)
    if len(matches) != 1:
        raise ValueError(f"expected one {symbol} offset in {source}")
    return int(matches[0])


def _rows(path: Path, rows: int, cols: int, start: int = 0) -> np.ndarray:
    row_bytes = cols * 4
    size = path.stat().st_size
    if start < 0 or size % row_bytes or start + rows > size // row_bytes:
        raise ValueError(f"{path} cannot provide [{rows}, {cols}] FP32 at row {start}")
    result = np.fromfile(path, dtype="<f4", count=rows * cols,
                         offset=start * row_bytes).reshape(rows, cols)
    if not np.isfinite(result).all():
        raise ValueError(f"{path} contains nonfinite values")
    return np.ascontiguousarray(result)


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("comparison requires equal finite tensors")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    worst = np.unravel_index(int(np.abs(delta).argmax()), delta.shape)
    return {
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "worst_row": int(worst[0]),
        "worst_col": int(worst[1]),
        "actual_at_worst": float(actual[worst]),
        "expected_at_worst": float(expected[worst]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-lib", type=Path, required=True)
    parser.add_argument("--tokenizer-lib", type=Path, required=True)
    parser.add_argument("--decoder-lib", type=Path, required=True)
    parser.add_argument("--generated-c", type=Path, required=True)
    parser.add_argument("--weights-bump", type=Path, required=True)
    parser.add_argument("--weight-symbol", required=True)
    parser.add_argument("--input-f32", type=Path, required=True)
    parser.add_argument("--expected-f32", type=Path, required=True)
    parser.add_argument("--input-start-row", type=int, default=0)
    parser.add_argument("--expected-start-row", type=int, default=0)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--output-cols", type=int, required=True)
    parser.add_argument("--segments", required=True, help="comma-separated row counts")
    parser.add_argument("--output-f32", type=Path)
    parser.add_argument("--post-norm-symbol", help="generated RMSNorm gamma offset symbol")
    parser.add_argument("--post-norm-expected-f32", type=Path)
    parser.add_argument("--post-norm-expected-start-row", type=int, default=0)
    parser.add_argument("--post-norm-eps", type=float)
    parser.add_argument("--post-norm-output-f32", type=Path)
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0 or args.output_cols <= 0 or args.cols % QK_K:
        parser.error("dimensions must be positive and input columns divisible by 256")
    try:
        segments = [int(value) for value in args.segments.split(",")]
    except ValueError as exc:
        parser.error(f"invalid segment plan: {exc}")
    if not segments or any(value <= 0 for value in segments) or sum(segments) != args.rows:
        parser.error("segments must be positive and sum to rows")
    post_norm_requested = any((args.post_norm_symbol, args.post_norm_expected_f32,
                               args.post_norm_eps, args.post_norm_output_f32))
    if post_norm_requested and (not args.post_norm_symbol or not args.post_norm_expected_f32
                                or args.post_norm_eps is None or not np.isfinite(args.post_norm_eps)
                                or args.post_norm_eps <= 0):
        parser.error("post-normalization replay requires gamma, expected output, and positive epsilon")

    source = args.generated_c.resolve()
    weights = args.weights_bump.resolve()
    offset = _offset(source, args.weight_symbol)
    matrix_bytes = args.output_cols * (args.cols // QK_K) * Q4_K_BLOCK_BYTES
    if offset + matrix_bytes > weights.stat().st_size:
        raise ValueError("weight offset is outside the converted bundle")
    x = _rows(args.input_f32, args.rows, args.cols, args.input_start_row)
    expected = _rows(args.expected_f32, args.rows, args.output_cols,
                     args.expected_start_row)
    post_expected = (
        _rows(args.post_norm_expected_f32, args.rows, args.output_cols,
              args.post_norm_expected_start_row)
        if post_norm_requested else None
    )
    norm_offset = _offset(source, args.post_norm_symbol) if post_norm_requested else None
    if norm_offset is not None and norm_offset + args.output_cols * 4 > weights.stat().st_size:
        raise ValueError("normalization weight offset is outside the converted bundle")

    ctypes.CDLL(str(args.tokenizer_lib.resolve()), mode=ctypes.RTLD_GLOBAL)
    engine = ctypes.CDLL(str(args.engine_lib.resolve()), mode=ctypes.RTLD_GLOBAL)
    model = ctypes.CDLL(str(args.decoder_lib.resolve()), mode=ctypes.RTLD_GLOBAL)
    model.ck_model_init.argtypes = [ctypes.c_char_p]
    model.ck_model_init.restype = ctypes.c_int
    model.ck_model_get_base_ptr.argtypes = []
    model.ck_model_get_base_ptr.restype = ctypes.c_uint64
    model.ck_model_free.argtypes = []
    model.ck_model_free.restype = None
    rc = model.ck_model_init(str(weights).encode())
    if rc != 0:
        raise RuntimeError(f"generated model initialization failed: rc={rc}")
    try:
        base = int(model.ck_model_get_base_ptr())
        if not base:
            raise RuntimeError("generated model returned null weight base")
        q8_row_bytes = args.cols // QK_K * Q8_K_ROW_BYTES
        q8 = np.empty((args.rows, q8_row_bytes), dtype=np.uint8)
        quantize = engine.quantize_row_q8_k
        quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        quantize.restype = None
        for row in range(args.rows):
            quantize(ctypes.c_void_p(x[row].ctypes.data),
                     ctypes.c_void_p(q8[row].ctypes.data), args.cols)
        result = np.empty((args.rows, args.output_cols), dtype=np.float32)
        segment_array = (ctypes.c_int * len(segments))(*segments)
        gemm = engine.gemm_nt_q4_k_q8_k_segmented_singleton_tail_parallel_dispatch
        gemm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                         ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        gemm.restype = None
        gemm(ctypes.c_void_p(q8.ctypes.data), ctypes.c_void_p(base + offset), None,
             ctypes.c_void_p(result.ctypes.data), args.rows, args.output_cols,
             args.cols, segment_array, len(segments))
        post_result = None
        if post_norm_requested:
            post_result = np.empty_like(result)
            norm = engine.rmsnorm_forward_llama_production_parallel_dispatch
            norm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_float]
            norm.restype = None
            norm(ctypes.c_void_p(result.ctypes.data), ctypes.c_void_p(base + norm_offset),
                 ctypes.c_void_p(post_result.ctypes.data), None, args.rows,
                 args.output_cols, args.output_cols, args.post_norm_eps)
    finally:
        model.ck_model_free()

    if args.output_f32:
        args.output_f32.parent.mkdir(parents=True, exist_ok=True)
        result.tofile(args.output_f32)
    if args.post_norm_output_f32:
        args.post_norm_output_f32.parent.mkdir(parents=True, exist_ok=True)
        post_result.tofile(args.post_norm_output_f32)
    report = {
        "status": "diagnostic_only",
        "provider": "gemm_nt_q4_k_q8_k_segmented_singleton_tail_parallel_dispatch",
        "weight_symbol": args.weight_symbol,
        "weight_offset": offset,
        "shape": [args.rows, args.cols, args.output_cols],
        "segments": segments,
        "input_start_row": args.input_start_row,
        "expected_start_row": args.expected_start_row,
        "engine_sha256": _sha256(args.engine_lib),
        "decoder_sha256": _sha256(args.decoder_lib),
        "input_sha256": _sha256(args.input_f32),
        "expected_sha256": _sha256(args.expected_f32),
        "comparison": _metrics(result, expected),
    }
    if post_norm_requested:
        report["post_norm"] = {
            "provider": "rmsnorm_forward_llama_production_parallel_dispatch",
            "weight_symbol": args.post_norm_symbol,
            "weight_offset": norm_offset,
            "epsilon": args.post_norm_eps,
            "expected_sha256": _sha256(args.post_norm_expected_f32),
            "expected_start_row": args.post_norm_expected_start_row,
            "comparison": _metrics(post_result, post_expected),
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
