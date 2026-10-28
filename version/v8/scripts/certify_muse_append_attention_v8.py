#!/usr/bin/env python3
"""Compare Muse's existing full attention with prior-cache chunked attention.

This is an opt-in kernel arithmetic check. The caller supplies a library built
from the candidate source and the already-certified baseline engine library.
It does not certify the complete model or a Muse Code task.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np


F32 = ctypes.POINTER(ctypes.c_float)
U16 = ctypes.POINTER(ctypes.c_uint16)
WORKSPACE_TYPES = [F32, ctypes.c_size_t, U16, ctypes.c_size_t,
                   U16, ctypes.c_size_t, U16, ctypes.c_size_t,
                   U16, ctypes.c_size_t]
HEADS, KV_HEADS, HEAD_DIM = 2, 1, 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace(rows: int, keys: int):
    arrays = [
        np.zeros((rows, keys), dtype=np.float32),
        np.zeros((keys, HEAD_DIM), dtype=np.uint16),
        np.zeros((HEAD_DIM, keys), dtype=np.uint16),
        np.zeros((rows, HEAD_DIM), dtype=np.uint16),
        np.zeros((rows, keys), dtype=np.uint16),
    ]
    return [item for array in arrays
            for item in (array.ctypes.data_as(F32 if array.dtype == np.float32 else U16),
                         array.nbytes)]


def _compare_case(engine, candidate, *, tokens: int, split: int, window: int):
    rng = np.random.default_rng(7281 + tokens + window)
    q = rng.normal(size=(HEADS, tokens, HEAD_DIM)).astype(np.float32)
    k = rng.normal(size=(KV_HEADS, tokens, HEAD_DIM)).astype(np.float32)
    v = rng.normal(size=(KV_HEADS, tokens, HEAD_DIM)).astype(np.float32)
    capacity = tokens + 3
    cache_k = np.zeros((KV_HEADS, capacity, HEAD_DIM), dtype=np.float32)
    cache_v = np.zeros_like(cache_k)

    full_name = "attention_forward_causal_head_major_gqa_muse_eager_bf16_storage"
    if window:
        full_name += "_sliding"
    full_fn = getattr(engine, full_name)
    full_fn.argtypes = [F32] * 4 + [ctypes.c_int] * (7 if window else 6) + WORKSPACE_TYPES
    full_fn.restype = None
    full = np.empty_like(q)
    full_args = [q.ctypes.data_as(F32), k.ctypes.data_as(F32),
                 v.ctypes.data_as(F32), full.ctypes.data_as(F32),
                 HEADS, KV_HEADS, tokens, HEAD_DIM, HEAD_DIM, tokens]
    if window:
        full_args.append(window)
    full_fn(*full_args, *_workspace(tokens, tokens))

    append = candidate.attention_forward_causal_head_major_gqa_muse_eager_bf16_storage_append
    append.argtypes = [F32] * 6 + [ctypes.c_int] * 8 + WORKSPACE_TYPES
    append.restype = None

    def run(begin: int, end: int, past: int):
        rows = end - begin
        out = np.empty((HEADS, rows, HEAD_DIM), dtype=np.float32)
        qi = np.ascontiguousarray(q[:, begin:end])
        ki = np.ascontiguousarray(k[:, begin:end])
        vi = np.ascontiguousarray(v[:, begin:end])
        append(qi.ctypes.data_as(F32), ki.ctypes.data_as(F32),
               vi.ctypes.data_as(F32), cache_k.ctypes.data_as(F32),
               cache_v.ctypes.data_as(F32), out.ctypes.data_as(F32),
               HEADS, KV_HEADS, rows, past, capacity, HEAD_DIM, HEAD_DIM,
               window, *_workspace(rows, past + rows))
        return out

    first = run(0, split, 0)
    cache_k[:, :split] = k[:, :split]
    cache_v[:, :split] = v[:, :split]
    second = run(split, tokens, split)
    joined = np.concatenate((first, second), axis=1)
    error = float(np.max(np.abs(full - joined)))
    wrong_position = run(split, tokens, split + 1)
    position_delta = float(np.max(np.abs(second - wrong_position)))
    cache_k[:, split - 1] += 4.0
    corrupted = run(split, tokens, split)
    fault_delta = float(np.max(np.abs(second - corrupted)))
    return {
        "tokens": tokens, "split": split, "sliding_window": window,
        "max_abs_error": error, "corrupted_cache_max_delta": fault_delta,
        "wrong_position_max_delta": position_delta,
        "finite": bool(np.isfinite(full).all() and np.isfinite(joined).all()),
        "status": "PASS" if error == 0.0 and fault_delta > 0.0
                  and position_delta > 0.0
                  and np.isfinite(full).all() and np.isfinite(joined).all() else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--append-library", type=Path, required=True)
    parser.add_argument("--sleef-library", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build-flags", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for path in (args.engine, args.append_library, args.sleef_library, args.source):
        if not path.is_file():
            parser.error(f"missing library: {path}")
    os.environ["CK_SLEEF_LIBRARY"] = str(args.sleef_library.resolve())
    engine = ctypes.CDLL(str(args.engine.resolve()), mode=os.RTLD_GLOBAL)
    candidate = ctypes.CDLL(str(args.append_library.resolve()), mode=os.RTLD_GLOBAL)
    cases = [
        _compare_case(engine, candidate, tokens=n, split=split, window=window)
        for n, split in ((4, 2), (10, 6), (129, 127), (257, 256))
        for window in (0, 3)
    ]
    report = {
        "schema": "cke.v8.muse_append_attention_kernel_parity.v1",
        "status": "PASS" if all(case["status"] == "PASS" for case in cases) else "FAIL",
        "engine_sha256": _sha256(args.engine),
        "append_library_sha256": _sha256(args.append_library),
        "sleef_library_sha256": _sha256(args.sleef_library),
        "source_sha256": _sha256(args.source),
        "build_flags": args.build_flags,
        "cpu_model": next((line.split(":", 1)[1].strip()
                           for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), "unknown"),
        "cases": cases,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
