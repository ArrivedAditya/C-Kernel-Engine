#!/usr/bin/env python3
"""Compare PyTorch, exact oneDNN, and CK BF16 linear arithmetic."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


EXPECTED_ONEDNN_VERSION = "3.7.1"
EXPECTED_ONEDNN_HASH = "8d263e693366ef8db40acc569cc7d8edf644556d"
CK_PROVIDER = "gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage"


def paths(workdir: Path) -> dict[str, Path]:
    return {name: workdir / f"{name}.bf16" for name in (
        "src", "weight", "bias", "pytorch", "onednn", "ck"
    )}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16_to_fp32(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32) << 16).view(np.float32)


def run_pytorch(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import torch.nn.functional as functional

    if not torch.backends.mkldnn.is_available():
        raise RuntimeError("PyTorch oneDNN backend is unavailable")
    torch.set_num_threads(args.threads)
    files = paths(args.workdir)
    rng = np.random.default_rng(args.seed)
    src = torch.from_numpy(rng.standard_normal((args.m, args.k), dtype=np.float32)).to(torch.bfloat16)
    weight = torch.from_numpy(rng.standard_normal((args.n, args.k), dtype=np.float32)).to(torch.bfloat16)
    bias = torch.from_numpy(rng.standard_normal((args.n,), dtype=np.float32)).to(torch.bfloat16)
    src.view(torch.uint16).numpy().tofile(files["src"])
    weight.view(torch.uint16).numpy().tofile(files["weight"])
    bias.view(torch.uint16).numpy().tofile(files["bias"])
    started = time.perf_counter()
    output = functional.linear(src, weight, bias)
    elapsed = time.perf_counter() - started
    if output.dtype != torch.bfloat16:
        raise RuntimeError(f"PyTorch linear returned {output.dtype}, expected torch.bfloat16")
    output.view(torch.uint16).numpy().tofile(files["pytorch"])
    return {
        "backend": "pytorch",
        "seconds": elapsed,
        "sha256": sha256(files["pytorch"]),
        "torch_version": torch.__version__,
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "output_dtype": str(output.dtype),
        "threads": torch.get_num_threads(),
        "input_sha256": {
            "src": sha256(files["src"]),
            "weight": sha256(files["weight"]),
            "bias": sha256(files["bias"]),
        },
    }


class DnnlVersion(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("patch", ctypes.c_int),
        ("hash", ctypes.c_char_p),
        ("cpu_runtime", ctypes.c_uint),
        ("gpu_runtime", ctypes.c_uint),
    ]


def load_onednn_version(library: ctypes.CDLL) -> dict[str, object]:
    function = library.dnnl_version
    function.argtypes = []
    function.restype = ctypes.POINTER(DnnlVersion)
    version = function().contents
    source_hash = version.hash.decode() if version.hash else None
    version_text = f"{version.major}.{version.minor}.{version.patch}"
    if version_text != EXPECTED_ONEDNN_VERSION or source_hash != EXPECTED_ONEDNN_HASH:
        raise RuntimeError(
            "oneDNN oracle identity mismatch: "
            f"expected {EXPECTED_ONEDNN_VERSION} {EXPECTED_ONEDNN_HASH}, "
            f"found {version_text} {source_hash}"
        )
    return {
        "version": version_text,
        "source_hash": source_hash,
        "cpu_runtime": int(version.cpu_runtime),
    }


def run_onednn(args: argparse.Namespace) -> dict[str, object]:
    files = paths(args.workdir)
    src = np.memmap(files["src"], dtype=np.uint16, mode="r", shape=(args.m, args.k))
    weight = np.memmap(files["weight"], dtype=np.uint16, mode="r", shape=(args.n, args.k))
    bias = np.memmap(files["bias"], dtype=np.uint16, mode="r", shape=(args.n,))
    output = np.memmap(files["onednn"], dtype=np.uint16, mode="w+", shape=(args.m, args.n))
    library = ctypes.CDLL(str(args.library.resolve()))
    version = load_onednn_version(library)
    function = library.onednn_linear_bf16
    pointer = ctypes.POINTER(ctypes.c_uint16)
    function.argtypes = [pointer, pointer, pointer, pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    function.restype = ctypes.c_int
    started = time.perf_counter()
    status = function(
        src.ctypes.data_as(pointer), weight.ctypes.data_as(pointer),
        bias.ctypes.data_as(pointer), output.ctypes.data_as(pointer),
        args.m, args.n, args.k,
    )
    elapsed = time.perf_counter() - started
    output.flush()
    if status != 0:
        raise RuntimeError(f"oneDNN probe failed with status {status}")
    return {
        "backend": "onednn",
        "seconds": elapsed,
        "sha256": sha256(files["onednn"]),
        "threads": args.threads,
        **version,
    }


def run_ck(args: argparse.Namespace) -> dict[str, object]:
    files = paths(args.workdir)
    src_bits = np.memmap(files["src"], dtype=np.uint16, mode="r", shape=(args.m, args.k))
    weight = np.memmap(files["weight"], dtype=np.uint16, mode="r", shape=(args.n, args.k))
    bias_bits = np.memmap(files["bias"], dtype=np.uint16, mode="r", shape=(args.n,))
    src = bf16_to_fp32(np.asarray(src_bits)).copy()
    bias = bf16_to_fp32(np.asarray(bias_bits)).copy()
    output = np.empty((args.m, args.n), dtype=np.float32)
    library = ctypes.CDLL(str(args.ck_library.resolve()))
    function = getattr(library, CK_PROVIDER)
    float_pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [
        float_pointer, ctypes.c_void_p, float_pointer, float_pointer,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    function.restype = None
    started = time.perf_counter()
    function(
        src.ctypes.data_as(float_pointer), ctypes.c_void_p(weight.ctypes.data),
        bias.ctypes.data_as(float_pointer), output.ctypes.data_as(float_pointer),
        args.m, args.n, args.k,
    )
    elapsed = time.perf_counter() - started
    (output.view(np.uint32) >> 16).astype(np.uint16).tofile(files["ck"])
    return {
        "backend": "ck",
        "provider": CK_PROVIDER,
        "seconds": elapsed,
        "sha256": sha256(files["ck"]),
        "threads": args.threads,
    }


def compare(args: argparse.Namespace, backend: str) -> dict[str, object]:
    files = paths(args.workdir)
    expected = np.memmap(files["pytorch"], dtype=np.uint16, mode="r", shape=(args.m, args.n))
    actual = np.memmap(files[backend], dtype=np.uint16, mode="r", shape=(args.m, args.n))
    mismatch_count = int(np.count_nonzero(expected != actual))
    delta = bf16_to_fp32(np.asarray(actual)) - bf16_to_fp32(np.asarray(expected))
    return {
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "threads": args.threads,
        "mismatch_count": mismatch_count,
        "element_count": int(expected.size),
        "exact_ratio": float(1.0 - mismatch_count / expected.size),
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
        "pytorch_sha256": sha256(files["pytorch"]),
        f"{backend}_sha256": sha256(files[backend]),
    }


def run_all(args: argparse.Namespace) -> None:
    args.workdir.mkdir(parents=True, exist_ok=True)
    pytorch = run_pytorch(args)
    onednn = run_onednn(args)
    ck = run_ck(args)
    onednn_comparison = compare(args, "onednn")
    ck_comparison = compare(args, "ck")
    result = {
        "status": "pass" if not onednn_comparison["mismatch_count"] and not ck_comparison["mismatch_count"] else "fail",
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "threads": args.threads,
        "seed": args.seed,
        "provider": CK_PROVIDER,
        "contract": "bf16_weight_bf16_input_pytorch_onednn_brgemm_bf16_output",
        "provenance": {
            "torch_version": pytorch["torch_version"],
            "torch_mkldnn_available": pytorch["mkldnn_available"],
            "torch_output_dtype": pytorch["output_dtype"],
            "onednn_version": onednn["version"],
            "onednn_source_hash": onednn["source_hash"],
            "onednn_cpu_runtime": onednn["cpu_runtime"],
            "input_sha256": pytorch["input_sha256"],
        },
        "timings_sec": {
            "pytorch": pytorch["seconds"],
            "onednn": onednn["seconds"],
            "ck": ck["seconds"],
        },
        "pytorch_vs_onednn": onednn_comparison,
        "pytorch_vs_ck": ck_comparison,
    }
    (args.workdir / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--ck-library", type=Path, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    env_threads = int(os.environ.get("OMP_NUM_THREADS", args.threads))
    if env_threads != args.threads:
        raise SystemExit(f"OMP_NUM_THREADS={env_threads} does not match --threads={args.threads}")
    run_all(args)


if __name__ == "__main__":
    main()
