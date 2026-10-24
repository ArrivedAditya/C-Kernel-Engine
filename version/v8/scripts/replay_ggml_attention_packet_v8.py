#!/usr/bin/env python3
"""Replay retained attention operands through pinned GGML attention graphs."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

if __package__:
    from .replay_attention_from_dumps_v8 import (
        _load_resolved_bundle, _metrics, _sha256, replay_resolved_segmented_prefill,
    )
    from .xray_operation_bundle_v8 import BundleError, load_bundle
else:
    from replay_attention_from_dumps_v8 import (
        _load_resolved_bundle, _metrics, _sha256, replay_resolved_segmented_prefill,
    )
    from xray_operation_bundle_v8 import BundleError, load_bundle


SOURCE = Path(__file__).with_name("ggml_nonflash_attention_packet_v8.c")
FLASH_SOURCE = Path(__file__).with_name("ggml_flash_attention_rows_v8.c")
REPO = Path(__file__).resolve().parents[3]


def validate_oracle_contract(manifest: dict, call: dict, segments: list[int]) -> None:
    context = manifest["context"]
    tokens = sum(segments)
    if context.get("oracle_mode") != "disabled":
        raise BundleError("GGML replay requires the retained non-flash oracle mode")
    if call["sliding_window"] < tokens:
        raise BundleError("this adapter cannot replay a wrapped sliding attention window")
    if context.get("valid_cache_positions") != [0, tokens]:
        raise BundleError("invalid post-store cache position range")
    if call["head_dim"] != call["aligned_head_dim"]:
        raise BundleError("GGML replay does not support padded head dimensions")
    if call["selector"] not in (
        "CK_ATTN_REDUCTION_F16_REGULAR_NOSCALE",
        "CK_ATTN_REDUCTION_F16_FLASH_AUTO_QTILE64_REGULAR_TAIL_NOSCALE",
    ):
        raise BundleError("GGML replay does not support this attention scaling contract")


def pinned_tree_identity(tree: Path) -> dict[str, object]:
    expected = subprocess.check_output(
        ["git", "ls-tree", "HEAD", "llama.cpp"], cwd=REPO, text=True,
    ).split()[2]
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tree, text=True,
    ).strip()
    if expected != actual:
        raise BundleError(f"llama.cpp checkout {actual} does not match CKE gitlink {expected}")
    lib = tree / "build" / "bin" / "libggml-cpu.so"
    base = tree / "build" / "bin" / "libggml-base.so"
    if not lib.is_file() or not base.is_file():
        raise BundleError("pinned llama.cpp GGML libraries are not built")
    return {"source_revision": actual, "ggml_cpu_sha256": _sha256(lib),
            "ggml_base_sha256": _sha256(base)}


def _compile_ggml_source(tree: Path, source: Path, executable: Path) -> dict:
    build_bin = tree / "build" / "bin"
    compile_cmd = [
        "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        "-I", str(tree / "ggml" / "include"), str(source),
        "-L", str(build_bin), "-Wl,-rpath," + str(build_bin),
        "-lggml-cpu", "-lggml-base", "-lggml", "-lm", "-o", str(executable),
    ]
    subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
    linked = subprocess.check_output(["ldd", str(executable)], text=True)
    resolved = {}
    for line in linked.splitlines():
        fields = line.strip().split("=>", 1)
        if len(fields) != 2 or not fields[0].startswith("libggml"):
            continue
        path = Path(fields[1].strip().split(" ", 1)[0]).resolve()
        if path.parent != build_bin.resolve() or not path.is_file():
            raise BundleError(f"GGML replay linked an unexpected library: {line.strip()}")
        resolved[fields[0].strip()] = {"path": str(path), "sha256": _sha256(path)}
    if not any(name.startswith("libggml-cpu") for name in resolved) or not any(
        name.startswith("libggml-base") for name in resolved
    ):
        raise BundleError("GGML replay did not link the expected CPU/base libraries")
    return resolved


def run_ggml(tree: Path, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, dict]:
    tokens, heads, dim = q.shape
    kv_heads = k.shape[1]
    identity = pinned_tree_identity(tree)
    build_bin = tree / "build" / "bin"
    with tempfile.TemporaryDirectory(prefix="cke-ggml-attn-replay-") as temp:
        directory = Path(temp)
        executable = directory / "replay"
        resolved = _compile_ggml_source(tree, SOURCE, executable)
        paths = [directory / name for name in ("q.f32", "k.f16", "v.f16", "out.f32")]
        np.asarray(q, dtype="<f4").tofile(paths[0])
        np.asarray(k, dtype="<f2").tofile(paths[1])
        np.asarray(v, dtype="<f2").tofile(paths[2])
        env = dict(os.environ, LD_LIBRARY_PATH=str(build_bin))
        subprocess.run(
            [str(executable), *(str(path) for path in paths),
             str(tokens), str(heads), str(kv_heads), str(dim)],
            env=env, check=True, capture_output=True, text=True,
        )
        output = np.fromfile(paths[3], dtype="<f4")
        if output.size != q.size or not np.isfinite(output).all():
            raise BundleError("GGML replay produced incomplete or nonfinite output")
        identity["replay_binary_sha256"] = _sha256(executable)
        identity["replay_source_sha256"] = _sha256(SOURCE)
        identity["resolved_libraries"] = resolved
        identity["library_build_revision_verified"] = False
        return output.reshape(q.shape), identity


def run_ggml_flash_rows(
    tree: Path, q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
    scale: float, threads: int = 1, allow_subset_queries: bool = False,
) -> tuple[np.ndarray, dict]:
    """Replay full, unmasked flash self-attention; subsets are diagnostic only."""
    q, k, v = np.asarray(q), np.asarray(k), np.asarray(v)
    if q.dtype != np.float32 or k.dtype != np.float16 or v.dtype != np.float16:
        raise BundleError("flash replay requires observed FP32 Q and post-store FP16 K/V")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise BundleError("flash replay requires Q[queries,heads,dim] and matching K/V[keys,kv_heads,dim]")
    queries, heads, dim = q.shape
    keys, kv_heads, key_dim = k.shape
    if not (1 <= queries <= 32768 and 1 <= keys <= 32768 and 1 <= heads <= 128 and
            1 <= kv_heads <= heads and heads % kv_heads == 0 and
            1 <= dim <= 256 and key_dim == dim and 1 <= threads <= 256):
        raise BundleError("unsupported full-flash attention geometry")
    if queries != keys and not allow_subset_queries:
        raise BundleError("full-query geometry is required for connected flash replay fidelity")
    if not math.isfinite(scale) or scale <= 0 or not all(np.isfinite(x).all() for x in (q, k, v)):
        raise BundleError("flash replay requires finite operands and positive finite scale")
    if q.nbytes + k.nbytes + v.nbytes > 256 * 1024 * 1024:
        raise BundleError("flash replay operands exceed capacity")

    identity = pinned_tree_identity(tree)
    build_bin = tree / "build" / "bin"
    with tempfile.TemporaryDirectory(prefix="cke-ggml-flash-replay-") as temp:
        directory = Path(temp)
        executable = directory / "replay"
        resolved = _compile_ggml_source(tree, FLASH_SOURCE, executable)
        paths = [directory / name for name in ("q.f32", "k.f16", "v.f16", "out.f32")]
        np.asarray(q, dtype="<f4").tofile(paths[0])
        np.asarray(k, dtype="<f2").tofile(paths[1])
        np.asarray(v, dtype="<f2").tofile(paths[2])
        env = dict(os.environ, LD_LIBRARY_PATH=str(build_bin))
        subprocess.run(
            [str(executable), *(str(path) for path in paths),
             str(queries), str(keys), str(heads), str(kv_heads), str(dim),
             repr(float(np.float32(scale))), str(threads)],
            env=env, check=True, capture_output=True, text=True, timeout=300,
        )
        output = np.fromfile(paths[3], dtype="<f4")
        if output.size != q.size or not np.isfinite(output).all():
            raise BundleError("GGML flash replay produced incomplete or nonfinite output")
        identity["replay_binary_sha256"] = _sha256(executable)
        identity["replay_source_sha256"] = _sha256(FLASH_SOURCE)
        identity["resolved_libraries"] = resolved
        identity["library_build_revision_verified"] = False
        identity["contract"] = {
            "mode": "full_noncausal_flash", "q_dtype": "fp32", "kv_dtype": "fp16",
            "accumulation": "ggml_prec_f32", "mask": "none", "scale": float(np.float32(scale)),
            "queries": queries, "keys": keys, "heads": heads, "kv_heads": kv_heads,
            "head_dim": dim, "threads": threads,
            "query_extent": "complete" if queries == keys else "subset_not_connected_fidelity",
        }
        return output.reshape(q.shape), identity


def fp64_full_flash_reference_rows(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, rows: list[int], *, scale: float,
) -> np.ndarray:
    """Evaluate selected noncausal rows with FP16-rounded K/V and FP64 arithmetic."""
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float16)
    v = np.asarray(v, dtype=np.float16)
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise BundleError("FP64 flash reference requires rank-three Q and matching K/V")
    queries, heads, dim = q.shape
    keys, kv_heads, key_dim = k.shape
    if (not queries or not keys or not heads or not kv_heads or key_dim != dim or
            heads % kv_heads or not math.isfinite(scale) or scale <= 0 or
            not all(np.isfinite(x).all() for x in (q, k, v))):
        raise BundleError("invalid FP64 flash reference contract")
    if not rows or len(set(rows)) != len(rows) or any(row < 0 or row >= queries for row in rows):
        raise BundleError("FP64 flash reference requires distinct in-range query rows")
    result = np.empty((len(rows), heads, dim), dtype=np.float64)
    k64 = k.astype(np.float64)
    v64 = v.astype(np.float64)
    for index, row in enumerate(rows):
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            scores = (k64[:, kv_head] @ q[row, head].astype(np.float64)) * scale
            weights = np.exp(scores - scores.max())
            result[index, head] = (weights / weights.sum()) @ v64[:, kv_head]
    return result


def fp64_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    tokens, heads, dim = q.shape
    kv_heads = k.shape[1]
    result = np.empty_like(q)
    for t in range(tokens):
        for h in range(heads):
            kh = h // (heads // kv_heads)
            scores = k[:t + 1, kh].astype(np.float64) @ q[t, h].astype(np.float64)
            weights = np.exp(scores - np.max(scores))
            weights /= np.sum(weights)
            result[t, h] = weights @ v[:t + 1, kh].astype(np.float64)
    return result


def compare_packet(bundle: Path, tree: Path) -> dict:
    dumps, layer, call, segments, engine = _load_resolved_bundle(bundle)
    manifest, tensors = load_bundle(bundle)
    validate_oracle_contract(manifest, call, segments)
    fidelity = replay_resolved_segmented_prefill(dumps, layer, call, segments, engine,
                                                  independent_fp64=True)
    if fidelity["status"] != "pass":
        return {"status": "fail_replay_fidelity", "cke_replay": fidelity,
                "oracle_graph": "not_run"}
    q = tensors["Qcur_rope"]
    k = tensors["Kcache_post_store"]
    v = tensors["Vcache_post_store"]
    oracle, identity = run_ggml(tree, q, k, v)
    reference = fp64_reference(q, k, v)
    cke = tensors["kqv_out"]
    worst = np.unravel_index(np.argmax(np.abs(cke - oracle)), cke.shape)
    return {
        "status": "incomplete_oracle_runtime_fidelity",
        "interpretation": "Compact valid-key GGML diagnostic only; the executed llama.cpp graph may use a larger masked physical cache and cannot be inferred from this result",
        "first_proven_local_divergence": None,
        "missing_oracle_capability": "oracle post-RoPE Q, complete post-store K/V and resolved mask for replay fidelity; CKE and oracle score/softmax captures for first-stage attribution",
        "bundle_sha256": _sha256(bundle),
        "cke_replay": {"status": fidelity["status"], "bitwise_equal": fidelity["bitwise_equal"],
                       "cache_reconstruction": fidelity["cache_reconstruction"]},
        "oracle_graph": identity,
        "contract": {"mode": "non_flash", "effective_mask_assumption": "causal prefix, valid keys 0..query_position; oracle mask not captured",
                     "scale": 1.0, "cache_dtype": "fp16", "heads": q.shape[1],
                     "kv_heads": k.shape[1], "head_dim": q.shape[2], "segments": segments,
                     "key_extent": "compact valid keys; not the oracle's physical cache extent"},
        "cke_vs_ggml": _metrics(cke, oracle),
        "cke_vs_fp64": _metrics(cke, reference),
        "ggml_vs_fp64": _metrics(oracle, reference),
        "worst_cke_ggml_index": list(map(int, worst)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--llama-tree", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = compare_packet(args.bundle, args.llama_tree)
    except (BundleError, ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        report = {"status": "incomplete", "error": str(exc)}
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
