#!/usr/bin/env python3
"""Compare a verified CKE attention packet with the pinned llama.cpp non-flash GGML graph."""

from __future__ import annotations

import argparse
import json
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


def run_ggml(tree: Path, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, dict]:
    tokens, heads, dim = q.shape
    kv_heads = k.shape[1]
    identity = pinned_tree_identity(tree)
    build_bin = tree / "build" / "bin"
    with tempfile.TemporaryDirectory(prefix="cke-ggml-attn-replay-") as temp:
        directory = Path(temp)
        executable = directory / "replay"
        compile_cmd = [
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-I", str(tree / "ggml" / "include"), str(SOURCE),
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
        "interpretation": "Identical-input pinned GGML non-flash graph comparison; actual llama.cpp runtime replay is unverified because complete oracle post-store KV was not captured",
        "first_proven_local_divergence": None,
        "missing_oracle_capability": "oracle post-RoPE Q, complete post-store K/V and resolved mask for replay fidelity; CKE and oracle score/softmax captures for first-stage attribution",
        "bundle_sha256": _sha256(bundle),
        "cke_replay": {"status": fidelity["status"], "bitwise_equal": fidelity["bitwise_equal"],
                       "cache_reconstruction": fidelity["cache_reconstruction"]},
        "oracle_graph": identity,
        "contract": {"mode": "non_flash", "effective_mask_assumption": "causal prefix, valid keys 0..query_position; oracle mask not captured",
                     "scale": 1.0, "cache_dtype": "fp16", "heads": q.shape[1],
                     "kv_heads": k.shape[1], "head_dim": q.shape[2], "segments": segments},
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
