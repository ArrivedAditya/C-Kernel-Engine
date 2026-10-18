#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
BUILD_DIR = REPO_ROOT / "build"
LLAMA_CPP_BIN = REPO_ROOT / "llama.cpp" / "build" / "bin"
V7_SCRIPTS = REPO_ROOT / "version" / "v7" / "scripts"

if str(V7_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(V7_SCRIPTS))

import parity_test  # type: ignore  # noqa: E402


def _require_tensor_any(dumps: list[parity_test.ParityDump], layer: int, names: list[str]) -> np.ndarray:
    for name in names:
        for dump in dumps:
            if dump.layer_id == layer and dump.op_name == name:
                return np.asarray(dump.data, dtype=np.float32)
    raise KeyError(f"missing tensor layer={layer} names={names}")


def _infer_num_tokens(flat: np.ndarray, num_heads: int, head_dim: int) -> int:
    denom = num_heads * head_dim
    if denom <= 0 or flat.size % denom != 0:
        raise ValueError(
            f"cannot infer tokens from {flat.size} elems with heads={num_heads} head_dim={head_dim}"
        )
    return flat.size // denom


def _reshape_token_major_qhd(flat: np.ndarray, num_tokens: int, num_heads: int, head_dim: int) -> np.ndarray:
    expected = num_tokens * num_heads * head_dim
    if flat.size != expected:
        raise ValueError(f"expected {expected} elems, got {flat.size}")
    return flat.reshape(num_tokens, num_heads, head_dim)


def _to_head_major(flat: np.ndarray, num_tokens: int, num_heads: int, head_dim: int) -> np.ndarray:
    token_major = _reshape_token_major_qhd(flat, num_tokens, num_heads, head_dim)
    return np.transpose(token_major, (1, 0, 2)).copy()


def _to_token_major(flat_head_major: np.ndarray, num_tokens: int, num_heads: int, aligned_head_dim: int) -> np.ndarray:
    return np.transpose(flat_head_major.reshape(num_heads, num_tokens, aligned_head_dim), (1, 0, 2)).copy()


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a.astype(np.float64) - b.astype(np.float64)
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    cosine = float(np.dot(a64.ravel(), b64.ravel()) / denom) if denom else 1.0
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine": cosine,
    }


def _resolve_attention_fn(lib: ctypes.CDLL, mode: str) -> Callable[..., None]:
    if mode == "ggml":
        fn = lib.attention_forward_full_head_major_gqa_ggml_strided
    elif mode == "exact":
        fn = lib.attention_forward_full_head_major_gqa_exact_strided
    elif mode == "flash":
        fn = lib.attention_forward_full_head_major_gqa_flash_strided
    elif mode == "tiled_f16kv":
        fn = lib.attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided
    else:
        raise ValueError(f"unsupported mode: {mode}")

    float_p = ctypes.POINTER(ctypes.c_float)
    fn.argtypes = [
        float_p,
        float_p,
        float_p,
        float_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    fn.restype = None
    return fn


def _default_ggml_cpu_so() -> str | None:
    for name in ("libggml-cpu.so.0.9.8", "libggml-cpu.so.0", "libggml-cpu.so"):
        path = LLAMA_CPP_BIN / name
        if path.exists():
            return str(path)
    return None


_SEGMENTED_F16_SLIDING = (
    "attention_forward_causal_head_major_gqa_prefill_segmented_f16cache_sliding_gemma4_workspace"
)
_F16_REDUCTIONS = {
    "CK_ATTN_REDUCTION_F16_REGULAR_NOSCALE": 5,
    "CK_ATTN_REDUCTION_F16_FLASH_AUTO_QTILE64_REGULAR_TAIL_NOSCALE": 6,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_tensor(dumps: list[parity_test.ParityDump], layer: int, name: str) -> np.ndarray:
    found = _unique_capture(dumps, layer, name)
    result = np.asarray(found.data, dtype=np.float32)
    if not result.size or not np.isfinite(result).all():
        raise ValueError(f"layer={layer} {name} capture is empty or nonfinite")
    return result


def _unique_capture(dumps: list[parity_test.ParityDump], layer: int, name: str) -> parity_test.ParityDump:
    found = [dump for dump in dumps if dump.layer_id == layer and dump.op_name == name]
    if len(found) != 1:
        raise ValueError(f"expected one layer={layer} {name} capture, found {len(found)}")
    return found[0]


def _resolved_segmented_call(call_ir: dict, layer: int) -> dict:
    matches = [
        op for op in call_ir.get("operations", [])
        if op.get("layer") == layer
        and op.get("section") == "body"
        and op.get("op") in ("attn_sliding", "attention_sliding")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one layer={layer} sliding attention call, found {len(matches)}")
    call = matches[0]
    contract = call.get("resolved_contract") or {}
    if not contract.get("kernel_id") or contract.get("kernel_id") != (call.get("call_abi") or {}).get("kernel_id"):
        raise ValueError("resolved kernel identity does not match the generated call ABI")
    if contract.get("function") != call.get("function"):
        raise ValueError("resolved provider function does not match the generated call")
    provider = (contract.get("implementation") or {}).get("segmented_query_provider") or {}
    if provider.get("function") != _SEGMENTED_F16_SLIDING:
        raise ValueError("resolved call has no supported segmented FP16 sliding provider")
    args = {arg["name"]: arg for arg in call.get("args", [])}
    expected_args = (
        "q", "k_cache", "v_cache", "output", "num_heads", "num_kv_heads",
        "q_tokens", "past_tokens", "cache_capacity", "head_dim", "aligned_head_dim",
        "sliding_window", "reduction", "token_workspace", "token_workspace_bytes",
    )
    if tuple(args) != expected_args or tuple(provider.get("base_args", ())) != expected_args:
        raise ValueError("resolved provider ABI does not match the replayed call")
    selector = contract.get("selector")
    if selector not in _F16_REDUCTIONS or args["reduction"].get("expr") != selector:
        raise ValueError("unsupported or contradictory resolved reduction selector")
    required = call.get("required_contract") or {}
    if required.get("tensor.kv.dtype") != "fp16" or required.get("tensor.layout") != "head_major":
        raise ValueError("resolved attention storage/layout contract is not FP16 head-major")
    def dimension(name: str) -> int:
        raw = args[name].get("expr")
        if not isinstance(raw, str) or not raw.isdecimal() or int(raw) <= 0:
            raise ValueError(f"missing positive resolved {name}")
        return int(raw)
    return {
        "function": provider["function"],
        "selector": selector,
        "reduction": _F16_REDUCTIONS[selector],
        "kernel_id": contract.get("kernel_id"),
        "num_heads": dimension("num_heads"),
        "num_kv_heads": dimension("num_kv_heads"),
        "head_dim": dimension("head_dim"),
        "aligned_head_dim": dimension("aligned_head_dim"),
        "cache_capacity": dimension("cache_capacity"),
        "sliding_window": dimension("sliding_window"),
        "workspace_bytes": dimension("token_workspace_bytes"),
    }


def _segments_from_parity_report(report: dict, dump_path: Path) -> list[int]:
    compare = report.get("dump_compare") or {}
    recorded_dump = compare.get("ck_dump_path")
    if not isinstance(recorded_dump, str) or Path(recorded_dump).resolve() != dump_path.resolve():
        raise ValueError("parity report does not identify the selected CKE dump")
    entries = compare.get("prefill_segments")
    if not isinstance(entries, list) or not entries:
        raise ValueError("parity report has no prefill segment plan")
    segments = []
    offset = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("malformed prefill segment")
        rows = entry.get("rows")
        start = entry.get("physical_start")
        if (isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
                or isinstance(start, bool) or not isinstance(start, int) or start != offset):
            raise ValueError("invalid or noncontiguous prefill segment plan")
        segments.append(rows)
        offset += rows
    return segments


def _validate_replay_artifacts(report: dict, call_ir: Path, engine_so: Path) -> None:
    decoder = report.get("decoder_runtime") or {}
    decoder_so = decoder.get("so_path")
    if not isinstance(decoder_so, str) or Path(decoder_so).resolve().parent != call_ir.resolve().parent:
        raise ValueError("lowered call is not from the parity report's decoder bundle")
    if engine_so.resolve().parent != call_ir.resolve().parent:
        raise ValueError("replay engine is not adjacent to the selected decoder bundle")
    manifest_path = engine_so.with_name(engine_so.name + ".build.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != _sha256(engine_so):
        raise ValueError("replay engine does not match its build manifest")


def replay_resolved_segmented_prefill(
    dumps: list[parity_test.ParityDump], layer: int, call: dict,
    segments: list[int], engine_so: Path, *, require_direct_cache: bool = True,
    independent_fp64: bool = False,
) -> dict:
    if not segments or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in segments):
        raise ValueError("segment lengths must be positive integers")
    heads, kv_heads, dim = call["num_heads"], call["num_kv_heads"], call["head_dim"]
    aligned = call["aligned_head_dim"]
    tokens = sum(segments)
    capacity = call["cache_capacity"]
    if tokens > capacity or heads % kv_heads or aligned != dim:
        raise ValueError("capture tokens or head geometry exceed the resolved call contract")

    q = _to_head_major(_unique_tensor(dumps, layer, "Qcur_rope"), tokens, heads, dim)
    k = _to_head_major(_unique_tensor(dumps, layer, "Kcur_rope"), tokens, kv_heads, dim)
    v = _to_head_major(_unique_tensor(dumps, layer, "Vcur_normed"), tokens, kv_heads, dim)
    captured = _reshape_token_major_qhd(_unique_tensor(dumps, layer, "kqv_out"), tokens, heads, dim)

    reconstructed_k = np.zeros((kv_heads, capacity, aligned), dtype=np.float16)
    reconstructed_v = np.zeros_like(reconstructed_k)
    reconstructed_k[:, :tokens, :dim] = k
    reconstructed_v[:, :tokens, :dim] = v
    cache_source = "reconstructed_fp16_from_pre_store_captures"
    reconstruction = None
    k_cache, v_cache = reconstructed_k, reconstructed_v
    if require_direct_cache:
        direct = {}
        for role in ("K", "V"):
            name = f"{role}cache_post_store"
            capture = _unique_capture(dumps, layer, name)
            if (capture.dtype != "fp16" or capture.token_id != 0 or
                    tuple(capture.data.shape) != (tokens, kv_heads, dim)):
                raise ValueError(f"{name} must be FP16 [token,kv_head,channel] from position zero")
            values = np.asarray(capture.data, dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains nonfinite values")
            cache = np.zeros((kv_heads, capacity, aligned), dtype=np.float16)
            cache[:, :tokens, :dim] = values.transpose(1, 0, 2)
            direct[role] = cache
        k_cache, v_cache = direct["K"], direct["V"]
        cache_source = "captured_post_store_fp16"
        reconstruction = {
            "k_bitwise_equal": bool(np.array_equal(
                k_cache[:, :tokens, :dim].view(np.uint16), reconstructed_k[:, :tokens, :dim].view(np.uint16))),
            "v_bitwise_equal": bool(np.array_equal(
                v_cache[:, :tokens, :dim].view(np.uint16), reconstructed_v[:, :tokens, :dim].view(np.uint16))),
            "k_metrics": _metrics(k_cache[:, :tokens, :dim].astype(np.float32),
                                  reconstructed_k[:, :tokens, :dim].astype(np.float32)),
            "v_metrics": _metrics(v_cache[:, :tokens, :dim].astype(np.float32),
                                  reconstructed_v[:, :tokens, :dim].astype(np.float32)),
        }
    output = np.zeros((heads, tokens, aligned), dtype=np.float32)
    workspace = np.zeros((call["workspace_bytes"] + 3) // 4, dtype=np.float32)
    segment_array = (ctypes.c_int * len(segments))(*segments)
    float_p = ctypes.POINTER(ctypes.c_float)
    half_p = ctypes.POINTER(ctypes.c_uint16)
    lib = ctypes.CDLL(str(engine_so.resolve()))
    fn = getattr(lib, call["function"])
    fn.argtypes = [
        float_p, half_p, half_p, float_p,
        *([ctypes.c_int] * 9), float_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ]
    fn.restype = ctypes.c_int
    status = fn(
        q.ctypes.data_as(float_p), k_cache.ctypes.data_as(half_p),
        v_cache.ctypes.data_as(half_p), output.ctypes.data_as(float_p),
        heads, kv_heads, tokens, 0, capacity, dim, aligned,
        call["sliding_window"], call["reduction"],
        workspace.ctypes.data_as(float_p), call["workspace_bytes"],
        segment_array, len(segments),
    )
    if status != 0:
        raise RuntimeError(f"selected attention provider returned {status}")
    replayed = _to_token_major(output, tokens, heads, aligned)[..., :dim]
    if not np.isfinite(replayed).all():
        raise RuntimeError("selected attention provider produced nonfinite output")
    bitwise = np.array_equal(replayed.view(np.uint32), captured.view(np.uint32))
    cache_matches = reconstruction is None or (
        reconstruction["k_bitwise_equal"] and reconstruction["v_bitwise_equal"])
    reference = None
    if independent_fp64:
        reference_out = np.zeros((tokens, heads, dim), dtype=np.float32)
        group_size = heads // kv_heads
        for query_pos in range(tokens):
            first_key = max(0, query_pos + 1 - call["sliding_window"])
            for head in range(heads):
                kv_head = head // group_size
                query = q[head, query_pos, :dim].astype(np.float64)
                keys = k_cache[kv_head, first_key:query_pos + 1, :dim].astype(np.float64)
                values = v_cache[kv_head, first_key:query_pos + 1, :dim].astype(np.float64)
                scores = keys @ query
                weights = np.exp(scores - np.max(scores))
                weights /= np.sum(weights)
                reference_out[query_pos, head] = weights @ values
        reference = {
            "contract": "unscaled causal sliding GQA, FP64 scores/softmax/value sum, FP32 output",
            "vs_captured_cke": _metrics(reference_out, captured),
            "vs_replayed_cke": _metrics(reference_out, replayed),
            "interpretation": "mathematical diagnostic; not the pinned oracle's reduction contract",
        }
    return {
        "status": "fail_replay_fidelity" if not bitwise else
                  "fail_cache_reconstruction" if not cache_matches else "pass",
        "interpretation": "replay fidelity only; not oracle parity or a local kernel-cause verdict",
        "layer": layer,
        "provider": call["function"],
        "kernel_id": call["kernel_id"],
        "reduction_selector": call["selector"],
        "segments": segments,
        "cache_source": cache_source,
        "cache_capture_is_direct": require_direct_cache,
        "cache_reconstruction": reconstruction,
        "original_runtime_loaded_identity_verified": False,
        "engine_so_sha256": _sha256(engine_so),
        "geometry": {"tokens": tokens, "heads": heads, "kv_heads": kv_heads,
                     "head_dim": dim, "cache_capacity": capacity,
                     "query_position_start": 0, "valid_cache_positions": [0, tokens]},
        "metrics": _metrics(replayed, captured),
        "bitwise_equal": bool(bitwise),
        "independent_fp64": reference,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay full-attention kernels from parity dumps")
    ap.add_argument("--dump", type=Path, required=True, help="Path to dump.bin")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument(
        "--mode",
        choices=("ggml", "exact", "flash", "tiled_f16kv"),
        default=None,
    )
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--disable-multihead-oracle", action="store_true")
    ap.add_argument("--disable-regular-oracle", action="store_true")
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--num-kv-heads", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=72)
    ap.add_argument("--aligned-head-dim", type=int, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--raw-output", type=Path, default=None)
    ap.add_argument("--resolved-call-ir", type=Path, default=None,
                    help="Replay the selected segmented FP16 prefill call from lowered IR")
    ap.add_argument("--segments", type=int, nargs="+", default=None,
                    help="Actual positive runtime segment lengths, in execution order")
    ap.add_argument("--engine-so", type=Path, default=None,
                    help="Engine library used for resolved-call replay")
    ap.add_argument("--parity-report", type=Path, default=None,
                    help="Report that records the dump and actual prefill segment plan")
    ap.add_argument("--allow-reconstructed-cache", action="store_true",
                    help="Diagnostic fallback for older dumps without post-store cache captures")
    ap.add_argument("--independent-fp64", action="store_true",
                    help="Compare captured operands with an explicit FP64 causal sliding reference")
    args = ap.parse_args(argv)

    dumps = parity_test.read_dump_file(args.dump)
    if args.resolved_call_ir is not None:
        if args.engine_so is None or args.parity_report is None:
            ap.error("--resolved-call-ir requires --engine-so and --parity-report")
        if args.mode is not None or args.raw_output is not None:
            ap.error("--mode and --raw-output are not supported with resolved-call replay")
        try:
            call_ir = json.loads(args.resolved_call_ir.read_text(encoding="utf-8"))
            parity_report = json.loads(args.parity_report.read_text(encoding="utf-8"))
            _validate_replay_artifacts(parity_report, args.resolved_call_ir, args.engine_so)
            segments = _segments_from_parity_report(parity_report, args.dump)
            if args.segments is not None and args.segments != segments:
                raise ValueError("requested segments do not match the retained parity report")
            call = _resolved_segmented_call(call_ir, args.layer)
            report = replay_resolved_segmented_prefill(
                dumps, args.layer, call, segments, args.engine_so,
                require_direct_cache=not args.allow_reconstructed_cache,
                independent_fp64=args.independent_fp64,
            )
        except (ValueError, KeyError, AttributeError, OSError) as exc:
            report = {
                "status": "incomplete",
                "interpretation": "replay evidence is missing or inconsistent; no kernel-cause verdict",
                "error": str(exc),
            }
        report["artifacts"] = {
            "dump_sha256": _sha256(args.dump),
            "call_ir_sha256": _sha256(args.resolved_call_ir),
            "parity_report_sha256": _sha256(args.parity_report),
        }
        print(json.dumps(report, indent=2))
        if args.output is not None:
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return {"pass": 0, "fail_replay_fidelity": 1, "fail_cache_reconstruction": 1}.get(report["status"], 2)
    if args.segments is not None or args.engine_so is not None or args.parity_report is not None:
        ap.error("--segments, --engine-so and --parity-report require --resolved-call-ir")
    q_flat = _require_tensor_any(dumps, args.layer, ["Qcur_rope"])
    num_tokens = _infer_num_tokens(q_flat, args.num_heads, args.head_dim)
    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else args.num_heads
    aligned_head_dim = args.aligned_head_dim if args.aligned_head_dim is not None else args.head_dim

    q = _to_head_major(q_flat, num_tokens, args.num_heads, args.head_dim)
    k = _to_head_major(_require_tensor_any(dumps, args.layer, ["Kcur_rope"]), num_tokens, num_kv_heads, args.head_dim)
    v = _to_head_major(_require_tensor_any(dumps, args.layer, ["v_proj", "Vcur"]), num_tokens, num_kv_heads, args.head_dim)
    ref = _reshape_token_major_qhd(_require_tensor_any(dumps, args.layer, ["kqv_out"]), num_tokens, args.num_heads, args.head_dim)

    lib = ctypes.CDLL(str(BUILD_DIR / "libckernel_engine.so"))
    lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
    lib.ck_set_strict_parity.restype = None
    lib.ck_set_strict_parity(1 if args.strict else 0)

    old_multi = os.environ.get("CK_STRICT_DISABLE_MULTIHEAD_ATTN_ORACLE")
    old_regular = os.environ.get("CK_STRICT_DISABLE_REGULAR_ATTN_ORACLE")
    old_ggml_cpu = os.environ.get("CK_GGML_CPU_SO")
    try:
        if args.strict and old_ggml_cpu is None:
            ggml_cpu_so = _default_ggml_cpu_so()
            if ggml_cpu_so is not None:
                os.environ["CK_GGML_CPU_SO"] = ggml_cpu_so
        if args.disable_multihead_oracle:
            os.environ["CK_STRICT_DISABLE_MULTIHEAD_ATTN_ORACLE"] = "1"
        else:
            os.environ.pop("CK_STRICT_DISABLE_MULTIHEAD_ATTN_ORACLE", None)
        if args.disable_regular_oracle:
            os.environ["CK_STRICT_DISABLE_REGULAR_ATTN_ORACLE"] = "1"
        else:
            os.environ.pop("CK_STRICT_DISABLE_REGULAR_ATTN_ORACLE", None)

        fn = _resolve_attention_fn(lib, args.mode or "ggml")
        out = np.zeros((args.num_heads, num_tokens, aligned_head_dim), dtype=np.float32)
        fn(
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            args.num_heads,
            num_kv_heads,
            num_tokens,
            args.head_dim,
            aligned_head_dim,
            num_tokens,
        )
    finally:
        if old_multi is None:
            os.environ.pop("CK_STRICT_DISABLE_MULTIHEAD_ATTN_ORACLE", None)
        else:
            os.environ["CK_STRICT_DISABLE_MULTIHEAD_ATTN_ORACLE"] = old_multi
        if old_regular is None:
            os.environ.pop("CK_STRICT_DISABLE_REGULAR_ATTN_ORACLE", None)
        else:
            os.environ["CK_STRICT_DISABLE_REGULAR_ATTN_ORACLE"] = old_regular
        if old_ggml_cpu is None:
            os.environ.pop("CK_GGML_CPU_SO", None)
        else:
            os.environ["CK_GGML_CPU_SO"] = old_ggml_cpu
        lib.ck_set_strict_parity(0)

    token_major = _to_token_major(out, num_tokens, args.num_heads, aligned_head_dim)[..., : args.head_dim]
    metrics = _metrics(token_major, ref)
    bitwise_equal = token_major.view(np.uint32) == ref.view(np.uint32)
    worst_idx = np.unravel_index(np.argmax(np.abs(token_major - ref)), token_major.shape)
    report = {
        "mode": args.mode or "ggml",
        "strict": args.strict,
        "num_tokens": num_tokens,
        "num_heads": args.num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": args.head_dim,
        "aligned_head_dim": aligned_head_dim,
        "disable_multihead_oracle": args.disable_multihead_oracle,
        "disable_regular_oracle": args.disable_regular_oracle,
        "metrics": metrics,
        "bitwise": {
            "equal": int(np.count_nonzero(bitwise_equal)),
            "different": int(bitwise_equal.size - np.count_nonzero(bitwise_equal)),
            "total": int(bitwise_equal.size),
        },
        "worst_index": {
            "query": int(worst_idx[0]),
            "head": int(worst_idx[1]),
            "dim": int(worst_idx[2]),
            "actual": float(token_major[worst_idx]),
            "expected": float(ref[worst_idx]),
        },
    }

    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.raw_output is not None:
        np.save(args.raw_output, token_major)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
