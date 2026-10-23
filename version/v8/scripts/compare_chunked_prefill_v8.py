#!/usr/bin/env python3
"""Compare two generated bundles on identical prompt and continuation tokens.

This is a numerical diagnostic, not model-quality certification. Each bundle
runs in a child process so dynamic libraries and model-global state cannot
alias across comparisons. Loaded symbol mappings are checked against bundle
files before their hashes are reported. X-Ray aligns the observed stages by
semantic boundary rather than by prefill call count. Only KV-only layouts are
admitted until recurrent and convolution state exporters are available.

For an asset-backed sweep, pass --matrix-lengths, --matrix-chunk-sizes and
--matrix-threads with the ordinary --full, --chunked and --tokens inputs.
Use --matrix-splits 0,120 to compare chunking from the start and staged append
after an existing 120-token prefix. A failing or missing case never counts as
a matrix pass. The original #872 one-call fixture uses no append option.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import itertools
import subprocess
import sys
import tempfile

import numpy as np

import xray_execution_state_v8
from xray_execution_state_v8 import compare_equivalent_stage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(("model name", "Hardware", "Processor")) and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _source_revision() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
        capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _identity(bundle: Path) -> dict[str, str]:
    return {name: _sha256(bundle / name) for name in (
        "weights.bump", "weights_manifest.json", "weights_manifest.map",
        "config.json", "model_v8.c", "libmodel.so",
        "libckernel_engine.so", "libckernel_tokenizer.so",
    )}


def _loaded_identity(library: ctypes.CDLL, bundle: Path,
                     maps_text: str | None = None) -> dict[str, str]:
    """Bind each exported symbol to an intact executable file mapping."""
    if maps_text is None:
        maps_text = Path("/proc/self/maps").read_text(encoding="utf-8")
    identities = {}
    for name, symbol in (
        ("libmodel.so", "ck_model_decode"),
        ("libckernel_engine.so", "ck_set_num_threads"),
        ("libckernel_tokenizer.so", "ck_tokenizer_create"),
    ):
        identities[name] = _loaded_symbol_digest(
            library, symbol, (bundle / name).resolve(), maps_text)
    return identities


def _loaded_symbol_digest(library: ctypes.CDLL, symbol: str,
                          expected: Path, maps_text: str) -> str:
    address = ctypes.cast(getattr(library, symbol), ctypes.c_void_p).value
    if not address:
        raise RuntimeError(f"missing loaded symbol: {symbol}")
    matches = []
    for row in maps_text.splitlines():
        fields = row.split(maxsplit=5)
        if len(fields) != 6 or "x" not in fields[1]:
            continue
        lo, hi = (int(part, 16) for part in fields[0].split("-", 1))
        if lo <= address < hi:
            matches.append(fields)
    if len(matches) != 1:
        raise RuntimeError(f"loaded symbol has no unique mapping: {symbol}")
    fields = matches[0]
    if fields[5].endswith(" (deleted)") or Path(fields[5]).resolve() != expected:
        raise RuntimeError(f"loaded library path mismatch: {expected.name}")
    with expected.open("rb") as source:
        stat = os.fstat(source.fileno())
        major, minor = (int(part, 16) for part in fields[3].split(":"))
        if (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino) != (
                major, minor, int(fields[4])):
            raise RuntimeError(f"loaded library backing changed: {expected.name}")
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        if (expected.stat().st_dev, expected.stat().st_ino) != (
                stat.st_dev, stat.st_ino):
            raise RuntimeError(f"loaded library backing changed: {expected.name}")
    return digest.hexdigest()


def _tokens(path: Path) -> list[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, list) or not value or
            any(not isinstance(token, int) or isinstance(token, bool) or
                token < 0 or token >= 2**31 for token in value)):
        raise ValueError(f"{path} must contain a nonempty JSON list of token IDs")
    return value


def _segments(prompt: list[int], split_at: int | None,
              append_chunk_size: int | None) -> list[tuple[int, list[int]]]:
    """Return absolute-position token ranges for initial and repeated appends."""
    if append_chunk_size is not None and append_chunk_size <= 0:
        raise ValueError("append chunk size must be positive")
    cuts = [0, split_at, len(prompt)] if split_at is not None else [0, len(prompt)]
    segments = []
    for start, end in zip(cuts, cuts[1:]):
        step = append_chunk_size or end - start
        for offset in range(start, end, step):
            segments.append((offset, prompt[offset:min(offset + step, end)]))
    return segments


def _kv_only_state_contract(bundle: Path) -> dict:
    """Fail closed when this comparator cannot export all mutable sequence state."""
    path = bundle / "layout_decode.json"
    layout = json.loads(path.read_text(encoding="utf-8"))
    buffers = layout["memory"]["activations"]["buffers"]
    unknown = sorted(buffer["name"] for buffer in buffers
                     if buffer.get("lifetime") not in {"call", "model", "sequence"}
                     or (buffer.get("lifetime") == "model" and buffer.get("mutable", True)))
    if unknown:
        raise RuntimeError(f"unknown persistent mutable state: {unknown!r}")
    sequence = sorted(buffer["name"] for buffer in buffers
                      if buffer.get("lifetime") == "sequence" and buffer.get("mutable", True))
    if sequence != ["kv_cache"]:
        raise RuntimeError(f"unexported persistent sequence state: {sequence!r}")
    return {"exported_state": sequence, "layout_decode_sha256": _sha256(path)}


def _probe(bundle: Path, prompt: list[int], continuation: list[int], output: Path,
           split_at: int | None = None, append_chunk_size: int | None = None) -> None:
    state_contract = _kv_only_state_contract(bundle)
    lib = ctypes.CDLL(str(bundle / "libmodel.so"), mode=ctypes.RTLD_GLOBAL)
    loaded = _loaded_identity(lib, bundle)
    int_ptr = ctypes.POINTER(ctypes.c_int32)
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [int_ptr, ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_get_logits_stride.restype = ctypes.c_int
    lib.ck_model_get_active_tokens.restype = ctypes.c_int
    lib.ck_model_debug_export_kv.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.ck_model_debug_export_kv.restype = ctypes.c_int
    lib.ck_model_free.restype = None
    if lib.ck_model_init(str(bundle / "weights.bump").encode()) != 0:
        raise RuntimeError("ck_model_init failed")
    try:
        lib.ck_get_num_threads.restype = ctypes.c_int
        native_threads = int(lib.ck_get_num_threads())
        segments = _segments(prompt, split_at, append_chunk_size)
        for _, segment in segments:
            rows = (ctypes.c_int32 * len(segment))(*segment)
            if lib.ck_model_embed_tokens(rows, len(segment)) != 0:
                raise RuntimeError("ck_model_embed_tokens failed")
            if lib.ck_model_forward(None) != 0:
                raise RuntimeError("ck_model_forward failed")
        layers = int(json.loads((bundle / "config.json").read_text())["num_hidden_layers"])
        vocab = int(lib.ck_model_get_vocab_size())
        stages = []

        def capture(label: str) -> None:
            stride = int(lib.ck_model_get_logits_stride())
            active = int(lib.ck_model_get_active_tokens())
            pointer = lib.ck_model_get_logits()
            if not pointer or vocab <= 0 or active <= 0:
                raise RuntimeError(f"missing logits at {label}")
            if stride > 0:
                if stride < vocab:
                    raise RuntimeError("logits stride smaller than vocabulary")
                raw = np.ctypeslib.as_array(pointer, shape=(active * stride,))
                logits = raw[(active - 1) * stride : (active - 1) * stride + vocab]
            else:
                logits = np.ctypeslib.as_array(pointer, shape=(vocab,))
            logits = np.array(logits, dtype="<f4", copy=True)
            if not np.isfinite(logits).all():
                raise RuntimeError(f"nonfinite logits at {label}")
            (output / f"{label}.f32").write_bytes(logits.tobytes())
            kv = []
            with tempfile.TemporaryDirectory() as temporary:
                for layer in range(layers):
                    path = Path(temporary) / f"kv-{layer}.bin"
                    if lib.ck_model_debug_export_kv(str(path).encode(), layer) != 0:
                        raise RuntimeError(f"KV export failed at layer {layer}")
                    kv.append(_sha256(path))
            stages.append({"label": label, "position": active,
                           "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
                           "top_token": int(np.argmax(logits)), "kv_sha256": kv})

        capture("prefill")
        for index, token in enumerate(continuation):
            if lib.ck_model_decode(token, None) != 0:
                raise RuntimeError(f"decode failed at continuation {index}")
            capture(f"decode-{index + 1:03d}")
        (output / "probe.json").write_text(
            json.dumps({"stages": stages, "loaded_libraries": loaded,
                        "native_threads": native_threads,
                        "state_contract": state_contract,
                        "calls": [{"start": start, "count": len(segment)}
                                  for start, segment in segments]}, indent=2) + "\n", encoding="utf-8")
    finally:
        lib.ck_model_free()


def _comparison(left: Path, right: Path, label: str,
                left_stage: dict, right_stage: dict) -> dict:
    left_bytes = (left / f"{label}.f32").read_bytes()
    right_bytes = (right / f"{label}.f32").read_bytes()
    a = np.frombuffer(left_bytes, dtype="<f4")
    b = np.frombuffer(right_bytes, dtype="<f4")
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError(f"invalid logits comparison at {label}")
    kv_left, kv_right = left_stage["kv_sha256"], right_stage["kv_sha256"]
    if len(kv_left) != len(kv_right):
        raise RuntimeError(f"state geometry differs at {label}")
    xray = compare_equivalent_stage(
        a, b, subject_position=right_stage["position"],
        reference_position=left_stage["position"],
        subject_state={str(i): digest for i, digest in enumerate(kv_right)},
        reference_state={str(i): digest for i, digest in enumerate(kv_left)},
        contract={"mode": "exact"})
    return {"label": label, "position": left_stage["position"],
            "logits_exact": xray["logits_exact"],
            "max_logit_abs": xray["metrics"]["max_abs"],
            "logit_cosine": xray["metrics"]["cosine"],
            "top_tokens": [left_stage["top_token"], right_stage["top_token"]],
            "kv_exact_layers": sum(x == y for x, y in zip(kv_left, kv_right)),
            "kv_mismatch_layers": [index for index, (x, y) in
                                   enumerate(zip(kv_left, kv_right)) if x != y],
            "kv_layers": len(kv_left), "xray": xray}


def _positive_csv(raw: str, label: str) -> list[int]:
    try:
        values = [int(part) for part in raw.split(",")]
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values) or len(values) != len(set(values)):
        raise ValueError(f"{label} must contain distinct positive integers")
    return values


def _matrix_cases(lengths: str, chunks: str, threads: str,
                  splits: str, available: int) -> list[tuple[int, int, int, int | None]]:
    sizes = _positive_csv(lengths, "matrix lengths")
    chunk_sizes = _positive_csv(chunks, "matrix chunk sizes")
    thread_counts = _positive_csv(threads, "matrix threads")
    try:
        split_positions = [None if part == "0" else int(part)
                           for part in splits.split(",")]
    except ValueError as exc:
        raise ValueError("matrix splits must be comma-separated integers") from exc
    if not split_positions or len(split_positions) != len(set(split_positions)):
        raise ValueError("matrix splits must contain distinct positions")
    if any(length > available for length in sizes):
        raise ValueError("matrix length exceeds serialized token fixture")
    cases = list(itertools.product(sizes, chunk_sizes, thread_counts, split_positions))
    if len(cases) > 64:
        raise ValueError("matrix exceeds 64 cases")
    for length, chunk, _, split in cases:
        if chunk >= length:
            raise ValueError("every matrix chunk must be shorter than its prompt")
        if split is not None and not 0 < split < length:
            raise ValueError("matrix split must be inside every prompt")
    return cases


def _run_matrix(args: argparse.Namespace, prompt: list[int]) -> int:
    if not args.matrix_chunk_sizes or not args.matrix_threads:
        raise ValueError("matrix requires --matrix-chunk-sizes and --matrix-threads")
    cases = _matrix_cases(args.matrix_lengths, args.matrix_chunk_sizes,
                          args.matrix_threads, args.matrix_splits, len(prompt))
    case_dir = args.json_out.parent / f"{args.json_out.stem}-cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, (length, chunk, threads, split) in enumerate(cases):
        token_path = case_dir / f"case-{index:03d}-tokens.json"
        report_path = case_dir / f"case-{index:03d}.json"
        report_path.unlink(missing_ok=True)
        token_path.write_text(json.dumps(prompt[:length]) + "\n", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--full", str(args.full.resolve()),
                   "--chunked", str(args.chunked.resolve()),
                   "--tokens", str(token_path),
                   "--json-out", str(report_path),
                   "--append-chunk-size", str(chunk), "--require-exact"]
        if args.continuation:
            command.extend(("--continuation", str(args.continuation.resolve())))
        if split is not None:
            command.extend(("--split-at", str(split)))
        environment = dict(os.environ, OMP_NUM_THREADS=str(threads), CK_NUM_THREADS=str(threads))
        try:
            run = subprocess.run(command, env=environment, capture_output=True,
                                 text=True, timeout=args.matrix_case_timeout_sec)
        except subprocess.TimeoutExpired:
            run = subprocess.CompletedProcess(command, 124, "", "case execution timed out")
        try:
            detail = json.loads(report_path.read_text()) if report_path.is_file() else None
        except (OSError, ValueError):
            detail = None
        bound = bool(detail and
                     detail.get("schema") == "cke.chunked_prefill_comparison.v1" and
                     detail.get("prompt_tokens") == length and
                     detail.get("append_chunk_size") == chunk and
                     detail.get("split_at") == split and
                     detail.get("token_sha256") == _sha256(token_path) and
                     detail.get("execution", {}).get("omp_num_threads") == str(threads) and
                     detail.get("execution", {}).get("native_threads") ==
                     {"full": threads, "chunked": threads})
        status = ("pass" if run.returncode == 0 and bound and detail.get("exact") and
                  detail.get("same_engine_build") else
                  "fail" if detail or run.returncode in (0, 124) else "not_tested")
        results.append({"length": length, "append_chunk_size": chunk,
                        "threads": threads, "split_at": split,
                        "status": status, "report": str(report_path) if detail else None,
                        "token_sha256": _sha256(token_path), "command": command,
                        "error": run.stderr[-2000:] if status != "pass" else None})
    status = "pass" if all(case["status"] == "pass" for case in results) else "fail"
    report = {"schema": "cke.xray_execution_equivalence_matrix.v1",
              "status": status, "subject": "chunked", "reference": "uninterrupted",
              "source_token_sha256": _sha256(args.tokens),
              "source_revision": _source_revision(),
              "continuation_sha256": _sha256(args.continuation) if args.continuation else None,
              "machine": platform.machine(), "cpu_model": _cpu_model(),
              "comparison_source_sha256": _sha256(Path(__file__)),
              "xray_source_sha256": _sha256(Path(xray_execution_state_v8.__file__)),
              "cases": results}
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "cases": len(results)}))
    return 0 if status == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path)
    parser.add_argument("--chunked", type=Path)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--continuation", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--split-at", type=int,
                        help="Append at this position in the chunked child (nonzero-position test)")
    parser.add_argument("--append-chunk-size", type=int,
                        help="Split the chunked child into repeated append calls of at most this many tokens")
    parser.add_argument("--matrix-lengths", help="Comma-separated prompt lengths for a bounded equivalence sweep")
    parser.add_argument("--matrix-chunk-sizes", help="Comma-separated repeated-append sizes for the sweep")
    parser.add_argument("--matrix-threads", help="Comma-separated native/OpenMP thread counts for the sweep")
    parser.add_argument("--matrix-splits", default="0",
                        help="Comma-separated initial append positions; 0 means chunk from the start")
    parser.add_argument("--matrix-case-timeout-sec", type=int, default=3600,
                        help="Per-case execution deadline in seconds (default 3600)")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        _probe(args.full.resolve(), _tokens(args.tokens),
               _tokens(args.continuation) if args.continuation else [],
               args.output.resolve(), args.split_at, args.append_chunk_size)
        return 0
    if not args.full or not args.chunked or not args.tokens or not args.json_out:
        parser.error("--full, --chunked, --tokens and --json-out are required")
    full, chunked = args.full.resolve(), args.chunked.resolve()
    identities = {"full": _identity(full), "chunked": _identity(chunked)}
    for name in ("weights.bump", "weights_manifest.json", "weights_manifest.map"):
        if identities["full"][name] != identities["chunked"][name]:
            raise RuntimeError(f"bundle {name} differs")
    prompt = _tokens(args.tokens)
    if args.split_at is not None and not (0 < args.split_at < len(prompt)):
        parser.error("--split-at must be inside the prompt")
    if args.append_chunk_size is not None and args.append_chunk_size <= 0:
        parser.error("--append-chunk-size must be positive")
    continuation = _tokens(args.continuation) if args.continuation else []
    if args.matrix_lengths:
        if args.split_at is not None or args.append_chunk_size is not None:
            parser.error("matrix options replace --split-at and --append-chunk-size")
        if args.matrix_case_timeout_sec <= 0:
            parser.error("--matrix-case-timeout-sec must be positive")
        return _run_matrix(args, prompt)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name, bundle in (("full", full), ("chunked", chunked)):
            output = root / name
            output.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--child", "--full", str(bundle), "--tokens", str(args.tokens),
                       "--output", str(output)]
            if args.continuation:
                command.extend(("--continuation", str(args.continuation)))
            if name == "chunked" and args.split_at is not None:
                command.extend(("--split-at", str(args.split_at)))
            if name == "chunked" and args.append_chunk_size is not None:
                command.extend(("--append-chunk-size", str(args.append_chunk_size)))
            subprocess.run(command, check=True)
        left = json.loads((root / "full" / "probe.json").read_text())
        right = json.loads((root / "chunked" / "probe.json").read_text())
        loaded = {"full": left["loaded_libraries"],
                  "chunked": right["loaded_libraries"]}
        if left["state_contract"]["exported_state"] != right["state_contract"]["exported_state"]:
            raise RuntimeError("sequence state inventories differ")
        for side in ("full", "chunked"):
            for name, digest in loaded[side].items():
                if digest != identities[side][name]:
                    raise RuntimeError(f"loaded {side} {name} differs from preflight")
        if [s["label"] for s in left["stages"]] != [s["label"] for s in right["stages"]]:
            raise RuntimeError("stage inventory differs")
        comparisons = [_comparison(root / "full", root / "chunked", a["label"], a, b)
                       for a, b in zip(left["stages"], right["stages"])]
    composition_pass = all(item["xray"]["logits_within_contract"]
                           for item in comparisons)
    state_pass = all(item["xray"]["state_exact"] and
                     item["xray"]["subject_position"] == item["xray"]["reference_position"]
                     for item in comparisons)
    report = {"schema": "cke.chunked_prefill_comparison.v1",
              "prompt_tokens": len(prompt), "continuation_tokens": len(continuation),
              "split_at": args.split_at,
              "append_chunk_size": args.append_chunk_size,
              "execution_calls": {"full": left["calls"], "chunked": right["calls"]},
              "state_contract": {"full": left["state_contract"],
                                 "chunked": right["state_contract"]},
              "token_sha256": _sha256(args.tokens), "identities": identities,
              "source_revision": _source_revision(),
              "loaded_libraries": loaded,
              "same_engine_build": loaded["full"]["libckernel_engine.so"] ==
                                   loaded["chunked"]["libckernel_engine.so"],
              "execution": {"machine": platform.machine(),
                            "cpu_model": _cpu_model(),
                            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                            "ck_num_threads": os.environ.get("CK_NUM_THREADS"),
                            "native_threads": {"full": left["native_threads"],
                                               "chunked": right["native_threads"]},
                            "strict_parity": os.environ.get("CK_STRICT_PARITY")},
              "comparison_source_sha256": _sha256(Path(__file__)),
              "xray_source_sha256": _sha256(Path(xray_execution_state_v8.__file__)),
              "comparisons": comparisons,
              "first_observed_divergence": next(
                  ({"label": item["label"], **item["xray"]}
                   for item in comparisons if item["xray"]["status"] == "fail"), None),
              "verdicts": {
                  "arithmetic_reference": "not_tested",
                  "reduction_equivalence": "not_tested",
                  "generated_composition": "pass" if composition_pass else "fail",
                  "state_continuity": "pass" if state_pass else "fail",
                  "negative_controls": "not_tested",
                  "artifact_identity": "pass" if loaded["full"]["libckernel_engine.so"] ==
                      loaded["chunked"]["libckernel_engine.so"] else "fail",
                  "recovery": "not_tested",
              },
              "exact": all(item["xray"]["status"] == "pass"
                           for item in comparisons)}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exact": report["exact"], "comparisons": comparisons}))
    return 1 if args.require_exact and not (
        report["exact"] and report["same_engine_build"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
