#!/usr/bin/env python3
"""Compare two generated bundles on identical prompt and continuation tokens.

This is a numerical diagnostic, not model-quality certification. Each bundle
runs in a child process so dynamic libraries and model-global state cannot
alias across comparisons. Loaded symbol mappings are checked against bundle
files before their hashes are reported.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _probe(bundle: Path, prompt: list[int], continuation: list[int], output: Path,
           split_at: int | None = None) -> None:
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
        segments = ([prompt[:split_at], prompt[split_at:]]
                    if split_at is not None else [prompt])
        for segment in segments:
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
            json.dumps({"stages": stages, "loaded_libraries": loaded}, indent=2) + "\n", encoding="utf-8")
    finally:
        lib.ck_model_free()


def _comparison(left: Path, right: Path, label: str,
                left_stage: dict, right_stage: dict) -> dict:
    left_bytes = (left / f"{label}.f32").read_bytes()
    right_bytes = (right / f"{label}.f32").read_bytes()
    a = np.frombuffer(left_bytes, dtype="<f4").astype(np.float64)
    b = np.frombuffer(right_bytes, dtype="<f4").astype(np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError(f"invalid logits comparison at {label}")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    kv_left, kv_right = left_stage["kv_sha256"], right_stage["kv_sha256"]
    if len(kv_left) != len(kv_right) or left_stage["position"] != right_stage["position"]:
        raise RuntimeError(f"state geometry differs at {label}")
    return {"label": label, "position": left_stage["position"],
            "logits_exact": left_bytes == right_bytes,
            "max_logit_abs": float(np.max(np.abs(a - b))),
            "logit_cosine": float(np.dot(a, b) / denominator) if denominator else None,
            "top_tokens": [left_stage["top_token"], right_stage["top_token"]],
            "kv_exact_layers": sum(x == y for x, y in zip(kv_left, kv_right)),
            "kv_mismatch_layers": [index for index, (x, y) in
                                   enumerate(zip(kv_left, kv_right)) if x != y],
            "kv_layers": len(kv_left)}


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
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        _probe(args.full.resolve(), _tokens(args.tokens),
               _tokens(args.continuation) if args.continuation else [],
               args.output.resolve(), args.split_at)
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
    continuation = _tokens(args.continuation) if args.continuation else []
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
            subprocess.run(command, check=True)
        left = json.loads((root / "full" / "probe.json").read_text())
        right = json.loads((root / "chunked" / "probe.json").read_text())
        loaded = {"full": left["loaded_libraries"],
                  "chunked": right["loaded_libraries"]}
        for side in ("full", "chunked"):
            for name, digest in loaded[side].items():
                if digest != identities[side][name]:
                    raise RuntimeError(f"loaded {side} {name} differs from preflight")
        if [s["label"] for s in left["stages"]] != [s["label"] for s in right["stages"]]:
            raise RuntimeError("stage inventory differs")
        comparisons = [_comparison(root / "full", root / "chunked", a["label"], a, b)
                       for a, b in zip(left["stages"], right["stages"])]
    report = {"schema": "cke.chunked_prefill_comparison.v1",
              "prompt_tokens": len(prompt), "continuation_tokens": len(continuation),
              "split_at": args.split_at,
              "token_sha256": _sha256(args.tokens), "identities": identities,
              "loaded_libraries": loaded,
              "same_engine_build": loaded["full"]["libckernel_engine.so"] ==
                                   loaded["chunked"]["libckernel_engine.so"],
              "execution": {"machine": platform.machine(),
                            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                            "strict_parity": os.environ.get("CK_STRICT_PARITY")},
              "comparisons": comparisons,
              "exact": all(item["logits_exact"] and
                           item["kv_exact_layers"] == item["kv_layers"]
                           for item in comparisons)}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exact": report["exact"], "comparisons": comparisons}))
    return 1 if args.require_exact and not (
        report["exact"] and report["same_engine_build"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
