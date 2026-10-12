#!/usr/bin/env python3
"""Compare one generated mixed row pair with serialized decode replay.

An optional, separately reported comparison checks staged prefill against the
ordinary generated prefill entry. It does not certify variable batch size or
HTTP concurrency.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from server.serving_bundle import verified_loaded_symbol_backing
from server.session_v8 import SESSION_LIB_PATH, SessionV8

CAP_BATCH_TWO_ROWS = 1 << 17


def _finite_logit_bytes(values: ctypes.Array, label: str) -> bytes:
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise AssertionError(f"{label} has nonfinite logit at index {index}")
    return bytes(values)


def _compare_prefill_snapshots(reference: tuple[bytes, int, str],
                               staged: tuple[bytes, int, str]) -> dict:
    """Keep arithmetic, position, and complete-KV verdicts independent."""
    ref_logits, ref_position, ref_kv = reference
    got_logits, got_position, got_kv = staged
    if len(ref_logits) != len(got_logits) or len(ref_logits) % 4:
        raise AssertionError("ordinary/staged logit extents differ")
    exact_logits = ref_logits == got_logits
    ref = array("f")
    got = array("f")
    ref.frombytes(ref_logits)
    got.frombytes(got_logits)
    if any(not math.isfinite(value) for value in ref) or any(
            not math.isfinite(value) for value in got):
        raise AssertionError("ordinary/staged logits contain nonfinite values")
    ref_top = max(range(len(ref)), key=ref.__getitem__)
    got_top = max(range(len(got)), key=got.__getitem__)
    if exact_logits:
        max_abs = rmse = 0.0
    else:
        squared = 0.0
        max_abs = 0.0
        for left, right in zip(ref, got, strict=True):
            delta = abs(left - right)
            max_abs = max(max_abs, delta)
            squared += delta * delta
        rmse = math.sqrt(squared / len(ref))
    exact_kv = ref_kv == got_kv
    return {
        "status": "pass" if exact_logits and exact_kv and ref_position == got_position else "fail",
        "logits_exact": exact_logits,
        "max_abs_logit_diff": max_abs,
        "logit_rmse": rmse,
        "top1_ordinary": ref_top,
        "top1_staged": got_top,
        "top1_equal": ref_top == got_top,
        "kv_exact": exact_kv,
        "kv_sha256_ordinary": ref_kv,
        "kv_sha256_staged": got_kv,
        "position_ordinary": ref_position,
        "position_staged": got_position,
    }


def _logit_distance(subject: bytes, oracle: bytes) -> dict:
    """Compare complete finite float32 vectors without a prose-quality verdict."""
    if len(subject) != len(oracle) or not subject or len(subject) % 4:
        raise AssertionError("subject/oracle logit extents differ")
    left = array("f")
    right = array("f")
    left.frombytes(subject)
    right.frombytes(oracle)
    if any(not math.isfinite(value) for value in left) or any(
            not math.isfinite(value) for value in right):
        raise AssertionError("subject/oracle logits contain nonfinite values")
    max_abs = squared = left_norm = right_norm = dot = 0.0
    for a, b in zip(left, right, strict=True):
        delta = a - b
        max_abs = max(max_abs, abs(delta))
        squared += delta * delta
        left_norm += a * a
        right_norm += b * b
        dot += a * b
    denominator = math.sqrt(left_norm * right_norm)
    return {
        "max_abs_logit_diff": max_abs,
        "logit_rmse": math.sqrt(squared / len(left)),
        "cosine": dot / denominator if denominator else 0.0,
        "top1_subject": max(range(len(left)), key=left.__getitem__),
        "top1_oracle": max(range(len(right)), key=right.__getitem__),
    }


def _require_shared_projections(capabilities: int, groups: int | None) -> int:
    if not capabilities & CAP_BATCH_TWO_ROWS:
        raise AssertionError("loaded model lacks the two-row batch capability")
    if groups is None or groups <= 0:
        raise AssertionError("loaded model has no shared projection groups")
    return groups


def _cpu_identity() -> dict:
    model = platform.processor() or platform.machine()
    flags = set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.partition(":")[2].strip()
            if line.startswith(("flags", "Features")):
                flags.update(line.partition(":")[2].split())
            if model and flags:
                break
    except OSError:
        pass
    relevant = ("avx", "avx2", "fma", "avx512f", "avx512_vnni", "asimd", "sve")
    return {"model": model, "machine": platform.machine(),
            "isa_flags": [flag for flag in relevant if flag in flags]}


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(part)
    return sha.hexdigest()


def certify(bundle: Path, *, threads: int = 4, context_length: int = 32,
            compare_ordinary: bool = False,
            oracle_gguf: Path | None = None) -> dict:
    bundle = bundle.resolve()
    if threads <= 0 or context_length < 16:
        raise ValueError("certification requires positive threads and at least 16 context tokens")
    if compare_ordinary and os.environ.get("CK_V8_FORCE_DECODE_PREFILL", "0") not in ("", "0"):
        raise ValueError("ordinary prefill comparison cannot force decode replay")
    if oracle_gguf is not None and not compare_ordinary:
        raise ValueError("llama.cpp oracle requires the ordinary prefill comparison")
    lengths = sorted({1, 2, 7, 8, 9, 10, 12, 16,
                      context_length - 4, context_length - 1})
    oracle_rows = {}
    oracle_identity = None
    if oracle_gguf is not None:
        from version.v8.scripts.compare_first_token_logits_v8 import (
            HELPER_SRC, LLAMA_CPP, run_llama_logits,
        )
        oracle_gguf = oracle_gguf.resolve(strict=True)
        expected_pin = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[3]),
             "ls-tree", "HEAD", "llama.cpp"], text=True).split()[2]
        actual_pin = subprocess.check_output(
            ["git", "-C", str(LLAMA_CPP), "rev-parse", "HEAD"],
            text=True).strip()
        if actual_pin != expected_pin:
            raise AssertionError("llama.cpp checkout differs from the repository pin")
        lib_dir = LLAMA_CPP / "build" / "bin"
        libraries = ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so")
        oracle_identity = {
            "gguf_sha256": _digest(oracle_gguf),
            "llama_commit": actual_pin,
            "helper_source_sha256": _digest(HELPER_SRC),
            "library_file_sha256": {name: _digest(lib_dir / name) for name in libraries},
            "source_pair_verified": False,
            "status": "diagnostic_only",
        }
        for length in sorted({1, 9, 10, context_length - 4}):
            tokens = list(range(100, 100 + length))
            oracle_rows[length] = {}
            for mode in ("batched", "sequential"):
                result = run_llama_logits(oracle_gguf, tokens,
                                          max(64, context_length), 16, threads,
                                          decode_mode=mode)
                if (int(result["meta"]["n_vocab"]) != len(result["logits"]) or
                        int(result["meta"]["token_count"]) != length):
                    raise AssertionError("llama.cpp returned the wrong token or vocabulary extent")
                oracle_rows[length][mode] = result["logits"].tobytes()
    session = SessionV8.open(bundle, context_length=context_length, num_threads=threads)
    try:
        batch = session.enable_batch2(1 << 30)
        model = ctypes.CDLL(str(bundle / "libmodel.so"))
        model.ck_model_sequence_state_requirements.argtypes = [
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        model.ck_model_sequence_state_requirements.restype = ctypes.c_int
        model.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
        model.ck_model_get_named_activation_ptr.restype = ctypes.c_size_t
        model.ck_model_generated_source_sha256.restype = ctypes.c_char_p
        model.ck_model_get_capabilities.restype = ctypes.c_uint64
        model.ck_model_get_context_window.restype = ctypes.c_int
        if not hasattr(model, "ck_model_batch_decode_projection_groups"):
            raise AssertionError("loaded model lacks shared-projection count")
        model.ck_model_batch_decode_projection_groups.restype = ctypes.c_int
        capabilities = int(model.ck_model_get_capabilities())
        projection_groups = _require_shared_projections(
            capabilities, int(model.ck_model_batch_decode_projection_groups()))
        kv_size = ctypes.c_size_t()
        kv_alignment = ctypes.c_size_t()
        if model.ck_model_sequence_state_requirements(
                ctypes.byref(kv_size), ctypes.byref(kv_alignment)) != 0 or not kv_size.value:
            raise AssertionError("loaded model lacks sequence-state extent")

        loaded = {}
        for name, symbol in (("libmodel.so", "ck_model_decode_batch2"),
                             ("libckernel_engine.so", "gemm_nt_q5_1_q8_1_m2")):
            identity = verified_loaded_symbol_backing(model, symbol, bundle / name)
            if identity["sha256"] != _digest(bundle / name):
                raise AssertionError(f"loaded {name} differs from its bundle file")
            loaded[name] = identity["sha256"]
        session_identity = verified_loaded_symbol_backing(
            session.lib, "ck_session_v8_batch2_mixed_step", SESSION_LIB_PATH)
        source = (bundle / "model_v8.c").read_bytes()
        marker = b'\n\nCK_EXPORT const char *ck_model_generated_source_sha256(void) {\n    return "'
        prefix, found, tail = source.rpartition(marker)
        source_sha256 = model.ck_model_generated_source_sha256()
        if (not found or not source_sha256 or
                tail != source_sha256 + b'";\n}\n' or
                hashlib.sha256(prefix).hexdigest().encode() != source_sha256):
            raise AssertionError("loaded generated source identity mismatch")

        def kv(slot: int) -> tuple[int, str]:
            position = batch.position(slot)  # Activates this sequence's KV arena.
            pointer = model.ck_model_get_named_activation_ptr(b"kv_cache")
            if not pointer:
                raise AssertionError("active KV arena is unavailable")
            data = ctypes.string_at(pointer, kv_size.value)
            return position, hashlib.sha256(data).hexdigest()

        def run(mixed: bool, prefill_slot: int, *, decode_prompt: tuple[int, ...],
                prompt: tuple[int, ...], decode_tokens: tuple[int, ...],
                continuation: tuple[int, int]) -> tuple[tuple, tuple, tuple]:
            decode_slot = prefill_slot ^ 1
            if len(decode_tokens) != len(prompt) - 1:
                raise AssertionError("fixture must pair every remaining prompt token")
            decode_ticket = batch.begin_prefill(decode_slot, len(decode_prompt))
            batch.append_prefill(decode_slot, decode_ticket, decode_prompt)
            prompt_ticket = batch.begin_prefill(prefill_slot, len(prompt))
            batch.append_prefill(prefill_slot, prompt_ticket, prompt[:1])
            tickets = [0, 0]
            tickets[decode_slot], tickets[prefill_slot] = decode_ticket, prompt_ticket
            steps = []
            for index, (decode_token, prompt_token) in enumerate(
                    zip(decode_tokens, prompt[1:], strict=True), start=2):
                tokens = [0, 0]
                tokens[decode_slot], tokens[prefill_slot] = decode_token, prompt_token
                if mixed:
                    mask, consumed, _ = batch.mixed_step(
                        tuple(tokens), tuple(tickets), prefill_slot)
                    if (mask, consumed) != (3, index):
                        raise AssertionError("mixed execution did not advance both rows")
                else:
                    decode_only = tokens.copy()
                    decode_only[prefill_slot] = 0
                    if batch.step(tuple(decode_only))[0] != 1 << decode_slot:
                        raise AssertionError("serialized decode advanced wrong slot")
                    if batch.append_prefill(prefill_slot, prompt_ticket, [prompt_token]) != index:
                        raise AssertionError("serialized prompt replay did not advance")
                steps.append(tuple(_finite_logit_bytes(batch.staged_logits(slot),
                                                       f"{'mixed' if mixed else 'reference'} step {index} slot {slot}")
                                   for slot in (0, 1)))
            continued = [0, 0]
            continued[decode_slot], continued[prefill_slot] = continuation
            if batch.step(tuple(continued))[0] != 3:
                raise AssertionError("post-prefill continuation did not advance both slots")
            steps.append(tuple(_finite_logit_bytes(batch.staged_logits(slot),
                                                   f"{'mixed' if mixed else 'reference'} continuation slot {slot}")
                               for slot in (0, 1)))
            result = (tuple(steps), kv(0), kv(1))
            batch.reset_slot(0)
            batch.reset_slot(1)
            return result

        cases = {}
        fixtures = {
            "unequal": ((100, 101, 102, 103, 104, 105, 106, 107, 108),
                        (200, 201, 202, 203, 204, 205, 206),
                        (109, 110, 111, 112, 113, 114), (115, 207)),
            "near_capacity": ((100,) * (context_length - 4),
                              (200, 201, 202, 203), (105, 106, 107), (108, 204)),
        }
        for name, (decode_prompt, prompt, decode_tokens, continuation) in fixtures.items():
            for prefill_slot in (0, 1):
                parameters = {"decode_prompt": decode_prompt, "prompt": prompt,
                              "decode_tokens": decode_tokens, "continuation": continuation}
                reference = run(False, prefill_slot, **parameters)
                actual = run(True, prefill_slot, **parameters)
                if actual != reference:
                    raise AssertionError(f"mixed/serialized full logits, complete KV, or positions differ in {name} slot {prefill_slot}")
                cases[f"{name}/prefill_slot_{prefill_slot}"] = {
                    "mixed_steps": len(decode_tokens),
                    "continuation_steps": 1,
                    "full_finite_logit_bytes_equal_at_every_step": True,
                    "complete_kv_sha256_equal": True,
                    "positions": [actual[1][0], actual[2][0]],
                }
        ordinary = None
        if compare_ordinary:
            def prefill_snapshot(slot: int, tokens: tuple[int, ...], *, staged: bool,
                                 chunk_size: int = 1) -> tuple[bytes, int, str]:
                try:
                    if staged:
                        ticket = batch.begin_prefill(slot, len(tokens))
                        for offset in range(0, len(tokens), chunk_size):
                            consumed = batch.append_prefill(
                                slot, ticket, tokens[offset:offset + chunk_size])
                            if consumed != min(len(tokens), offset + chunk_size):
                                raise AssertionError("staged prefill did not report exact progress")
                        logits = batch.staged_logits(slot)
                    else:
                        _, logits = batch.prefill(slot, tokens)
                    return (_finite_logit_bytes(logits, "staged" if staged else "ordinary"),
                            *kv(slot))
                finally:
                    batch.reset_slot(slot)

            ordinary_cases = {}
            repeatability = {}
            oracle_cases = {}
            for slot in (0, 1):
                for length in lengths:
                    tokens = tuple(100 + index for index in range(length))
                    reference = prefill_snapshot(slot, tokens, staged=False)
                    staged_single = None
                    for chunk_size in sorted({1, 4, length}):
                        staged = prefill_snapshot(slot, tokens, staged=True,
                                                  chunk_size=chunk_size)
                        if chunk_size == 1:
                            staged_single = staged
                        verdict = _compare_prefill_snapshots(reference, staged)
                        ordinary_cases[f"slot_{slot}/tokens_{length}/chunk_{chunk_size}"] = {
                            "token_ids": list(tokens), **verdict,
                        }
                    if length in oracle_rows:
                        assert staged_single is not None
                        oracle_cases[f"slot_{slot}/tokens_{length}"] = {
                            f"{subject}_vs_llama_{mode}": _logit_distance(
                                snapshot[0], oracle_rows[length][mode])
                            for subject, snapshot in (("ordinary", reference),
                                                      ("staged", staged_single))
                            for mode in ("batched", "sequential")
                        }
                    if length in {9, 10, context_length - 4}:
                        repeatability[f"slot_{slot}/tokens_{length}"] = {
                            "ordinary_exact": reference == prefill_snapshot(
                                slot, tokens, staged=False),
                            "staged_exact": prefill_snapshot(
                                slot, tokens, staged=True, chunk_size=1) ==
                                prefill_snapshot(slot, tokens, staged=True, chunk_size=1),
                        }
            failures = [key for key, value in ordinary_cases.items()
                        if value["status"] == "fail"]
            unstable = [key for key, value in repeatability.items()
                        if not all(value.values())]
            ordinary = {
                "status": "fail" if failures or unstable else "pass",
                "contract": "exact full-vocabulary finite logits, complete KV bytes, and position",
                "first_failure": failures[0] if failures else None,
                "repeatability": repeatability,
                "first_unstable": unstable[0] if unstable else None,
                "cases": ordinary_cases,
                "llama_oracle": {"identity": oracle_identity, "cases": oracle_cases}
                if oracle_identity else None,
            }
        return {
            "status": "fail" if ordinary and ordinary["status"] == "fail" else "pass",
            "scope": ("one decode row plus one staged prompt row; two slots; generated decode-replay parity"
                      + (" and separately reported ordinary-prefill comparison"
                         if compare_ordinary else " only")),
            "bundle": str(bundle),
            "loaded_sha256": loaded,
            "session_library_sha256": session_identity["sha256"],
            "generated_source_sha256": source_sha256.decode(),
            "weights_sha256": _digest(bundle / "weights.bump"),
            "capabilities": capabilities,
            "shared_projection_groups": projection_groups,
            "session_context_length": session.context_length,
            "generated_context_length": int(model.ck_model_get_context_window()),
            "threads": threads,
            "cpu": _cpu_identity(),
            "kv_bytes_per_sequence": kv_size.value,
            "mixed_decode_replay_status": "pass",
            "cases_by_prefill_slot": cases,
            "ordinary_prefill": ordinary,
        }
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--compare-ordinary", action="store_true",
                        help="Fail if staged and ordinary prefill differ at any tested length")
    parser.add_argument("--oracle-gguf", type=Path,
                        help="Add diagnostic logits from the pinned llama.cpp GGUF replay")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = certify(args.bundle, threads=args.threads,
                     context_length=args.context_length,
                     compare_ordinary=args.compare_ordinary,
                     oracle_gguf=args.oracle_gguf)
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
