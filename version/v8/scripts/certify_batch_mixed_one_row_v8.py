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


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify_oracle_mappings(meta: dict, lib_dir: Path) -> dict[str, str]:
    """Match the helper's live /proc/self/maps identities to the intended files."""
    mappings = meta.get("loaded_libraries")
    names = ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so")
    if not isinstance(mappings, dict) or set(mappings) != set(names):
        raise AssertionError("oracle did not report its complete loaded-library set")
    hashes = {}
    for name in names:
        item = mappings[name]
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise AssertionError(f"malformed loaded oracle mapping: {name}")
        path = Path(item["path"])
        try:
            expected = (lib_dir / name).resolve(strict=True)
            if path.resolve(strict=True) != expected or " (deleted)" in item["path"]:
                raise AssertionError(f"alternate or deleted loaded oracle library: {name}")
            with path.open("rb") as backing:
                stat = os.fstat(backing.fileno())
                device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}"
                if item.get("device", "").lower() != device or item.get("inode") != stat.st_ino:
                    raise AssertionError(f"stale loaded oracle library backing: {name}")
                digest = hashlib.sha256()
                for part in iter(lambda: backing.read(1024 * 1024), b""):
                    digest.update(part)
                current = path.stat()
                if current.st_dev != stat.st_dev or current.st_ino != stat.st_ino:
                    raise AssertionError(f"loaded oracle library replaced while hashing: {name}")
                hashes[name] = digest.hexdigest()
        except OSError as exc:
            raise AssertionError(f"cannot inspect loaded oracle library: {name}") from exc
    return hashes


def _validated_oracle_result(result: dict, tokens: list[int], mode: str,
                             lib_dir: Path) -> tuple[bytes, dict[str, str], dict]:
    if not isinstance(result, dict) or not isinstance(result.get("meta"), dict):
        raise AssertionError("llama.cpp returned malformed metadata")
    meta = result["meta"]
    logits = result.get("logits")
    if (logits is None or int(meta.get("n_vocab", -1)) != len(logits) or
            int(meta.get("token_count", -1)) != len(tokens) or
            meta.get("tokens") != tokens or meta.get("decode_mode") != mode):
        raise AssertionError("llama.cpp returned wrong tokens, mode, or vocabulary extent")
    settings = meta.get("effective_settings")
    if not isinstance(settings, dict) or not settings:
        raise AssertionError("llama.cpp omitted effective execution settings")
    row = logits.tobytes()
    _logit_distance(row, row)  # Reject nonfinite or empty output.
    return row, _verify_oracle_mappings(meta, lib_dir), settings


def _verify_oracle_checkout(root: Path, llama_cpp: Path) -> str:
    expected_pin = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "HEAD", "llama.cpp"], text=True).split()[2]
    actual_pin = subprocess.check_output(
        ["git", "-C", str(llama_cpp), "rev-parse", "HEAD"], text=True).strip()
    if actual_pin != expected_pin:
        raise AssertionError("llama.cpp checkout differs from the repository pin")
    if subprocess.check_output(
            ["git", "-C", str(llama_cpp), "status", "--porcelain",
             "--untracked-files=no"], text=True).strip():
        raise AssertionError("llama.cpp checkout has tracked source changes")
    return actual_pin


def _sample_gguf_bump_pair(gguf: Path, bundle: Path) -> dict:
    """Check selected source tensor bytes and graph metadata, not a full conversion."""
    from version.v8.scripts.convert_gguf_to_bump_v8 import (
        GGUFReader, _gguf_read_value, _gguf_skip_value, align_up,
    )
    wanted = {
        "general.architecture", "gemma3.block_count", "gemma3.embedding_length",
        "gemma3.feed_forward_length", "gemma3.attention.head_count",
        "gemma3.attention.head_count_kv", "gemma3.context_length",
        "gemma3.rope.freq_base", "gemma3.attention.sliding_window",
        "general.alignment",
    }
    pairs = {
        "token_embd.weight": "token_emb",
        "blk.0.attn_q.weight": "layer.0.wq",
        "blk.0.attn_k.weight": "layer.0.wk",
        "blk.0.attn_v.weight": "layer.0.wv",
        "blk.0.ffn_down.weight": "layer.0.w2",
        "blk.0.attn_norm.weight": "layer.0.ln1_gamma",
        "output_norm.weight": "final_ln_weight",
    }
    manifest = json.loads((bundle / "weights_manifest.json").read_text())
    entries = {item["name"]: item for item in manifest["entries"]}
    config = json.loads((bundle / "config.json").read_text())
    with gguf.open("rb") as source, (bundle / "weights.bump").open("rb") as converted:
        reader = GGUFReader(source)
        if reader._read_exact(4) != b"GGUF":
            raise AssertionError("source is not GGUF")
        version = reader.u32()
        tensor_count, metadata_count = ((reader.u64(), reader.u64()) if version >= 2
                                        else (reader.u32(), reader.u32()))
        metadata = {}
        for _ in range(metadata_count):
            key, kind = reader.key_str(), reader.u32()
            if key in wanted:
                metadata[key] = _gguf_read_value(reader, kind)
            else:
                _gguf_skip_value(reader, kind)
        tensors = {}
        for _ in range(tensor_count):
            name = reader.key_str()
            dimensions = [reader.u64() for _ in range(reader.u32())]
            quant_type, offset = reader.u32(), reader.u64()
            tensors[name] = {"dimensions": dimensions, "quant_type": quant_type,
                             "offset": offset}
        data_start = align_up(reader.tell(), int(metadata.get("general.alignment", 32)))
        expected = {
            "general.architecture": "gemma3",
            "gemma3.block_count": config["num_hidden_layers"],
            "gemma3.embedding_length": config["hidden_size"],
            "gemma3.feed_forward_length": config["intermediate_size"],
            "gemma3.attention.head_count": config["num_attention_heads"],
            "gemma3.attention.head_count_kv": config["num_key_value_heads"],
            "gemma3.context_length": config["max_position_embeddings"],
            "gemma3.rope.freq_base": config["rope_theta"],
            "gemma3.attention.sliding_window": config["sliding_window"],
        }
        metadata_matches = {key: metadata.get(key) == value for key, value in expected.items()}
        samples = {}
        for source_name, converted_name in pairs.items():
            tensor, entry = tensors[source_name], entries[converted_name]
            size = int(entry["size"])
            if size < 64:
                raise AssertionError(f"sample tensor is too short: {source_name}")
            offsets = sorted({0, (size // 2) // 64 * 64, size - 64})
            matches = []
            for offset in offsets:
                source.seek(data_start + tensor["offset"] + offset)
                converted.seek(int(entry["file_offset"]) + offset)
                source_bytes, converted_bytes = source.read(64), converted.read(64)
                matches.append(len(source_bytes) == len(converted_bytes) == 64 and
                               source_bytes == converted_bytes)
            samples[source_name] = {
                "bump_name": converted_name, "gguf_type": tensor["quant_type"],
                "dimensions": tensor["dimensions"], "sample_offsets": offsets,
                "raw_64_byte_samples_match": matches,
            }
    return {
        "config_metadata_matches": metadata_matches,
        "sampled_tensor_bytes_match": all(all(item["raw_64_byte_samples_match"])
                                          for item in samples.values()),
        "samples": samples,
        "converter_source_candidate_sha256": _digest(
            Path(__file__).with_name("convert_gguf_to_bump_v8.py")),
        "manifest_sha256": _digest(bundle / "weights_manifest.json"),
        "converter_invocation_verified": False,
        "full_tensor_inventory_verified": False,
    }


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
            HELPER_SRC, LLAMA_CPP, ensure_llama_helper, run_llama_logits,
        )
        oracle_gguf = oracle_gguf.resolve(strict=True)
        actual_pin = _verify_oracle_checkout(
            Path(__file__).resolve().parents[3], LLAMA_CPP)
        lib_dir = LLAMA_CPP / "build" / "bin"
        libraries = ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so")
        helper = ensure_llama_helper()
        library_hashes = {name: _digest(lib_dir / name) for name in libraries}
        oracle_identity = {
            "gguf_sha256": _digest(oracle_gguf),
            "llama_commit": actual_pin,
            "submodule_tracked_dirty": False,
            "helper_source_sha256": _digest(HELPER_SRC),
            "helper_binary_sha256": _digest(helper),
            "library_file_sha256": library_hashes,
            "loaded_library_sha256": None,
            "library_build_provenance_verified": False,
            "source_pair": _sample_gguf_bump_pair(oracle_gguf, bundle),
            "source_pair_verified": False,
            "status": "diagnostic_only",
        }
        oracle_self = {}
        oracle_settings = {}
        reference_mappings = None
        for length in sorted({1, 9, 10, context_length - 4}):
            tokens = list(range(100, 100 + length))
            oracle_rows[length] = {}
            repeatability = {}
            settings = {}
            for mode in ("batched", "sequential"):
                runs = []
                for _ in range(2):
                    result = run_llama_logits(oracle_gguf, tokens,
                                              max(64, context_length), 16, threads,
                                              decode_mode=mode)
                    rows, mappings, effective = _validated_oracle_result(
                        result, tokens, mode, lib_dir)
                    meta = result["meta"]
                    if mappings != library_hashes:
                        raise AssertionError("loaded oracle library bytes differ from candidate files")
                    if reference_mappings is not None and meta["loaded_libraries"] != reference_mappings:
                        raise AssertionError("loaded oracle library mappings changed between replays")
                    reference_mappings = meta["loaded_libraries"]
                    runs.append(rows)
                    if mode in settings and settings[mode] != effective:
                        raise AssertionError("effective llama settings changed between replays")
                    settings[mode] = effective
                oracle_rows[length][mode] = runs[0]
                repeatability[mode] = {
                    "exact": runs[0] == runs[1],
                    "logits_sha256": [_digest_bytes(row) for row in runs],
                }
            oracle_self[f"tokens_{length}"] = {
                "batched_vs_sequential": _logit_distance(
                    oracle_rows[length]["batched"], oracle_rows[length]["sequential"]),
                "repeatability": repeatability,
            }
            oracle_settings[f"tokens_{length}"] = settings
        one_token_sensitivity = {}
        for sensitivity_threads in sorted({1, threads}):
            for no_repack in (False, True):
                result = run_llama_logits(
                    oracle_gguf, [100], max(64, context_length), 16,
                    sensitivity_threads, decode_mode="batched", no_repack=no_repack)
                row, mappings, effective = _validated_oracle_result(
                    result, [100], "batched", lib_dir)
                if mappings != library_hashes or result["meta"]["loaded_libraries"] != reference_mappings:
                    raise AssertionError("loaded oracle identity changed during one-token sensitivity check")
                one_token_sensitivity[f"threads_{sensitivity_threads}/no_repack_{no_repack}"] = {
                    "vs_baseline": _logit_distance(row, oracle_rows[1]["batched"]),
                    "effective_settings": effective,
                }
        if _digest(helper) != oracle_identity["helper_binary_sha256"]:
            raise AssertionError("oracle helper binary changed during replay")
        oracle_identity["loaded_library_sha256"] = library_hashes
        oracle_identity["loaded_library_mappings"] = reference_mappings
        oracle_identity["effective_settings_by_case"] = oracle_settings
        oracle_identity["reference_self_comparison"] = oracle_self
        oracle_identity["one_token_sensitivity"] = one_token_sensitivity
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
                "llama_oracle": {
                    "identity": oracle_identity,
                    "summary": {
                        "earliest_cke_ordinary_staged_failure": failures[0] if failures else None,
                        "one_token_cke_oracle_top1_equal": (
                            oracle_cases["slot_0/tokens_1"]["ordinary_vs_llama_batched"]
                            ["top1_subject"] ==
                            oracle_cases["slot_0/tokens_1"]["ordinary_vs_llama_batched"]
                            ["top1_oracle"]),
                        "llama_batched_sequential_top1_disagreement_lengths": [
                            length for length in sorted(oracle_rows)
                            if oracle_identity["reference_self_comparison"]
                            [f"tokens_{length}"]["batched_vs_sequential"]
                            ["top1_subject"] != oracle_identity["reference_self_comparison"]
                            [f"tokens_{length}"]["batched_vs_sequential"]
                            ["top1_oracle"]],
                        "all_oracle_replays_exact": all(
                            item["exact"] for row in oracle_identity["reference_self_comparison"]
                            .values() for item in row["repeatability"].values()),
                        "source_pair_verified": False,
                    },
                    "cases": oracle_cases,
                }
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
