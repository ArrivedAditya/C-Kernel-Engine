#!/usr/bin/env python3
"""Compare the bounded native two-slot session against isolated Gemma decode.

This is a local artifact-specific check. It does not certify HTTP scheduling,
prefill cancellation, sampling, or a throughput improvement.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from server.serving_bundle import verified_loaded_symbol_backing


PROMPTS = ((100, 101), (200,))
NEXT_TOKENS = (102, 201)


class Config(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("model_library_path", ctypes.c_char_p), ("weights_path", ctypes.c_char_p),
        ("manifest_path", ctypes.c_char_p), ("context_length", ctypes.c_int32),
        ("num_threads", ctypes.c_int32), ("required_capabilities", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 8),
    ]


def bind_model(model):
    for name in ("ck_model_get_vocab_size", "ck_model_get_context_window",
                 "ck_model_get_active_tokens"):
        getattr(model, name).restype = ctypes.c_int
    model.ck_model_init.argtypes = [ctypes.c_char_p]
    model.ck_model_init.restype = ctypes.c_int
    model.ck_model_free.argtypes = []
    model.ck_model_kv_cache_reset.argtypes = []
    model.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    model.ck_model_embed_tokens.restype = ctypes.c_int
    model.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    model.ck_model_forward.restype = ctypes.c_int
    model.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    model.ck_model_decode.restype = ctypes.c_int
    model.ck_model_sequence_state_requirements.argtypes = [
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    model.ck_model_sequence_state_requirements.restype = ctypes.c_int
    model.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
    model.ck_model_get_named_activation_ptr.restype = ctypes.c_size_t


def bind_session(session):
    session.ck_session_v8_open.argtypes = [ctypes.POINTER(Config), ctypes.POINTER(ctypes.c_void_p)]
    session.ck_session_v8_open.restype = ctypes.c_int
    session.ck_session_v8_close.argtypes = [ctypes.c_void_p]
    session.ck_session_v8_batch2_enable.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    session.ck_session_v8_batch2_enable.restype = ctypes.c_int
    session.ck_session_v8_batch2_prefill.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint64)]
    session.ck_session_v8_batch2_prefill.restype = ctypes.c_int
    session.ck_session_v8_batch2_step.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.c_uint32)]
    session.ck_session_v8_batch2_step.restype = ctypes.c_int
    session.ck_session_v8_batch2_position.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)]
    session.ck_session_v8_batch2_position.restype = ctypes.c_int


def certify(bundle: Path, session_library: Path, max_extra_bytes: int) -> dict:
    model_path = bundle / "libmodel.so"
    weights_path = bundle / "weights.bump"
    manifest_path = bundle / "weights_manifest.map"
    for path in (model_path, weights_path, manifest_path, session_library):
        if not path.is_file():
            raise FileNotFoundError(path)
    model = ctypes.CDLL(str(model_path), mode=ctypes.RTLD_GLOBAL)
    bind_model(model)
    model_identity = verified_loaded_symbol_backing(model, "ck_model_decode_batch2", model_path)
    engine_identity = verified_loaded_symbol_backing(
        model, "gemm_nt_q5_1_q8_1_m2", bundle / "libckernel_engine.so")
    tokenizer_identity = verified_loaded_symbol_backing(
        model, "ck_tokenizer_encode", bundle / "libckernel_tokenizer.so")

    def file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    kv_bytes, kv_alignment = ctypes.c_size_t(), ctypes.c_size_t()
    if model.ck_model_sequence_state_requirements(
            ctypes.byref(kv_bytes), ctypes.byref(kv_alignment)) != 0:
        raise RuntimeError("KV requirements unavailable")
    vocab = model.ck_model_get_vocab_size()
    context = model.ck_model_get_context_window()
    if vocab <= 0 or context < max(map(len, PROMPTS)) + 1:
        raise RuntimeError("artifact is incompatible with the bounded fixture")

    def kv_hash():
        address = model.ck_model_get_named_activation_ptr(b"kv_cache")
        if not address:
            raise RuntimeError("active KV pointer unavailable")
        return hashlib.sha256(ctypes.string_at(address, kv_bytes.value)).hexdigest()

    if model.ck_model_init(str(weights_path).encode()) != 0:
        raise RuntimeError("isolated model init failed")
    isolated = []
    try:
        for prompt, token in zip(PROMPTS, NEXT_TOKENS):
            model.ck_model_kv_cache_reset()
            values = (ctypes.c_int32 * len(prompt))(*prompt)
            output = (ctypes.c_float * vocab)()
            if model.ck_model_embed_tokens(values, len(prompt)) != 0 or \
                    model.ck_model_forward(output) != 0 or \
                    model.ck_model_decode(token, output) != 0:
                raise RuntimeError("isolated prompt/decode failed")
            isolated.append((bytes(output), kv_hash(), model.ck_model_get_active_tokens()))
    finally:
        model.ck_model_free()

    session = ctypes.CDLL(str(session_library))
    bind_session(session)
    session_identity = verified_loaded_symbol_backing(
        session, "ck_session_v8_batch2_step", session_library)
    config = Config(ctypes.sizeof(Config), 1, str(model_path).encode(),
                    str(weights_path).encode(), str(manifest_path).encode(),
                    context, 2, 0, (ctypes.c_uint64 * 8)())
    handle = ctypes.c_void_p()
    if session.ck_session_v8_open(ctypes.byref(config), ctypes.byref(handle)) != 0:
        raise RuntimeError("native session open failed")
    try:
        needed = ctypes.c_size_t()
        status = session.ck_session_v8_batch2_enable(
            handle, max_extra_bytes, ctypes.byref(needed))
        if status != 0:
            raise RuntimeError(f"batch admission failed: {status}; needs {needed.value} bytes")
        outputs = [(ctypes.c_float * vocab)() for _ in range(2)]
        for slot, prompt in enumerate(PROMPTS):
            values = (ctypes.c_int32 * len(prompt))(*prompt)
            ticket = ctypes.c_uint64()
            status = session.ck_session_v8_batch2_prefill(
                handle, slot, values, len(prompt), outputs[slot], ctypes.byref(ticket))
            if status != 0 or not ticket.value:
                raise RuntimeError(f"slot {slot} prefill failed: {status}")
        output_ptrs = (ctypes.POINTER(ctypes.c_float) * 2)(*outputs)
        mask = ctypes.c_uint32()
        status = session.ck_session_v8_batch2_step(
            handle, (ctypes.c_int32 * 2)(*NEXT_TOKENS), output_ptrs,
            ctypes.byref(mask))
        if status != 0 or mask.value != 3:
            raise RuntimeError(f"two-slot step failed: {status}, mask={mask.value}")
        comparisons = []
        for slot in range(2):
            position = ctypes.c_int32()
            if session.ck_session_v8_batch2_position(
                    handle, slot, ctypes.byref(position)) != 0:
                raise RuntimeError("position query failed")
            exact_logits = bytes(outputs[slot]) == isolated[slot][0]
            exact_kv = kv_hash() == isolated[slot][1]
            exact_position = position.value == isolated[slot][2]
            comparisons.append({"slot": slot, "exact_logits": exact_logits,
                                "exact_kv": exact_kv, "exact_position": exact_position,
                                "position": position.value})
        if not all(all(value for key, value in row.items() if key.startswith("exact_"))
                   for row in comparisons):
            raise AssertionError("native two-slot output/state differs from isolated execution")
        return {"status": "pass", "model": model_identity,
                "engine": engine_identity, "tokenizer": tokenizer_identity,
                "session": session_identity,
                "weights_sha256": file_hash(weights_path),
                "manifest_sha256": file_hash(manifest_path),
                "required_extra_bytes": needed.value,
                "comparisons": comparisons}
    finally:
        session.ck_session_v8_close(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--session-library", type=Path, required=True)
    parser.add_argument("--max-extra-bytes", type=int, default=1 << 30)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = certify(args.bundle.resolve(), args.session_library.resolve(),
                     args.max_extra_bytes)
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
