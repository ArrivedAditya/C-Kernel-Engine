#!/usr/bin/env python3
"""Compare generated ordinary and staged prefill KV at logical coordinates.

This is a diagnostic, not a numerical certification or a replacement for the
mixed-step acceptance test. It reuses the generated CKX export and never reads
the physical KV arena as if its padding were logical sequence state.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from server.serving_bundle import verified_loaded_symbol_backing
from server.session_v8 import SESSION_LIB_PATH, SessionV8

_HEADER = struct.Struct("<8I")
_MAGIC = 0x564B5843
_ELEMENT_BYTES = {0: 4, 1: 2, 2: 2}  # FP32, FP16, BF16.


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def parse_kv_export(data: bytes, *, layer: int, position: int) -> dict:
    """Validate CKX v2 and retain only valid [kind, head, position, channel] rows."""
    if len(data) < _HEADER.size:
        raise ValueError("truncated KV export header")
    magic, version, got_layer, valid, heads, capacity, width, dtype = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != 2:
        raise ValueError("unsupported KV export format")
    if got_layer != layer or valid != position or not (0 <= valid <= capacity):
        raise ValueError("KV export layer or position mismatch")
    if heads < 1 or width < 1 or dtype not in _ELEMENT_BYTES:
        raise ValueError("invalid KV export geometry or dtype")
    row_bytes = width * _ELEMENT_BYTES[dtype]
    expected = _HEADER.size + 2 * heads * valid * row_bytes
    if len(data) != expected:
        raise ValueError("KV export payload extent mismatch")
    return {
        "layer": layer, "position": valid, "heads": heads, "capacity": capacity,
        "head_width": width, "dtype": dtype, "row_bytes": row_bytes,
        "sha256": hashlib.sha256(data).hexdigest(), "payload": data[_HEADER.size:],
    }


def compare_kv_exports(ordinary: dict, staged: dict, *, ordinary_owner: int,
                       staged_owner: int) -> dict:
    """Report the first differing logical cache element, without a tolerance."""
    if ordinary_owner != staged_owner:
        raise ValueError("KV sequence owner mismatch")
    geometry = ("layer", "position", "heads", "capacity", "head_width", "dtype", "row_bytes")
    if any(ordinary[key] != staged[key] for key in geometry):
        raise ValueError("KV export geometry mismatch")
    left, right = ordinary["payload"], staged["payload"]
    if left == right:
        return {"exact": True, "first_difference": None}
    element_bytes = _ELEMENT_BYTES[ordinary["dtype"]]
    row_bytes = ordinary["row_bytes"]
    first_byte = next(index for index, pair in enumerate(zip(left, right, strict=True))
                      if pair[0] != pair[1])
    linear_row, offset = divmod(first_byte, row_bytes)
    kind_index, local_row = divmod(linear_row, ordinary["heads"] * ordinary["position"])
    head, token_position = divmod(local_row, ordinary["position"])
    element_offset = linear_row * row_bytes + (offset // element_bytes) * element_bytes
    raw_ordinary = left[element_offset:element_offset + element_bytes]
    raw_staged = right[element_offset:element_offset + element_bytes]

    def decode(raw: bytes) -> dict:
        dtype = ordinary["dtype"]
        value = (struct.unpack("<f", raw)[0] if dtype == 0 else
                 struct.unpack("<e", raw)[0] if dtype == 1 else
                 struct.unpack("<f", b"\0\0" + raw)[0])
        if math.isnan(value):
            classification = "nan"
        elif math.isinf(value):
            classification = "positive_infinity" if value > 0 else "negative_infinity"
        else:
            classification = "finite"
        return {"raw_le_hex": raw.hex(), "value": value if math.isfinite(value) else None,
                "classification": classification}

    left_value = decode(raw_ordinary)
    right_value = decode(raw_staged)
    return {
        "exact": False,
        "first_difference": {
            "kind": ("K", "V")[kind_index], "head": head,
            "position": token_position, "channel": offset // element_bytes,
            "byte_in_element": offset % element_bytes,
            "ordinary": left_value, "staged": right_value,
            "absolute_difference": abs(left_value["value"] - right_value["value"])
            if left_value["value"] is not None and right_value["value"] is not None else None,
        },
    }


def retain_prefix_result(cases: dict, *, slot: int, length: int, max_prefix: int,
                         by_layer: dict, previous: tuple[str, dict] | None) -> tuple[str, dict]:
    """Retain each failure and its true immediately preceding control."""
    key = f"slot_{slot}/prefix_{length}"
    if any(not item["exact"] for item in by_layer.values()):
        if previous is not None:
            cases.setdefault(previous[0], previous[1])
        cases[key] = by_layer
    elif length >= max_prefix - 1:
        cases[key] = by_layer
    return key, by_layer


def attention_span(*, position: int, sliding_window: int,
                   observed_first: int | None = None,
                   observed_last: int | None = None) -> dict:
    """Validate a supplied read span against the declared causal/window bounds."""
    if position < 0 or sliding_window < 0:
        raise ValueError("invalid attention position or window")
    first = max(0, position - sliding_window + 1) if sliding_window else 0
    if (observed_first is None) != (observed_last is None):
        raise ValueError("incomplete observed attention span")
    if observed_first is not None and (observed_first, observed_last) != (first, position):
        raise ValueError("attention read violates causal/window boundary")
    return {"first": first, "last": position,
            "observed": observed_first is not None}


def _attention_contract(bundle: Path, layer: int) -> dict:
    contracts = {}
    for phase in ("prefill", "decode"):
        operations = json.loads((bundle / f"lowered_{phase}.json").read_text())["operations"]
        candidates = [op for op in operations if op.get("layer") == layer and
                      op.get("op") in {"attn", "attn_sliding"}]
        if len(candidates) != 1:
            raise ValueError(f"expected one attention operation for layer {layer} in {phase}")
        op = candidates[0]
        contracts[phase] = {
            "function": op["function"],
            "sliding_window": int(op["params"].get("sliding_window") or 0),
            "reduction": op["resolved_contract"]["contract_id"],
        }
    if contracts["prefill"]["sliding_window"] != contracts["decode"]["sliding_window"]:
        raise ValueError("prefill/decode attention windows disagree")
    return contracts


def diagnose(bundle: Path, *, max_prefix: int = 10, layers: tuple[int, ...] = (16,),
             slots: tuple[int, ...] = (0, 1), threads: int = 1) -> dict:
    if (max_prefix < 1 or not layers or not slots or threads < 1 or
            len(set(layers)) != len(layers) or len(set(slots)) != len(slots) or
            any(slot not in (0, 1) for slot in slots)):
        raise ValueError("invalid diagnostic bounds")
    bundle = bundle.resolve(strict=True)
    config = json.loads((bundle / "config.json").read_text())
    layer_count = int(config["num_hidden_layers"])
    if any(layer < 0 or layer >= layer_count for layer in layers):
        raise ValueError("requested KV layer is outside the compiled graph")
    contracts = {str(layer): _attention_contract(bundle, layer) for layer in layers}
    session = SessionV8.open(bundle, context_length=max(32, max_prefix), num_threads=threads)
    try:
        batch = session.enable_batch2(1 << 30)
        model = ctypes.CDLL(str(bundle / "libmodel.so"))
        engine = ctypes.CDLL(str(bundle / "libckernel_engine.so"))
        model.ck_model_debug_export_kv.argtypes = [ctypes.c_char_p, ctypes.c_int]
        model.ck_model_debug_export_kv.restype = ctypes.c_int
        model.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
        model.ck_model_get_named_activation_ptr.restype = ctypes.c_size_t
        model.ck_model_generated_source_sha256.restype = ctypes.c_char_p
        loaded_model = verified_loaded_symbol_backing(
            model, "ck_model_debug_export_kv", bundle / "libmodel.so")
        loaded_session = verified_loaded_symbol_backing(
            session.lib, "ck_session_v8_batch2_mixed_step", SESSION_LIB_PATH)
        selected_attention = contracts[str(layers[0])]["prefill"]["function"]
        loaded_engine = verified_loaded_symbol_backing(
            engine, selected_attention, bundle / "libckernel_engine.so")
        source = (bundle / "model_v8.c").read_bytes()
        marker = b'\n\nCK_EXPORT const char *ck_model_generated_source_sha256(void) {\n    return "'
        prefix, found, tail = source.rpartition(marker)
        source_sha256 = model.ck_model_generated_source_sha256()
        if (not found or not source_sha256 or
                tail != source_sha256 + b'";\n}\n' or
                hashlib.sha256(prefix).hexdigest().encode() != source_sha256):
            raise AssertionError("loaded generated source identity mismatch")
        owner_pointers: dict[int, int] = {}

        def snapshot(slot: int, tokens: tuple[int, ...], staged: bool) -> dict[int, dict]:
            if staged:
                ticket = batch.begin_prefill(slot, len(tokens))
                for index, token in enumerate(tokens, 1):
                    if batch.append_prefill(slot, ticket, (token,)) != index:
                        raise AssertionError("staged prefill made incorrect progress")
            else:
                batch.prefill(slot, tokens)
            try:
                if batch.position(slot) != len(tokens):
                    raise AssertionError("prefill position mismatch")
                pointer = int(model.ck_model_get_named_activation_ptr(b"kv_cache"))
                if not pointer:
                    raise AssertionError("active sequence has no KV arena")
                previous = owner_pointers.setdefault(slot, pointer)
                if pointer != previous or pointer in (value for key, value in owner_pointers.items()
                                                    if key != slot):
                    raise AssertionError("KV arena owner changed or aliases another slot")
                with tempfile.TemporaryDirectory(prefix="cke_kv_xray_") as directory:
                    result = {}
                    for layer in layers:
                        path = Path(directory) / f"layer_{layer}.ckx"
                        rc = model.ck_model_debug_export_kv(str(path).encode(), layer)
                        if rc:
                            raise RuntimeError(f"KV export failed at layer {layer}: {rc}")
                        result[layer] = parse_kv_export(path.read_bytes(), layer=layer,
                                                        position=len(tokens))
                    return result
            finally:
                batch.reset_slot(slot)

        cases = {}
        first_failure = None
        exact_layer_checks = 0
        failure_count = 0
        for slot in slots:
            previous_case = None
            for length in range(1, max_prefix + 1):
                tokens = tuple(range(100, 100 + length))
                ordinary = snapshot(slot, tokens, False)
                staged = snapshot(slot, tokens, True)
                by_layer = {}
                for layer in layers:
                    result = compare_kv_exports(ordinary[layer], staged[layer],
                                                ordinary_owner=slot, staged_owner=slot)
                    key = f"slot_{slot}/prefix_{length}/layer_{layer}"
                    exact_layer_checks += int(result["exact"])
                    failure_count += int(not result["exact"])
                    by_layer[str(layer)] = {
                        **result,
                        "ordinary_sha256": ordinary[layer]["sha256"],
                        "staged_sha256": staged[layer]["sha256"],
                        "valid_position": length,
                        "storage_dtype": ordinary[layer]["dtype"],
                        "kv_heads": ordinary[layer]["heads"],
                        "head_width": ordinary[layer]["head_width"],
                        "attention_read_span": attention_span(
                            position=length - 1,
                            sliding_window=contracts[str(layer)]["prefill"]["sliding_window"]),
                    }
                    if not result["exact"] and first_failure is None:
                        first_failure = key
                previous_case = retain_prefix_result(
                    cases, slot=slot, length=length, max_prefix=max_prefix,
                    by_layer=by_layer, previous=previous_case)
        return {
            "schema": "cke.v8.batch-prefill-kv-xray.v1",
            "status": "fail" if first_failure else "pass",
            "scope": "completed-prefix logical KV writes; attention read span is declared, not observed",
            "full_logit_parity": "not_tested_by_this_probe",
            "first_failure": first_failure,
            "checked_prefixes_by_slot": {
                str(slot): list(range(1, max_prefix + 1)) for slot in slots},
            "exact_layer_checks": exact_layer_checks,
            "failing_layer_checks": failure_count,
            "bundle": str(bundle),
            "loaded_model_sha256": loaded_model["sha256"],
            "loaded_session_sha256": loaded_session["sha256"],
            "loaded_engine_sha256": loaded_engine["sha256"],
            "generated_source_sha256": source_sha256.decode(),
            "diagnostic_source_sha256": _sha256_path(Path(__file__)),
            "diagnostic_git_head": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=Path(__file__).resolve().parents[3],
                text=True).strip(),
            "diagnostic_source_uncommitted_at_capture": subprocess.run(
                ("git", "diff", "--quiet", "HEAD", "--",
                 "version/v8/scripts/xray_batch_prefill_kv_v8.py"),
                cwd=Path(__file__).resolve().parents[3], check=False).returncode != 0,
            "requested_threads": threads,
            "cpu": platform.processor() or platform.machine(),
            "declared_bundle_assets_sha256": {
                name: _sha256_path(bundle / name)
                for name in ("config.json", "lowered_prefill.json", "lowered_decode.json")},
            "lowering_to_loaded_binary_relationship": "unverified",
            "weights_sha256": _sha256_path(bundle / "weights.bump"),
            "tokens": list(range(100, 100 + max_prefix)),
            "attention_contracts": contracts,
            "cases": cases,
        }
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--max-prefix", type=int, default=10)
    parser.add_argument("--layers", default="16", help="comma-separated decoder layer IDs")
    parser.add_argument("--slots", default="0,1", help="comma-separated batch slot IDs")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = diagnose(args.bundle, max_prefix=args.max_prefix,
                      layers=tuple(int(x) for x in args.layers.split(",")),
                      slots=tuple(int(x) for x in args.slots.split(",")),
                      threads=args.threads)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
