#!/usr/bin/env python3
"""Check that one live batch slot cannot change another slot's logical KV.

This is an artifact-bound generated-runtime diagnostic. It tests both slot
directions with distinct prompts, cancellation, slot reuse and an isolated
full-logit/KV reference. It does not certify mixed prefill/decode arithmetic.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from server.serving_bundle import verified_loaded_symbol_backing
from server.session_v8 import SESSION_LIB_PATH, SessionV8
from version.v8.scripts.xray_batch_prefill_kv_v8 import (
    _attention_contract, _cpu_identity, _sha256_path, parse_kv_export,
)

PROMPTS = ((100, 101, 102), (110, 111))
NEXT_TOKENS = (103, 112)
FOLLOWUP_TOKENS = (104, 113)
REUSE_PROMPT = (120, 121, 122)


def assert_parked_unchanged(before: dict, after: dict) -> None:
    if before["position"] != after["position"] or before["kv"] != after["kv"]:
        raise AssertionError("non-advancing slot changed")


def assert_isolated_parity(shared: dict, isolated: dict) -> None:
    if (shared["position"] != isolated["position"] or
            shared["kv"] != isolated["kv"] or
            shared["logits_sha256"] != isolated["logits_sha256"]):
        raise AssertionError("advancing slot differs from isolated execution")


def assert_new_logical_rows(before: dict, after: dict) -> None:
    """A position-only header change must not count as KV progress."""
    if after["position"] != before["position"] + 1:
        raise AssertionError("advancing slot position did not progress")
    if not any(layer["last_row_nonzero"] for layer in after["kv"].values()):
        raise AssertionError("new logical KV row was not written")


def inject_parked_kv_write(batch, model, slot: int) -> None:
    """Corrupt parked KV storage; the logical export must expose the write."""
    if batch.position(slot) < 1:
        raise AssertionError("injected slot has no logical KV row")
    pointer = int(model.ck_model_get_named_activation_ptr(b"kv_cache"))
    if not pointer:
        raise AssertionError("injected slot has no KV arena")
    first_byte = ctypes.c_ubyte.from_address(pointer)
    first_byte.value ^= 1


def diagnose(bundle: Path, *, layers: tuple[int, ...] = (0, 16, 17),
             threads: int = 1, inject_parked_write: bool = False) -> dict:
    if not layers or len(set(layers)) != len(layers) or threads < 1:
        raise ValueError("invalid layer or thread selection")
    if inject_parked_write and 0 not in layers:
        raise ValueError("parked-write injection requires layer 0 in the KV export")
    bundle = bundle.resolve(strict=True)
    config = json.loads((bundle / "config.json").read_text())
    if any(layer < 0 or layer >= int(config["num_hidden_layers"]) for layer in layers):
        raise ValueError("requested layer is outside compiled graph")
    session = SessionV8.open(bundle, context_length=32, num_threads=threads)
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
        loaded_engine = verified_loaded_symbol_backing(
            engine, _attention_contract(bundle, layers[0])["prefill"]["function"],
            bundle / "libckernel_engine.so")
        loaded_session = verified_loaded_symbol_backing(
            session.lib, "ck_session_v8_batch2_step", SESSION_LIB_PATH)
        source = (bundle / "model_v8.c").read_bytes()
        marker = b'\n\nCK_EXPORT const char *ck_model_generated_source_sha256(void) {\n    return "'
        prefix, found, tail = source.rpartition(marker)
        source_sha = model.ck_model_generated_source_sha256()
        if (not found or not source_sha or tail != source_sha + b'";\n}\n' or
                hashlib.sha256(prefix).hexdigest().encode() != source_sha):
            raise AssertionError("loaded generated source identity mismatch")

        def snapshot(slot: int, logits: ctypes.Array | None = None) -> dict:
            position = batch.position(slot)  # Activates this slot's KV arena.
            pointer = int(model.ck_model_get_named_activation_ptr(b"kv_cache"))
            if not pointer:
                raise AssertionError("active KV arena is unavailable")
            with tempfile.TemporaryDirectory(prefix="cke_batch_slot_xray_") as directory:
                kv = {}
                for layer in layers:
                    path = Path(directory) / f"layer_{layer}.ckx"
                    if model.ck_model_debug_export_kv(str(path).encode(), layer):
                        raise RuntimeError(f"KV export failed at layer {layer}")
                    exported = parse_kv_export(
                        path.read_bytes(), layer=layer, position=position)
                    row_bytes = exported["row_bytes"]
                    payload = exported["payload"]
                    last_rows = [
                        payload[row * row_bytes:(row + 1) * row_bytes]
                        for kind in range(2) for head in range(exported["heads"])
                        for row in [((kind * exported["heads"] + head) * position
                                    + position - 1)]
                    ]
                    kv[str(layer)] = {
                        "ckx_sha256": exported["sha256"],
                        "logical_payload_sha256": hashlib.sha256(payload).hexdigest(),
                        "last_row_sha256": [hashlib.sha256(row).hexdigest()
                                            for row in last_rows],
                        "last_row_nonzero": any(any(row) for row in last_rows),
                    }
            result = {"position": position, "kv": kv, "arena_pointer": pointer}
            if logits is not None:
                if any(not math.isfinite(value) for value in logits):
                    raise AssertionError("nonfinite generated logits")
                result["logits_sha256"] = hashlib.sha256(bytes(logits)).hexdigest()
            return result

        cases = {}
        for advancing in (0, 1):
            parked = advancing ^ 1
            for slot in (0, 1):
                batch.reset_slot(slot)
            tickets = []
            for slot in (0, 1):
                ticket, _ = batch.prefill(slot, PROMPTS[slot])
                tickets.append(ticket)
            before = [snapshot(0), snapshot(1)]
            if before[0]["arena_pointer"] == before[1]["arena_pointer"]:
                raise AssertionError("live slots alias the same KV arena")
            batch.request_cancel(tickets[parked])
            mask, logits = batch.step(NEXT_TOKENS)
            if mask != 1 << advancing:
                raise AssertionError("wrong slot advanced after cancellation")
            if inject_parked_write and advancing == 0:
                inject_parked_kv_write(batch, model, parked)
            after = [snapshot(0), snapshot(1)]
            assert_parked_unchanged(before[parked], after[parked])
            assert_new_logical_rows(before[advancing], after[advancing])
            shared = snapshot(advancing, logits[advancing])

            batch.reset_slot(parked)
            replacement_ticket, replacement_logits = batch.prefill(parked, REUSE_PROMPT)
            after_reuse = snapshot(advancing)
            assert_parked_unchanged(after[advancing], after_reuse)
            replacement = snapshot(parked, replacement_logits)

            batch.request_cancel(replacement_ticket)
            followup_mask, followup_logits = batch.step(FOLLOWUP_TOKENS)
            if followup_mask != 1 << advancing:
                raise AssertionError("surviving slot failed to advance after reuse")
            continued = snapshot(advancing, followup_logits[advancing])
            assert_new_logical_rows(shared, continued)
            assert_parked_unchanged(replacement, snapshot(parked))

            for slot in (0, 1):
                batch.reset_slot(slot)
            _, clean_replacement_logits = batch.prefill(parked, REUSE_PROMPT)
            clean_replacement = snapshot(parked, clean_replacement_logits)
            assert_isolated_parity(replacement, clean_replacement)

            for slot in (0, 1):
                batch.reset_slot(slot)
            batch.prefill(advancing, PROMPTS[advancing])
            isolated_mask, isolated_logits = batch.step(NEXT_TOKENS)
            if isolated_mask != 1 << advancing:
                raise AssertionError("isolated execution advanced wrong slot")
            isolated = snapshot(advancing, isolated_logits[advancing])
            assert_isolated_parity(shared, isolated)
            isolated_followup_mask, isolated_followup_logits = batch.step(FOLLOWUP_TOKENS)
            if isolated_followup_mask != 1 << advancing:
                raise AssertionError("isolated follow-up advanced wrong slot")
            isolated_continued = snapshot(advancing, isolated_followup_logits[advancing])
            assert_isolated_parity(continued, isolated_continued)
            cases[str(advancing)] = {
                "advancing_slot": advancing, "parked_slot": parked,
                "shared_advanced_mask": mask,
                "followup_advanced_mask": followup_mask,
                "before": [{key: value for key, value in item.items()
                            if key != "arena_pointer"} for item in before],
                "after": [{key: value for key, value in item.items()
                           if key != "arena_pointer"} for item in after],
                "shared_logits_sha256": shared["logits_sha256"],
                "isolated_logits_sha256": isolated["logits_sha256"],
                "followup_shared_logits_sha256": continued["logits_sha256"],
                "followup_isolated_logits_sha256": isolated_continued["logits_sha256"],
                "parked_state_exact": True, "reuse_kept_advancing_state": True,
                "isolated_full_logit_and_kv_exact": True,
                "replacement_matches_clean_execution": True,
                "survivor_continuation_matches_isolated": True,
            }
        return {
            "schema": "cke.v8.batch-slot-isolation-xray.v1",
            "status": "pass", "scope": "two live KV-only slots; cancelled parked slot; slot reuse",
            "bundle": str(bundle), "layers": list(layers),
            "prompts": [list(tokens) for tokens in PROMPTS],
            "next_tokens": list(NEXT_TOKENS), "reuse_prompt": list(REUSE_PROMPT),
            "followup_tokens": list(FOLLOWUP_TOKENS),
            "requested_threads": threads,
            "cpu": _cpu_identity(),
            "loaded_model_sha256": loaded_model["sha256"],
            "loaded_engine_sha256": loaded_engine["sha256"],
            "loaded_session_sha256": loaded_session["sha256"],
            "generated_source_sha256": source_sha.decode(),
            "weights_sha256": _sha256_path(bundle / "weights.bump"),
            "diagnostic_source_sha256": _sha256_path(Path(__file__)),
            "diagnostic_git_head": subprocess.check_output(
                ("git", "rev-parse", "HEAD"),
                cwd=Path(__file__).resolve().parents[3], text=True).strip(),
            "diagnostic_source_uncommitted_at_capture": bool(subprocess.check_output(
                ("git", "status", "--porcelain", "--",
                 "version/v8/scripts/xray_batch_slot_isolation_v8.py"),
                cwd=Path(__file__).resolve().parents[3], text=True).strip()),
            "cases": cases,
        }
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--layers", default="0,16,17")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--inject-parked-write", action="store_true",
                        help="Negative control: corrupt parked KV and require isolation rejection")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = diagnose(args.bundle, layers=tuple(int(x) for x in args.layers.split(",")),
                      threads=args.threads,
                      inject_parked_write=args.inject_parked_write)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
