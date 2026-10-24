#!/usr/bin/env python3
"""Certify opt-in phase-aware mixed decode/prefill against isolated execution.

This compares one generated two-row step at each prompt prefix with ordinary
prefill of that prefix, and the survivor with isolated generated decode.
It does not certify HTTP scheduling, general chunk sizes, or other models.
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
from server.session_v8 import SESSION_LIB_PATH, SessionError, SessionV8
from certify_batch_mixed_one_row_v8 import _cpu_identity, _digest, _finite_logit_bytes


def certify(bundle: Path, *, context_length: int, threads: int) -> dict:
    session = SessionV8.open(bundle, context_length=context_length,
                             num_threads=threads)
    try:
        batch = session.enable_batch2(1 << 30)
        if session.batch2_phase_capabilities() != 3:
            raise AssertionError("loaded model lacks both phase-aware staged entries")
        model = ctypes.CDLL(str(bundle / "libmodel.so"))
        required_symbol = "ck_model_decode_batch2_mixed_prefill"
        if not hasattr(model, required_symbol):
            raise AssertionError("loaded model has no phase-aware mixed entry")
        model.ck_model_batch_decode_projection_groups.restype = ctypes.c_int
        model.ck_model_batch_decode_shared_layers.restype = ctypes.c_int
        model.ck_model_sequence_state_requirements.argtypes = [
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        model.ck_model_sequence_state_requirements.restype = ctypes.c_int
        model.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
        model.ck_model_get_named_activation_ptr.restype = ctypes.c_size_t
        model.ck_model_generated_source_sha256.restype = ctypes.c_char_p
        groups = model.ck_model_batch_decode_projection_groups()
        layers = model.ck_model_batch_decode_shared_layers()
        if groups < 2 or layers < 1:
            raise AssertionError("loaded model did not share generated projections")
        kv_bytes = ctypes.c_size_t()
        alignment = ctypes.c_size_t()
        if model.ck_model_sequence_state_requirements(
                ctypes.byref(kv_bytes), ctypes.byref(alignment)) or not kv_bytes.value:
            raise AssertionError("loaded model has no KV state extent")

        loaded = {}
        for name, symbol in (("libmodel.so", required_symbol),
                             ("libckernel_engine.so", "gemm_nt_q5_1_q8_1_m2")):
            identity = verified_loaded_symbol_backing(model, symbol,
                                                       bundle / name)
            if identity["sha256"] != _digest(bundle / name):
                raise AssertionError(f"loaded {name} differs from bundle")
            loaded[name] = identity["sha256"]
        native = verified_loaded_symbol_backing(
            session.lib, "ck_session_v8_batch2_mixed_step_prefill_phase",
            SESSION_LIB_PATH)
        source = (bundle / "model_v8.c").read_bytes()
        marker = b'\n\nCK_EXPORT const char *ck_model_generated_source_sha256(void) {\n    return "'
        prefix, found, tail = source.rpartition(marker)
        embedded = model.ck_model_generated_source_sha256()
        if (not found or not embedded or tail != embedded + b'";\n}\n' or
                hashlib.sha256(prefix).hexdigest().encode() != embedded):
            raise AssertionError("loaded generated source identity mismatch")

        def snapshot(slot: int, label: str) -> tuple[bytes, int, str]:
            position = batch.position(slot)
            pointer = model.ck_model_get_named_activation_ptr(b"kv_cache")
            if not pointer:
                raise AssertionError("active KV arena is unavailable")
            return (_finite_logit_bytes(batch.staged_logits(slot), label),
                    position, hashlib.sha256(
                        ctypes.string_at(pointer, kv_bytes.value)).hexdigest())

        def reset() -> None:
            batch.reset_slot(0)
            batch.reset_slot(1)

        cases = {}
        standalone_cases = {}
        for length in (10, 12, 31):
            prompt = tuple(100 + index for index in range(length))
            survivor_prompt = (200, 201, 202)
            survivor_tokens = tuple(203 + index for index in range(length - 1))
            for prefill_slot in (0, 1):
                decode_slot = prefill_slot ^ 1
                # A prompt can also advance without a concurrent decode row.
                # Every prefix must have the same arithmetic and stored state
                # as the ordinary generated prefill, independently of timing.
                reset()
                solo_ticket = batch.begin_prefill(prefill_slot, length)
                solo = []
                for token in prompt:
                    batch.append_prefill(prefill_slot, solo_ticket, (token,),
                                         prefill_phase=True)
                    solo.append(snapshot(prefill_slot, "standalone phase append"))
                reset()
                for prefix, actual in enumerate(solo, start=1):
                    batch.prefill(prefill_slot, prompt[:prefix])
                    if snapshot(prefill_slot, "ordinary standalone prefix") != actual:
                        raise AssertionError(
                            f"standalone phase append differs at length {length}, "
                            f"slot {prefill_slot}, prefix {prefix}")
                    batch.reset_slot(prefill_slot)
                standalone_cases[f"length_{length}/prefill_slot_{prefill_slot}"] = {
                    "prefixes": length,
                    "full_logits_position_kv_exact": True,
                }
                reset()
                decode_ticket, _ = batch.prefill(decode_slot, survivor_prompt)
                prompt_ticket = batch.begin_prefill(prefill_slot, length)
                if batch.append_prefill(prefill_slot, prompt_ticket, prompt[:1]) != 1:
                    raise AssertionError("first staged row did not complete")
                staged = [snapshot(prefill_slot, "initial staged prompt")]
                decoded = []
                tickets = [0, 0]
                tickets[decode_slot], tickets[prefill_slot] = decode_ticket, prompt_ticket
                for index, (prompt_token, decode_token) in enumerate(
                        zip(prompt[1:], survivor_tokens, strict=True), start=2):
                    tokens = [0, 0]
                    tokens[prefill_slot], tokens[decode_slot] = prompt_token, decode_token
                    mask, consumed, _ = batch.mixed_step(
                        tuple(tokens), tuple(tickets), prefill_slot,
                        prefill_phase=True)
                    if (mask, consumed) != (3, index):
                        raise AssertionError("mixed step failed to advance both rows")
                    staged.append(snapshot(prefill_slot, f"mixed prompt {index}"))
                    decoded.append(snapshot(decode_slot, f"mixed decode {index}"))
                # Finishing prefill must leave the slot ready for ordinary
                # generated decode, including its next KV write and position.
                continuation_token = 300
                continuation_tokens = [0, 0]
                continuation_tokens[prefill_slot] = continuation_token
                if batch.step(tuple(continuation_tokens))[0] != 3:
                    raise AssertionError("completed prompt did not enter decode")
                continued = snapshot(prefill_slot, "decode after mixed prefill")
                reset()

                # Ordinary prefill resets the tested slot for every prefix.
                for prefix, actual in enumerate(staged, start=1):
                    _, logits = batch.prefill(prefill_slot, prompt[:prefix])
                    expected = snapshot(prefill_slot, f"ordinary prefix {prefix}")
                    if actual != expected:
                        raise AssertionError(
                            f"ordinary/mixed logits, position or KV differ at "
                            f"length {length}, slot {prefill_slot}, prefix {prefix}")
                    batch.reset_slot(prefill_slot)
                batch.prefill(prefill_slot, prompt)
                if batch.step(tuple(continuation_tokens))[0] != 1 << prefill_slot or \
                        snapshot(prefill_slot, "isolated decode after prefill") != continued:
                    raise AssertionError(
                        f"decode after mixed prefill differs at length {length}, "
                        f"slot {prefill_slot}")
                batch.reset_slot(prefill_slot)

                # Decode the other row alone from the same original prefix.
                batch.prefill(decode_slot, survivor_prompt)
                for index, (token, actual) in enumerate(
                        zip(survivor_tokens, decoded, strict=True), start=1):
                    tokens = [0, 0]
                    tokens[decode_slot] = token
                    if batch.step(tuple(tokens))[0] != 1 << decode_slot:
                        raise AssertionError("isolated survivor advanced wrong slot")
                    expected = snapshot(decode_slot, f"isolated decode {index}")
                    if actual != expected:
                        raise AssertionError(
                            f"isolated/mixed survivor differs at length {length}, "
                            f"slot {decode_slot}, step {index}")
                reset()
                cases[f"length_{length}/prefill_slot_{prefill_slot}"] = {
                    "steps": length - 1,
                    "prompt_prefix_logits_position_kv_exact": True,
                    "subsequent_decode_logits_position_kv_exact": True,
                    "survivor_logits_position_kv_exact": True,
                }

        # Deliberately take the original decode-phase path on the staged row.
        # This must fail the numerical gate on the captured sensitive prompt;
        # otherwise this certificate would not distinguish the two phases.
        wrong_prompt = tuple(100 + index for index in range(12))
        reset()
        decode_ticket, _ = batch.prefill(1, (200, 201, 202))
        staged_ticket = batch.begin_prefill(0, len(wrong_prompt))
        batch.append_prefill(0, staged_ticket, wrong_prompt[:1])
        wrong_phase = [snapshot(0, "wrong-phase initial row")]
        for index, token in enumerate(wrong_prompt[1:], start=2):
            mask, consumed, _ = batch.mixed_step(
                (token, 202 + index), (staged_ticket, decode_ticket), 0,
                prefill_phase=False)
            if (mask, consumed) != (3, index):
                raise AssertionError("wrong-phase control failed before numerical comparison")
            wrong_phase.append(snapshot(0, f"wrong-phase prefix {index}"))
        reset()
        first_wrong_phase_difference = None
        wrong_phase_components = None
        for prefix, actual in enumerate(wrong_phase, start=1):
            batch.prefill(0, wrong_prompt[:prefix])
            expected = snapshot(0, f"wrong-phase ordinary prefix {prefix}")
            if expected != actual:
                first_wrong_phase_difference = prefix
                wrong_phase_components = {
                    "full_logits": actual[0] != expected[0],
                    "position": actual[1] != expected[1],
                    "complete_kv": actual[2] != expected[2],
                }
                break
            batch.reset_slot(0)
        if first_wrong_phase_difference is None:
            raise AssertionError("wrong attention phase escaped the numerical gate")
        if not (wrong_phase_components["full_logits"] or
                wrong_phase_components["complete_kv"]):
            raise AssertionError("wrong phase changed only the position check")
        reset()

        # A cancelled staged row cannot advance the survivor. Retiring that
        # ticket and reusing the slot must not let stale cancellation reach a
        # new request; the survivor is compared with isolated decode again.
        reset()
        survivor_ticket, _ = batch.prefill(1, (200, 201, 202))
        cancelled_ticket = batch.begin_prefill(0, 3)
        batch.append_prefill(0, cancelled_ticket, (100,))
        before = (snapshot(0, "before cancellation"),
                  snapshot(1, "survivor before cancellation"))
        batch.request_cancel(cancelled_ticket)
        try:
            batch.mixed_step((101, 203), (cancelled_ticket, survivor_ticket),
                             0, prefill_phase=True)
        except SessionError as exc:
            if exc.status != -9:
                raise AssertionError("cancelled step returned wrong status") from exc
        else:
            raise AssertionError("cancelled staged row advanced")
        if before != (snapshot(0, "cancelled row"),
                      snapshot(1, "cancelled survivor")):
            raise AssertionError("cancellation mutated a sequence")
        batch.reset_slot(0)
        batch.request_cancel(cancelled_ticket)
        new_ticket = batch.begin_prefill(0, 2)
        batch.append_prefill(0, new_ticket, (110,))
        mask, consumed, _ = batch.mixed_step(
            (111, 203), (new_ticket, survivor_ticket), 0,
            prefill_phase=True)
        if (mask, consumed) != (3, 2):
            raise AssertionError("reused staged slot or survivor did not advance")
        reused = snapshot(0, "reused prompt")
        survivor = snapshot(1, "surviving decode")
        reset()
        batch.prefill(0, (110, 111))
        if snapshot(0, "ordinary reused prompt") != reused:
            raise AssertionError("reused prompt differs from ordinary prefill")
        batch.reset_slot(0)
        batch.prefill(1, (200, 201, 202))
        if batch.step((0, 203))[0] != 2 or \
                snapshot(1, "isolated survivor after reuse") != survivor:
            raise AssertionError("survivor differs after cancellation and reuse")
        reset()

        return {
            "status": "pass",
            "scope": "one-row phase-aware append and two-row mixed native execution",
            "bundle": str(bundle), "loaded_sha256": loaded,
            "session_library_sha256": native["sha256"],
            "generated_source_sha256": embedded.decode(),
            "weights_sha256": _digest(bundle / "weights.bump"),
            "shared_projection_groups": groups, "shared_layers": layers,
            "context_length": context_length, "threads": threads,
            "cpu": _cpu_identity(), "kv_bytes_per_sequence": kv_bytes.value,
            "cases": cases,
            "standalone_phase_append": standalone_cases,
            "wrong_phase_negative_control": {
                "status": "detected",
                "first_different_prefix": first_wrong_phase_difference,
                "different_components": wrong_phase_components,
            },
            "cancellation_reuse_survivor": "pass",
        }
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = certify(args.bundle, context_length=args.context_length,
                     threads=args.threads)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
