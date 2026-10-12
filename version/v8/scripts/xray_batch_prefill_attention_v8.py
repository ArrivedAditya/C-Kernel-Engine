#!/usr/bin/env python3
"""Compare existing CKDMP attention captures at one logical query.

Capture production execution in fresh processes. The layer IDs in CKDMP are
invocation ordinals, so the caller must explicitly identify each phase's
ordinal and query. This analyzer does not infer them from model names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import ctypes
from pathlib import Path

from server.serving_bundle import verified_loaded_symbol_backing
from server.session_v8 import SESSION_LIB_PATH, SessionV8
from version.v8.scripts.xray_batch_prefill_kv_v8 import attention_span

_HEADER = struct.Struct("<8sIi32sII4qIi32s")
_NAME = re.compile(r"(kq_scores|kq_soft|kqv_out|vcols)_h(\d+)_q(\d+)")


def read_dump(path: Path, *, layer: int, query: int) -> dict:
    """Read the native CKDMP v1 vector records, rejecting stale/ambiguous data."""
    data = path.read_bytes()
    records = {}
    offset = 0
    while offset < len(data):
        if len(data) - offset < _HEADER.size:
            raise ValueError("truncated attention header")
        header = _HEADER.unpack_from(data, offset)
        offset += _HEADER.size
        magic, version, actual_layer, raw_name, dtype, rank = header[:6]
        count, actual_query = header[10:12]
        name = raw_name.split(b"\0", 1)[0].decode("ascii")
        match = _NAME.fullmatch(name)
        if (magic != b"CKDMP\0\0\0" or version != 1 or
                actual_layer != layer or actual_query != query or
                not match or int(match[3]) != query or dtype != 0 or rank != 1 or
                header[6] != count or count <= 0 or name in records):
            raise ValueError("invalid or ambiguous attention record")
        size = count * 4
        if len(data) - offset < size:
            raise ValueError("truncated attention vector")
        records[name] = struct.unpack_from(f"<{count}f", data, offset)
        offset += size
    if not records:
        raise ValueError("empty attention dump")
    return records


def _half(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def compare_dumps(ordinary: dict, staged: dict, *, ordinary_query: int,
                  staged_query: int, position: int, sliding_window: int) -> dict:
    heads = sorted(int(match[2]) for name in ordinary
                   if (match := _NAME.fullmatch(name)) and
                   match[1] == "kq_scores")
    if not heads or len(heads) != len(set(heads)):
        raise ValueError("missing attention heads")
    staged_heads = sorted(int(match[2]) for name in staged
                          if (match := _NAME.fullmatch(name)) and
                          match[1] == "kq_scores")
    if staged_heads != heads:
        raise ValueError("attention head sets differ")
    result = {}
    for head in heads:
        vectors = {}
        for kind in ("kq_scores", "kq_soft", "kqv_out"):
            a = ordinary.get(f"{kind}_h{head}_q{ordinary_query}")
            b = staged.get(f"{kind}_h{head}_q{staged_query}")
            if a is None or b is None:
                raise ValueError("missing attention vector")
            vectors[kind] = a, b
        scores_a, scores_b = vectors["kq_scores"]
        probs_a, probs_b = vectors["kq_soft"]
        out_a, out_b = vectors["kqv_out"]
        if (len(scores_a) != len(probs_a) or len(scores_b) != len(probs_b) or
                len(out_a) != len(out_b) or min(len(scores_a), len(scores_b)) <= position):
            raise ValueError("invalid attention vector extents")
        observed = []
        for scores in (scores_a, scores_b):
            if any(not math.isfinite(value) and value != -math.inf for value in scores):
                raise ValueError("invalid masked attention score")
            finite = [i for i, value in enumerate(scores) if math.isfinite(value)]
            if not finite or finite != list(range(finite[0], finite[-1] + 1)):
                raise ValueError("noncontiguous attention read span")
            observed.append(attention_span(position=position,
                                           sliding_window=sliding_window,
                                           observed_first=finite[0],
                                           observed_last=finite[-1]))
        if any(not math.isfinite(value) or value < 0 for value in probs_a + probs_b):
            raise ValueError("invalid attention probability")
        if not all(math.isfinite(value) for value in out_a + out_b):
            raise ValueError("nonfinite attention output")
        differences = [abs(a - b) for a, b in zip(out_a, out_b)]
        info = {
            "read_span": observed[0],
            "ordinary_padded_tokens": len(scores_a),
            "staged_padded_tokens": len(scores_b),
            "score_max_abs_diff": max(abs(a - b) for a, b in
                                      zip(scores_a, scores_b) if math.isfinite(a)),
            "output_max_abs_diff": max(differences),
            "output_differing_channels": sum(a != b for a, b in zip(out_a, out_b)),
        }
        v_a = ordinary.get(f"vcols_h{head}_q{ordinary_query}")
        v_b = staged.get(f"vcols_h{head}_q{staged_query}")
        if (v_a is None) != (v_b is None):
            raise ValueError("unpaired V columns")
        if v_a is not None:
            if len(v_a) != len(out_a) * len(scores_a) or len(v_b) != len(out_b) * len(scores_b):
                raise ValueError("invalid V-column extents")
            first, last = observed[0]["first"], observed[0]["last"]
            if any(v_a[d * len(scores_a) + t] != v_b[d * len(scores_b) + t]
                   for d in range(len(out_a)) for t in range(first, last + 1)):
                raise ValueError("logical attention V differs")
            info["logical_v_exact"] = True
            info["fp16_probability_differences"] = [
                t for t in range(first, last + 1)
                if _half(probs_a[t]) != _half(probs_b[t])]
            channel = max(range(len(out_a)), key=lambda d: differences[d])
            reference = [sum(_half(probs[t]) * _half(v[d * len(probs) + t])
                             for t in range(first, last + 1))
                         for probs, v, d in ((probs_a, v_a, channel),
                                             (probs_b, v_b, channel))]
            info["max_difference_channel"] = channel
            info["independent_fp16_operand_sum_delta"] = reference[0] - reference[1]
            info["measured_output_delta"] = out_a[channel] - out_b[channel]
        result[str(head)] = info
    return result


def capture(bundle: Path, *, mode: str, directory: Path, layer_ordinal: int,
            query: int, kv_layer: int, tokens: tuple[int, ...], slot: int,
            enabled: bool) -> dict:
    """Run one phase in a fresh process; optionally capture native vectors."""
    if mode not in ("ordinary", "staged") or slot not in (0, 1) or not tokens:
        raise ValueError("invalid attention capture mode, slot or token sequence")
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = directory / "strict_internal.bin"
    if enabled and trace_path.exists():
        raise ValueError("attention capture directory already contains a trace")
    if enabled:
        os.environ.pop("CK_STRICT_ATTN_DUMP_HEAD", None)
        os.environ.update({
            "CK_PARITY_DIR": str(directory.resolve()),
            "CK_STRICT_ATTN_VEC_DUMP": "1",
            "CK_STRICT_ATTN_DUMP_VCOLS": "1",
            "CK_STRICT_ATTN_DUMP_LAYER": str(layer_ordinal),
            "CK_STRICT_ATTN_DUMP_QUERY": str(query),
        })
    else:
        os.environ.pop("CK_STRICT_ATTN_VEC_DUMP", None)
    session = SessionV8.open(bundle, context_length=max(32, len(tokens)), num_threads=1)
    try:
        batch = session.enable_batch2(1 << 30)
        model = ctypes.CDLL(str(bundle / "libmodel.so"))
        model.ck_model_debug_export_kv.argtypes = [ctypes.c_char_p, ctypes.c_int]
        model.ck_model_debug_export_kv.restype = ctypes.c_int
        loaded_model = verified_loaded_symbol_backing(
            model, "ck_model_debug_export_kv", bundle / "libmodel.so")
        loaded_session = verified_loaded_symbol_backing(
            session.lib, "ck_session_v8_batch2_mixed_step", SESSION_LIB_PATH)
        if mode == "ordinary":
            _, logits = batch.prefill(slot, tokens)
        else:
            ticket = batch.begin_prefill(slot, len(tokens))
            for index, token in enumerate(tokens, 1):
                if batch.append_prefill(slot, ticket, (token,)) != index:
                    raise AssertionError("staged prefill progress mismatch")
            logits = batch.staged_logits(slot)
        if batch.position(slot) != len(tokens):
            raise AssertionError("captured prefill position mismatch")
        kv_path = directory / f"kv_layer_{kv_layer}.ckx"
        if model.ck_model_debug_export_kv(str(kv_path).encode(), kv_layer):
            raise RuntimeError("KV capture failed")
        if enabled:
            read_dump(trace_path, layer=layer_ordinal, query=query)
        return {
            "mode": mode, "slot": slot, "tokens": list(tokens),
            "position": batch.position(slot),
            "logits_sha256": hashlib.sha256(bytes(logits)).hexdigest(),
            "kv_sha256": hashlib.sha256(kv_path.read_bytes()).hexdigest(),
            "loaded_model_sha256": loaded_model["sha256"],
            "loaded_session_sha256": loaded_session["sha256"],
            "attention_capture_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()
            if enabled else None,
        }
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-bundle", type=Path,
                        help="capture one ordinary or staged prefill in a fresh process")
    parser.add_argument("--mode", choices=("ordinary", "staged"))
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--layer-ordinal", type=int)
    parser.add_argument("--query", type=int)
    parser.add_argument("--kv-layer", type=int, default=16)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--tokens", default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("ordinary", type=Path, nargs="?")
    parser.add_argument("staged", type=Path, nargs="?")
    parser.add_argument("--ordinary-layer", type=int)
    parser.add_argument("--staged-layer", type=int)
    parser.add_argument("--ordinary-query", type=int)
    parser.add_argument("--staged-query", type=int)
    parser.add_argument("--position", type=int)
    parser.add_argument("--sliding-window", type=int)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.capture_bundle:
        if (args.capture_dir is None or args.mode is None or
                args.layer_ordinal is None or args.query is None):
            parser.error("capture needs mode, directory, layer ordinal and query")
        result = capture(args.capture_bundle, mode=args.mode,
                         directory=args.capture_dir, layer_ordinal=args.layer_ordinal,
                         query=args.query, kv_layer=args.kv_layer,
                         tokens=tuple(int(token) for token in args.tokens.split(",")),
                         slot=args.slot, enabled=not args.no_capture)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if (args.ordinary is None or args.staged is None or
            any(value is None for value in (args.ordinary_layer, args.staged_layer,
                                            args.ordinary_query, args.staged_query,
                                            args.position, args.sliding_window))):
        parser.error("comparison needs both dumps, ordinals, queries, position and window")
    result = {
        "schema": "cke.v8.batch-prefill-attention-xray.v1",
        "capture_sha256": {"ordinary": hashlib.sha256(args.ordinary.read_bytes()).hexdigest(),
                           "staged": hashlib.sha256(args.staged.read_bytes()).hexdigest()},
        "selection": {"ordinary_layer_ordinal": args.ordinary_layer,
                      "staged_layer_ordinal": args.staged_layer,
                      "ordinary_query": args.ordinary_query,
                      "staged_query": args.staged_query,
                      "position": args.position,
                      "sliding_window": args.sliding_window},
        "heads": compare_dumps(
            read_dump(args.ordinary, layer=args.ordinary_layer, query=args.ordinary_query),
            read_dump(args.staged, layer=args.staged_layer, query=args.staged_query),
            ordinary_query=args.ordinary_query, staged_query=args.staged_query,
            position=args.position, sliding_window=args.sliding_window),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
