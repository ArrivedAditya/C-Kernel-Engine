"""Resolve the first, deliberately narrow generated two-row decode contract.

This is an IR/layout capability check. It does not identify models by name and
does not rewrite emitted C. Unsupported graphs retain their ordinary decoder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sequence_state_contract_v8 import resolve_sequence_state_contract


_WEIGHT = re.compile(r"^\(const void\*\)\(model->bump \+ (W_[A-Z0-9_]+)\)$")


def _args(op: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {arg["name"]: arg for arg in op.get("args", []) if isinstance(arg, dict) and isinstance(arg.get("name"), str)}


def resolve_two_row_batch_contract(
    ops: list[dict[str, Any]], layout: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any] | None:
    """Admit only a KV-only decoder with two equivalent first-layer projections.

    The initial split is intentionally exact: embedding, residual save, norm,
    Q/K projections, then the rest of the normal generated decode. The two
    projections must have the same input and the same map-declared M=2 provider.
    This leaves all attention and cache writes sequence-local.
    """
    if resolve_sequence_state_contract(layout, config) is None or len(ops) < 6:
        return None
    if [op.get("op") for op in ops[:5]] != [
        "dense_embedding_lookup", "residual_save", "attn_norm", "q_proj", "k_proj"
    ]:
        return None
    q, k = ops[3:5]
    if q.get("layer") != k.get("layer") or q.get("function") != k.get("function"):
        return None
    qargs, kargs = _args(q), _args(k)
    if not all(name in qargs and name in kargs for name in ("x", "y", "W", "M", "K")):
        return None
    if qargs["x"].get("buffer_ref") != kargs["x"].get("buffer_ref"):
        return None
    if qargs["K"].get("expr") != kargs["K"].get("expr"):
        return None
    try:
        input_dim = int(qargs["K"]["expr"])
        q_dim, k_dim = int(qargs["M"]["expr"]), int(kargs["M"]["expr"])
    except (TypeError, ValueError, KeyError):
        return None
    if min(input_dim, q_dim, k_dim) <= 0 or input_dim % 32:
        return None
    buffers = {
        row.get("name"): row
        for row in layout.get("memory", {}).get("activations", {}).get("buffers", [])
        if isinstance(row, dict)
    }
    refs = {
        "input": qargs["x"].get("buffer_ref"),
        "q": qargs["y"].get("buffer_ref"),
        "k": kargs["y"].get("buffer_ref"),
        "residual": _args(ops[1]).get("dst", {}).get("buffer_ref"),
    }
    if len(set(refs.values())) != 4 or any(ref not in buffers for ref in refs.values()):
        return None
    sizes = {name: input_dim * 4 if name == "input" else
             q_dim * 4 if name == "q" else
             k_dim * 4 if name == "k" else
             int(buffers[ref].get("size", 0)) for name, ref in refs.items()}
    for name, ref in refs.items():
        row = buffers[ref]
        if row.get("lifetime") != "call" or row.get("mutable") is not True:
            return None
        if int(row.get("size", 0)) < sizes[name] or not re.fullmatch(r"A_[A-Z0-9_]+", str(row.get("define", ""))):
            return None
    weights = []
    for args in (qargs, kargs):
        match = _WEIGHT.fullmatch(str(args["W"].get("expr", "")))
        if not match:
            return None
        weights.append(match.group(1))
    # The output path must be last-only; a per-position logits buffer requires
    # a separate destination contract before batched decode can be advertised.
    logits = buffers.get("logits")
    vocab = config.get("vocab_size")
    if not logits or not isinstance(vocab, int) or logits.get("size") != vocab * 4:
        return None
    map_root = Path(__file__).resolve().parents[1] / "kernel_maps"
    matches = []
    for path in map_root.glob("*.json"):
        if path.name == "KERNEL_REGISTRY.json":
            continue
        # Only explicit two-row bindings participate; normal prefill selection
        # is not silently reinterpreted as a decode guarantee.
        raw = path.read_text(encoding="utf-8")
        if '"batch_decode_two_rows"' not in raw:
            continue
        provider = json.loads(raw)
        binding = provider.get("batch_decode_two_rows") or {}
        if binding.get("decode_function") == q.get("function") and binding.get("rows") == 2:
            matches.append(binding)
    if len(matches) != 1:
        return None
    function = matches[0].get("function")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", str(function or "")):
        return None
    max_input_dim = matches[0].get("max_input_dim")
    if not isinstance(max_input_dim, int) or max_input_dim <= 0 or input_dim > max_input_dim:
        return None
    return {
        "prefix_len": 3,
        "suffix_start": 5,
        "input_dim": input_dim,
        "q_dim": q_dim,
        "k_dim": k_dim,
        "buffers": {name: buffers[ref]["define"] for name, ref in refs.items()},
        "sizes": sizes,
        "weights": weights,
        "gemm_function": function,
    }
