"""Resolve the bounded decoder state that may be switched between sequences.

This contract is deliberately narrower than general model state. A generated
runtime may switch the declared KV buffer, token positions, and the two
explicitly planned recurrent-state buffers. Unknown persistent mutable state,
encoder state, and multimodal state remain ineligible.
"""

from __future__ import annotations

import re
from typing import Any


def recurrent_state_operand(arg: dict[str, Any]) -> tuple[str, int, str] | None:
    """Return a checked buffer name, literal byte offset, and active pointer.

    The returned expression has the same pointer type as the lowered operand.
    Arbitrary aliases and computed offsets do not qualify for state switching.
    """
    ref = arg.get("buffer_ref")
    if ref not in ("recurrent_conv_state", "recurrent_ssm_state"):
        return None
    macro = "A_" + ref.upper()
    match = re.fullmatch(
        rf"\((const float|float)\*\)\(model->bump \+ (?:{macro}|\({macro} \+ ([0-9]+)\))\)",
        str(arg.get("expr", "")),
    )
    if match is None:
        return None
    offset = int(match.group(2) or 0)
    tail = f" + {offset}" if offset else ""
    return ref, offset, f"({match.group(1)}*)((uint8_t*)model->{ref}{tail})"


def resolve_sequence_state_contract(
    layout: dict[str, Any], config: dict[str, Any],
    ops: list[dict[str, Any]] | None = None,
) -> dict[str, int] | None:
    """Return checked per-sequence regions for a text-only decoder, or none.

    Offsets and sizes come from the compiler's physical layout.  They are never
    inferred from model names or from the generated C text.
    """
    buffers = layout.get("memory", {}).get("activations", {}).get("buffers", [])
    if not isinstance(buffers, list):
        raise ValueError("activation buffers must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for row in buffers:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("activation buffer has no name")
        name = row["name"]
        if name in by_name:
            raise ValueError(f"duplicate activation buffer {name}")
        by_name[name] = row
    # A missing lifetime is unknown, not a call-local allocation. Older
    # layouts consequently cannot advertise sequence isolation by accident.
    if any(
        row.get("lifetime") not in {"call", "sequence", "model"}
        or not isinstance(row.get("mutable"), bool)
        for row in buffers
    ):
        return None
    if (
        bool(config.get("uses_cross_attention"))
        or bool(config.get("_template_uses_persistent_cross_kv_cache"))
        or bool(config.get("uses_vision"))
        or str(config.get("artifact_scope", "")).lower() == "encoder_only"
    ):
        return None
    kv = by_name.get("kv_cache")
    if kv is None or kv.get("lifetime") != "sequence" or kv.get("mutable") is not True:
        return None
    recurrent_names = ("recurrent_conv_state", "recurrent_ssm_state")
    recurrent = [by_name.get(name) for name in recurrent_names]
    if any(kind not in (None, "none", "full_attention") for kind in
           config.get("layer_recurrent_policy", [])) and recurrent[0] is None:
        return None
    if any(row is not None for row in recurrent) != all(row is not None for row in recurrent):
        return None
    if any(row is not None and (row.get("lifetime") != "sequence" or
                                row.get("mutable") is not True) for row in recurrent):
        return None
    if any(
        all(row is not state for state in (kv, *recurrent)) and (
            row.get("lifetime") == "sequence"
            or (row.get("lifetime") == "model" and row["mutable"])
        ) for row in buffers
    ):
        return None
    offset = kv.get("abs_offset")
    size = kv.get("size")
    if not isinstance(offset, int) or not isinstance(size, int) or offset < 0 or size <= 0:
        raise ValueError("KV cache requires a positive size and absolute offset")
    if offset % 64:
        raise ValueError("KV cache must preserve the planner's 64-byte alignment")
    arena_size = layout.get("memory", {}).get("arena", {}).get("total_size")
    if isinstance(arena_size, int) and offset + size > arena_size:
        raise ValueError("KV cache exceeds the planned arena")
    regions = [kv, *(row for row in recurrent if row is not None)]
    for row in regions:
        start, length = row.get("abs_offset"), row.get("size")
        if not isinstance(start, int) or not isinstance(length, int) or start < 0 or length <= 0:
            raise ValueError(f"{row['name']} requires a positive size and absolute offset")
        if start % 64:
            raise ValueError(f"{row['name']} must preserve the planner's 64-byte alignment")
        if isinstance(arena_size, int) and start + length > arena_size:
            raise ValueError(f"{row['name']} exceeds the planned arena")
        for name, other in by_name.items():
            if other is row:
                continue
            other_offset, other_size = other.get("abs_offset"), other.get("size")
            if isinstance(other_offset, int) and isinstance(other_size, int) and other_size > 0:
                if start < other_offset + other_size and other_offset < start + length:
                    raise ValueError(f"{row['name']} overlaps {name}")
    result = {"kv_offset": offset, "kv_bytes": size, "kv_alignment": 64}
    if recurrent[0] is not None:
        if not ops:
            return None
        layers = config.get("num_layers")
        if not isinstance(layers, int) or layers <= 0:
            return None
        strides: dict[str, int] = {}
        for name in recurrent_names:
            region_bytes = by_name[name]["size"]
            if region_bytes % layers:
                return None
            strides[name] = region_bytes // layers
        seen: set[str] = set()
        for op in ops:
            for arg in op.get("args", []):
                ref = arg.get("buffer_ref")
                if ref not in recurrent_names:
                    continue
                seen.add(ref)
                operand = recurrent_state_operand(arg)
                if (operand is None or operand[1] % strides[ref] or
                        operand[1] + strides[ref] > by_name[ref]["size"]):
                    return None
        if seen != set(recurrent_names):
            return None
        # Each caller arena is compact and aligned independently of the model
        # arena's absolute offsets. The compiler's region sizes remain exact.
        cursor = (size + 63) & ~63
        for name, row in zip(recurrent_names, recurrent):
            assert row is not None
            result[f"{name}_offset"] = row["abs_offset"]
            result[f"{name}_bytes"] = row["size"]
            result[f"{name}_arena_offset"] = cursor
            cursor = (cursor + row["size"] + 63) & ~63
        result["arena_bytes"] = cursor
    return result
