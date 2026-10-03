"""Resolve the bounded decoder state that may be switched between sequences.

This contract is deliberately narrower than general model state.  A generated
runtime may expose sequence switching only when its persistent mutable state is
the declared KV buffer and token positions.  Recurrent, encoder, and multimodal
state need their own contracts before they can be admitted.
"""

from __future__ import annotations

from typing import Any


def resolve_sequence_state_contract(
    layout: dict[str, Any], config: dict[str, Any]
) -> dict[str, int] | None:
    """Return the checked KV region for a text-only decoder, or no capability.

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
    if any(
        row is not kv and (
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
    for name, row in by_name.items():
        if name == "kv_cache":
            continue
        other_offset, other_size = row.get("abs_offset"), row.get("size")
        if isinstance(other_offset, int) and isinstance(other_size, int) and other_size > 0:
            if offset < other_offset + other_size and other_offset < offset + size:
                raise ValueError(f"KV cache overlaps {name}")
    return {"kv_offset": offset, "kv_bytes": size, "kv_alignment": 64}
