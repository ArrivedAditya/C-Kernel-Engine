#!/usr/bin/env python3
from __future__ import annotations

"""Audit a safetensors index without downloading or opening weight shards."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


def _load_index(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "model.safetensors.index.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if "weight_map" not in data:
        raise SystemExit(f"{path} is not a safetensors index with weight_map")
    return data


def _family(name: str) -> str:
    if name in {"lm_head.weight", "backbone.embeddings.weight", "model.embed_tokens.weight"}:
        return name
    if ".mixer." not in name:
        return "other"
    tail = name.split(".mixer.", 1)[1]
    if tail.startswith("self_attn.") or tail in {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"}:
        return "attention"
    if tail.startswith("experts."):
        return "moe_expert"
    if tail.startswith("shared_experts."):
        return "moe_shared_expert"
    if tail.startswith("gate."):
        return "moe_router"
    if tail in {"up_proj.weight", "down_proj.weight"}:
        return "dense_mlp"
    if tail.startswith(("in_proj.", "out_proj.", "conv1d.", "dt_bias", "A_log", "D", "norm.")):
        return "mamba"
    return "mixer_other"


def audit_index(path: Path) -> dict[str, Any]:
    data = _load_index(path)
    wm = data["weight_map"]
    families = Counter(_family(name) for name in wm)
    layers: dict[int, Counter[str]] = defaultdict(Counter)
    expert_ids: dict[int, set[int]] = defaultdict(set)
    shards = Counter(wm.values())
    for name in wm:
        m = LAYER_RE.search(name)
        if not m:
            continue
        layer = int(m.group(1))
        fam = _family(name)
        layers[layer][fam] += 1
        em = EXPERT_RE.search(name)
        if em:
            expert_ids[layer].add(int(em.group(1)))
    layer_summary = {
        str(layer): {
            "families": dict(sorted(counter.items())),
            "expert_count": len(expert_ids.get(layer, set())),
        }
        for layer, counter in sorted(layers.items())
    }
    return {
        "metadata": data.get("metadata", {}),
        "tensor_count": len(wm),
        "shard_count": len(shards),
        "shards": dict(sorted(shards.items())),
        "families": dict(sorted(families.items())),
        "layer_count": len(layers),
        "layers": layer_summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit safetensors index tensor families without loading shards")
    ap.add_argument("index", type=Path, help="model.safetensors.index.json or directory")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    report = audit_index(args.index)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
