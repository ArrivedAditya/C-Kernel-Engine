#!/usr/bin/env python3
"""Capture the first ALBERT unmasked attention context from pinned Q/K/V.

This is offline oracle tooling; the native circuit executes model arithmetic.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "tests/fixtures/tts/kokoro_qkv_pinned.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = dict(np.load(SOURCE))
    tensors = {name: np.ascontiguousarray(source[f"{name}_expected"])
               for name in ("query", "key", "value")}
    heads, head_dim = 12, 64
    def as_heads(name):
        return torch.from_numpy(tensors[name]).reshape(36, heads, head_dim).transpose(0, 1)
    q, k, v = (as_heads(name) for name in ("query", "key", "value"))
    with torch.no_grad():
        scores = torch.matmul(q, k.transpose(1, 2)) * (head_dim ** -0.5)
        probabilities = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, v).transpose(0, 1).contiguous().reshape(36, 768)
    tensors["context_expected"] = context.numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **tensors)
    sidecar = {
        "oracle": f"PyTorch {torch.__version__} eager matmul/softmax/matmul",
        "scope": "first shared ALBERT attention context, all 36 tokens valid, no dropout",
        "source_fixture_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "heads": heads, "head_dim": head_dim, "tokens": 36,
        "shapes": {name: list(array.shape) for name, array in tensors.items()},
        "complete_albert_layer": "NOT_TESTED", "generated_waveform": "NOT_TESTED",
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")


if __name__ == "__main__":
    main()
