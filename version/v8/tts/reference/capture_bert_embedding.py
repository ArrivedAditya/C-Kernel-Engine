#!/usr/bin/env python3
"""Capture the pinned Kokoro ALBERT embedding boundary for kernel certification.

Offline PyTorch oracle only; this script is never part of native execution.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from kokoro import KModel


HERE = Path(__file__).resolve().parent
REFERENCE = json.loads((HERE / "fixture_manifest.json").read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir
    for name, expected in REFERENCE["assets"].items():
        path = model_dir / name
        if digest(path) != expected:
            raise ValueError(f"pinned asset mismatch: {name}")
    for package in ("kokoro", "transformers", "torch"):
        actual = importlib.metadata.version(package)
        expected = REFERENCE["environment"]["packages"][package]
        if actual != expected:
            raise ValueError(f"{package} {actual} != pinned {expected}")
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    model = KModel(repo_id="hexgrad/Kokoro-82M",
                   config=str(model_dir / "config.json"),
                   model=str(model_dir / "kokoro-v1_0.pth")).eval()
    ids = torch.tensor(REFERENCE["input_ids"], dtype=torch.long)[None, :]
    embeddings = model.bert.embeddings(ids)
    layer = model.bert.embeddings
    arrays = {
        "ids": ids[0].to(torch.int32).numpy(),
        "type_ids": np.zeros(ids.shape[1], dtype=np.int32),
        "word": layer.word_embeddings.weight.detach().numpy(),
        "position": layer.position_embeddings.weight.detach().numpy(),
        "token_type": layer.token_type_embeddings.weight.detach().numpy(),
        "gamma": layer.LayerNorm.weight.detach().numpy(),
        "beta": layer.LayerNorm.bias.detach().numpy(),
        "expected": embeddings[0].detach().numpy(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest = {
        "oracle": "Kokoro 0.9.4 / Transformers 5.17.0 / PyTorch 2.8.0+cpu",
        "reference_model_sha256": REFERENCE["assets"]["kokoro-v1_0.pth"],
        "input_ids": REFERENCE["input_ids"],
        "epsilon": model.bert.config.layer_norm_eps,
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "fixture_sha256": digest(args.output),
        "status": "oracle_embedding_boundary_only; complete_encoder_not_tested",
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
