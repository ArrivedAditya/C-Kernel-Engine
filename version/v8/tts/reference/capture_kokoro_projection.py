#!/usr/bin/env python3
"""Offline PyTorch F.linear oracle for Kokoro's ALBERT input projection."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "version/v8/tts"))
import export_kokoro_bump as exporter

PREFIX = "phoneme_encoder.encoder.embedding_hidden_mapping_in"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = exporter.verify_bundle(args.bundle_dir)
    entries = {entry["name"]: entry for entry in manifest["entries"]}
    embedding_path = ROOT / "tests/fixtures/tts/bert_embedding_pinned.npz"
    embedding = np.load(embedding_path)["expected"]
    tensors = {}
    with (args.bundle_dir / "weights.bump").open("rb") as stream:
        for suffix in ("weight", "bias"):
            entry = entries[f"{PREFIX}.{suffix}"]
            stream.seek(entry["file_offset"])
            payload = stream.read(entry["size"])
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"BUMP payload hash mismatch: {suffix}")
            tensors[suffix] = np.frombuffer(payload, dtype="<f4").copy().reshape(entry["shape"])
    with torch.no_grad():
        expected = torch.nn.functional.linear(
            torch.from_numpy(embedding), torch.from_numpy(tensors["weight"]),
            torch.from_numpy(tensors["bias"])).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, input=embedding, weight=tensors["weight"],
                        bias=tensors["bias"], expected=expected)
    sidecar = {
        "oracle": f"PyTorch {importlib.metadata.version('torch')} torch.nn.functional.linear",
        "reference_scope": "pinned Kokoro embedding plus exported BUMP projection weights; isolated linear operation",
        "bundle_manifest_sha256": hashlib.sha256(
            (args.bundle_dir / "weights_manifest.json").read_bytes()).hexdigest(),
        "embedding_fixture_sha256": hashlib.sha256(embedding_path.read_bytes()).hexdigest(),
        "shapes": {"input": list(embedding.shape), "weight": list(tensors["weight"].shape),
                   "bias": list(tensors["bias"].shape), "expected": list(expected.shape)},
        "fixture_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "complete_encoder": "NOT_TESTED", "generated_waveform": "NOT_TESTED",
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")


if __name__ == "__main__":
    main()
