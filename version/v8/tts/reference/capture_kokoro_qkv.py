#!/usr/bin/env python3
"""Capture the first shared ALBERT layer's Q/K/V linear outputs with PyTorch.

This is offline oracle tooling. The generated C circuit owns deployed execution.
"""
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

PREFIX = "phoneme_encoder.encoder.albert_layer_groups.0.albert_layers.0.attention"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = exporter.verify_bundle(args.bundle_dir)
    entries = {entry["name"]: entry for entry in manifest["entries"]}
    projection_path = ROOT / "tests/fixtures/tts/kokoro_projection_pinned.npz"
    projection = dict(np.load(projection_path))
    inputs = np.ascontiguousarray(projection["expected"])
    tensors = {"input": inputs}
    with (args.bundle_dir / "weights.bump").open("rb") as stream:
        for name in ("query", "key", "value"):
            for suffix in ("weight", "bias"):
                key = f"{PREFIX}.{name}.{suffix}"
                entry = entries[key]
                stream.seek(entry["file_offset"])
                payload = stream.read(entry["size"])
                if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise ValueError(f"BUMP payload hash mismatch: {key}")
                tensors[f"{name}_{suffix}"] = np.frombuffer(
                    payload, dtype="<f4").copy().reshape(entry["shape"])
            with torch.no_grad():
                tensors[f"{name}_expected"] = torch.nn.functional.linear(
                    torch.from_numpy(inputs),
                    torch.from_numpy(tensors[f"{name}_weight"]),
                    torch.from_numpy(tensors[f"{name}_bias"])).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **tensors)
    sidecar = {
        "oracle": f"PyTorch {importlib.metadata.version('torch')} torch.nn.functional.linear",
        "scope": "first shared ALBERT Q/K/V projections from pinned generated-encoder input",
        "bundle_manifest_sha256": sha256(args.bundle_dir / "weights_manifest.json"),
        "input_fixture_sha256": sha256(projection_path),
        "fixture_sha256": sha256(args.output),
        "shapes": {key: list(value.shape) for key, value in tensors.items()},
        "complete_albert_layer": "NOT_TESTED",
        "generated_waveform": "NOT_TESTED",
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")


if __name__ == "__main__":
    main()
