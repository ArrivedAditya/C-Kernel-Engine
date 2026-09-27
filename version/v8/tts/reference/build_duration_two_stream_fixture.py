"""Package pinned torch captures for the generated two-stream duration graph.

The upstream capture supplies both feature tensors, durations, and the actual
frame-to-token alignment. This script only packages those captured values.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "version/v8/tts/reference/fixture_manifest.json"
OUTPUT = ROOT / "tests/fixtures/tts"
NAMES = ("predictor_text_encoder", "text_encoder", "predicted_duration",
         "frame_to_token")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    tensors = {}
    for name in NAMES:
        info = manifest["tensors"][name]
        source = args.capture_dir / info["file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"pinned capture hash mismatch: {name}")
        tensors[name] = np.load(source, allow_pickle=False)
    durations = tensors["predicted_duration"].astype(np.int32)
    frame_to_token = tensors["frame_to_token"].astype(np.int32)
    if durations.shape != (36,) or frame_to_token.shape != (103,):
        raise ValueError("unexpected pinned duration or alignment shape")
    if not np.array_equal(np.repeat(np.arange(36), durations), frame_to_token):
        raise ValueError("captured duration and alignment disagree")
    duration = np.ascontiguousarray(tensors["predictor_text_encoder"][0].T)
    text = np.ascontiguousarray(tensors["text_encoder"][0])
    if duration.shape != (640, 36) or text.shape != (512, 36):
        raise ValueError("unexpected pinned feature shape")
    duration_features = np.full((640, 40), -77.0, dtype=np.float32)
    text_features = np.full((640, 40), -77.0, dtype=np.float32)
    duration_features[:, :36] = duration
    text_features[:512, :36] = text
    arrays = {
        "duration_features": duration_features,
        "text_features": text_features,
        "duration_expected": np.ascontiguousarray(duration[:, frame_to_token]),
        "text_expected": np.ascontiguousarray(text[:, frame_to_token]),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "duration_two_stream_reference.npz", **arrays)
    metadata = {
        "reference": "Pinned PyTorch Kokoro capture; expected expansions indexed by captured frame_to_token alignment",
        "source_manifest": "version/v8/tts/reference/fixture_manifest.json",
        "source_sha256": {name: manifest["tensors"][name]["sha256"] for name in NAMES},
        "arrays_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest()
                          for name, value in arrays.items()},
        "durations": [int(value) for value in durations],
        "frames": len(frame_to_token),
        "duration_channels": 640,
        "text_channels": 512,
        "input_stride": 40,
        "output_stride": 128,
        "text_physical_channels": 640,
        "unused_text_channels_sentinel": -77.0,
        "unused_input_columns_sentinel": -77.0,
    }
    (OUTPUT / "duration_two_stream_reference.json").write_text(
        json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
