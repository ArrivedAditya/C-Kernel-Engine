#!/usr/bin/env python3
"""Connect generated first-frame Qwen speech codes to the complete codec.

One native entry owns prompt execution, first-frame code generation, all
sixteen codec ID edges, and the one-frame codec decoder. This is a bounded
integration fixture: later talker frames, EOS, and useful speech remain.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

if __package__:
    from .build_qwen3_tts_talker_predictor_frame_circuit import (
        ROOT, build_circuit as generated_codes,
    )
else:
    from build_qwen3_tts_talker_predictor_frame_circuit import (
        ROOT, build_circuit as generated_codes,
    )
from build_qwen3_tts_codec_waveform_circuit import build_circuit as codec_waveform

OUTPUT = ROOT / "version/v8/circuits/qwen3_tts_first_frame_waveform_bounded.json"
SAMPLES = 1920


def build_circuit() -> dict:
    graph = generated_codes()
    graph["name"] = "qwen3_tts_first_frame_waveform_bounded"
    graph["native_entry"]["function"] = "ck_qwen3_tts_first_frame_waveform_bounded"
    graph["contract"]["runtime_invariants"].update({
        "generated_codec_frames": 1,
        "generated_sample_count": SAMPLES,
        "first_frame_codec_connected": True,
        "complete_text_to_wav": False,
        "audible_speech": False,
    })
    ids = {"semantic_ids": "talker_first_code_id"}
    ids.update({f"acoustic_ids_{i}": (
        "pred_predictor_codebook0_selected_id" if i == 0 else
        f"pred_step{i}_selected_id") for i in range(15)})
    return attach_codec(graph, frames=1, ids=ids)


def attach_codec(graph: dict, *, frames: int, ids: dict[str, str]) -> dict:
    """Append codec operations with explicit, already-produced codebook spans."""
    codec = codec_waveform(frames=frames)
    samples = frames * SAMPLES
    shapes, bindings = graph["activation_buffers"], graph["activation_bindings"]

    def slot(value: str) -> str:
        external = value.startswith("external:")
        name = value.removeprefix("external:")
        if name in ids:
            return ids[name]
        return ("external:" if external else "") + "codec_" + name

    for name, shape in codec["activation_buffers"].items():
        if name in ids:
            if shape["shape"] != [frames] or shapes[ids[name]]["shape"] != [frames]:
                raise ValueError(f"codec ID geometry differs: {name}")
            continue
        mapped = "codec_" + name
        if mapped in shapes:
            raise ValueError(f"codec slot collides: {mapped}")
        shapes[mapped] = copy.deepcopy(shape)
        bindings[mapped] = mapped
    for block_name in codec["sequence"]:
        block = copy.deepcopy(codec["block_types"][block_name])
        for op in block.get("header", []) + block.get("body", {}).get("ops", []) + block.get("footer", []):
            op["id"] = "codec_" + op["id"]
            for direction in ("inputs", "outputs"):
                slots = op.get("graph_slots", {}).get(direction, {})
                op["graph_slots"][direction] = {
                    port: slot(name) for port, name in slots.items()}
        mapped = "codec_" + block_name
        graph["sequence"].append(mapped)
        graph["block_types"][mapped] = block
    exports = graph["semantic_checkpoints"]["exports"]
    for name, value in codec["semantic_checkpoints"]["exports"].items():
        item = copy.deepcopy(value)
        item["template_op_id"] = "codec_" + item["template_op_id"]
        for checkpoint in item["checkpoints"]:
            checkpoint["id"] = "connected_" + checkpoint["id"]
            checkpoint["producer"] = "codec_" + checkpoint["producer"]
            checkpoint["tensor"] = slot(checkpoint["tensor"])
        exports["codec_" + name] = item

    shapes["waveform_valid_samples"] = {"shape": [1]}
    bindings["waveform_valid_samples"] = "waveform_valid_samples"
    graph["block_types"][graph["sequence"][0]]["header"].insert(0, {
        "id": "clear_waveform_sample_count",
        "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
        "returns_status": True,
        "params": {"call_constants": {
            "extent_affine_source": 0, "extent_affine_factor": 1,
            "extent_affine_offset": 0, "extent_affine_capacity": samples}},
        "graph_slots": {"inputs": {},
                        "outputs": {"valid_extent": "waveform_valid_samples"}},
    })
    graph["sequence"].append("publish_waveform_samples")
    graph["block_types"]["publish_waveform_samples"] = {
        "sequence": ["header", "body", "footer"], "header": [],
        "body": {"type": "dense", "ops": [{
            "id": "publish_waveform_sample_count",
            "op": "runtime_extent_affine", "kernel": "runtime_extent_affine_i32",
            "returns_status": True,
            "params": {"call_constants": {
                "extent_affine_source": 0, "extent_affine_factor": 1,
                "extent_affine_offset": samples,
                "extent_affine_capacity": samples}},
            "graph_slots": {"inputs": {},
                            "outputs": {"valid_extent": "waveform_valid_samples"}},
        }
        ]}, "footer": [],
    }
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("first-frame waveform circuit differs from authoring")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
