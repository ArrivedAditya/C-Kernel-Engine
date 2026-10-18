#!/usr/bin/env python3
"""Pack pinned Kokoro weights and constants for a native request.

This is offline import tooling. It does not execute the model. The native host
binds request IDs, voice conditioning and noise before calling generated C.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

from export_kokoro_bump import verify_bundle


HERE = Path(__file__).resolve().parent
DEFAULT_BINDINGS = HERE / "kokoro_request_bindings.json"
REQUIRED_BUFFERS = (
    "word_ids", "predictor_style", "decoder_style", "source_gaussian",
    "source_stft_window", "source_stft_cos", "source_stft_sin",
    "generator_waveform",
)
MAX_ARENA_BYTES = 4 * 1024 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _extent(item: dict, capacity: int) -> tuple[int, int]:
    offset, size = item["abs_offset"], item["size"]
    if (not isinstance(offset, int) or not isinstance(size, int) or
            offset < 0 or size <= 0 or offset % 4 or size % 4 or
            offset > capacity or size > capacity - offset):
        raise ValueError(f"invalid arena extent: {item.get('name')}")
    return offset, offset + size


def _nonoverlapping(regions: list[tuple[int, int, str]]) -> None:
    for previous, current in zip(sorted(regions), sorted(regions)[1:]):
        if previous[1] > current[0]:
            raise ValueError(f"overlapping package regions: {previous[2]} and {current[2]}")


def package_request(layout_path: Path, bump_dir: Path, output: Path,
                    bindings_path: Path = DEFAULT_BINDINGS,
                    *, runtime_lengths: bool = False) -> dict:
    layout_bytes = layout_path.read_bytes()
    layout = json.loads(layout_bytes)
    bindings = json.loads(bindings_path.read_text())
    if bindings.get("schema") != "cke.kokoro.request_weight_bindings.v1":
        raise ValueError("unsupported Kokoro request binding schema")
    manifest_path = bump_dir / "weights_manifest.json"
    bump_path = bump_dir / "weights.bump"
    if (sha256_file(manifest_path) != bindings["source_manifest_sha256"] or
            sha256_file(bump_path) != bindings["source_bump_sha256"]):
        raise ValueError("BUMP artifact does not match pinned request bindings")
    manifest = verify_bundle(bump_dir)
    source_entries = {entry["name"]: entry for entry in manifest["entries"]}
    memory = layout["memory"]
    capacity = memory["arena"]["total_size"]
    if not isinstance(capacity, int) or not 0 < capacity <= MAX_ARENA_BYTES:
        raise ValueError("unsupported arena capacity")
    weights = memory["weights"]["entries"]
    if {item["name"] for item in weights} != set(bindings["weights"]) or \
            len(weights) != len(bindings["weights"]):
        raise ValueError("compiled graph weights differ from pinned bindings")
    activation_items = memory["activations"]["buffers"]
    buffers = {item["name"]: item for item in activation_items}
    if len(buffers) != len(activation_items):
        raise ValueError("duplicate compiled activation names")
    required = (*REQUIRED_BUFFERS, "request_length") if runtime_lengths else REQUIRED_BUFFERS
    if any(name not in buffers for name in required):
        raise ValueError("compiled graph lacks required request buffers")
    if (any(item.get("dtype") != "fp32" for item in weights) or
            any(buffers[name].get("dtype") !=
                ("i32" if name in ("word_ids", "request_length") else "fp32")
                for name in required)):
        raise ValueError("compiled request dtype differs from native ABI")
    tokens = buffers["word_ids"]["size"] // 4
    if (buffers["word_ids"]["size"] != tokens * 4 or
            (tokens != 64 if runtime_lengths else not 3 <= tokens <= 40) or
            (runtime_lengths and buffers["request_length"]["size"] != 4) or
            buffers["predictor_style"]["size"] != 512 or
            buffers["decoder_style"]["size"] != 512 or
            buffers["source_gaussian"]["size"] != 76800 * 9 * 4 or
            any(buffers[name]["size"] != size for name, size in (
                ("source_stft_window", 80), ("source_stft_cos", 880),
                ("source_stft_sin", 880)))):
        raise ValueError("compiled request buffer geometry differs")
    regions = []
    for item in weights:
        start, end = _extent(item, capacity)
        regions.append((start, end, item["name"]))
    for name in required:
        start, end = _extent(buffers[name], capacity)
        regions.append((start, end, name))
    _nonoverlapping(regions)

    constants = bindings["constants_fp32_le_hex"]
    if set(constants) != {"source_stft_window", "source_stft_cos", "source_stft_sin"}:
        raise ValueError("missing STFT import constants")
    segments = []
    with bump_path.open("rb") as stream:
        for item in weights:
            rule = bindings["weights"][item["name"]]
            sources = rule["source"]
            if len(sources) not in (1, 2):
                raise ValueError(f"invalid source count: {item['name']}")
            if len(sources) == 2 and rule.get("transform") != "concatenate-forward-reverse":
                raise ValueError(f"unrecognized recurrent transform: {item['name']}")
            payload = bytearray()
            for name in sources:
                if name not in source_entries:
                    raise ValueError(f"missing pinned BUMP source: {name}")
                entry = source_entries[name]
                if entry["dtype"] != "fp32":
                    raise ValueError(f"unsupported weight dtype: {name}")
                stream.seek(entry["file_offset"])
                data = stream.read(entry["size"])
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise ValueError(f"BUMP payload mismatch: {name}")
                payload.extend(data)
            if (len(payload) != item["size"] or
                    hashlib.sha256(payload).hexdigest() != rule["sha256"]):
                raise ValueError(f"transformed weight mismatch: {item['name']}")
            segments.append((item["abs_offset"], bytes(payload)))
    for name, encoded in constants.items():
        payload = bytes.fromhex(encoded)
        if len(payload) != buffers[name]["size"]:
            raise ValueError(f"STFT constant size mismatch: {name}")
        segments.append((buffers[name]["abs_offset"], payload))
    segments.sort(key=lambda segment: segment[0])
    if len(segments) > 1_000_000:
        raise ValueError("too many package segments")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent,
                                         prefix=output.name + ".",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            fields = (
                capacity,
                buffers["generator_waveform"]["abs_offset"],
                buffers["generator_waveform"]["size"] // 4,
                buffers["source_gaussian"]["abs_offset"],
                buffers["source_gaussian"]["size"] // 4,
                buffers["word_ids"]["abs_offset"], tokens,
                buffers["predictor_style"]["abs_offset"],
                buffers["decoder_style"]["abs_offset"],
            )
            if runtime_lengths:
                fields += (buffers["request_length"]["abs_offset"],)
            stream.write((b"CKTTS004" if runtime_lengths else b"CKTTS003") +
                         struct.pack("<10QI" if runtime_lengths else "<9QI",
                                     *fields, len(segments)))
            for offset, payload in segments:
                stream.write(struct.pack("<QQ", offset, len(payload)))
                stream.write(payload)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "schema": "cke.kokoro.request_package.v1",
        "package_sha256": sha256_file(output),
        "layout_sha256": hashlib.sha256(layout_bytes).hexdigest(),
        "bindings_sha256": sha256_file(bindings_path),
        "bump_sha256": bindings["source_bump_sha256"],
        "phoneme_count": tokens,
        "runtime_lengths": runtime_lengths,
        "arena_capacity": capacity,
        "weights": len(weights),
        "segments": len(segments),
        "scope": "offline static model package; native C binds request IDs, voice and noise",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--bump-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_request(args.layout, args.bump_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
