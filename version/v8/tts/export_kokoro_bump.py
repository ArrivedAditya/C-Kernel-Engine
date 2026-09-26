#!/usr/bin/env python3
"""Export pinned Kokoro v1 weights and one selected voice to BUMPWGT5.

This is offline import tooling. Generated native execution must consume the
resulting canonical tensors without importing or scheduling Kokoro in Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from convert_gguf_to_bump_v8 import (  # noqa: E402
    BUMP_META_FOOTER_MAGIC, BUMP_VERSION_V5, CACHE_ALIGN, CK_DT_FP32, DATA_START,
    EXT_METADATA_SIZE, HEADER_SIZE, _canonical_json_bytes,
    calculate_manifest_hash, calculate_metadata_hash, write_bumpv5_footer,
)


PIN = json.loads((HERE / "reference/kokoro_v1_reference.json").read_text())
REFERENCE = json.loads((HERE / "reference/fixture_manifest.json").read_text())
CANONICAL_PREFIXES = {
    "bert": "phoneme_encoder",
    "bert_encoder": "phoneme_projection",
    "predictor": "duration_prosody",
    "text_encoder": "acoustic_text_encoder",
    "decoder": "waveform_decoder",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pinned_assets(model_dir: Path) -> dict[str, str]:
    actual = {}
    for name, expected in REFERENCE["assets"].items():
        path = model_dir / name
        if not path.is_file():
            raise ValueError(f"missing pinned Kokoro asset: {path}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"{path}: SHA-256 {digest} does not match pinned {expected}")
        actual[name] = digest
    return actual


def canonical_name(source_name: str) -> str:
    prefix, separator, suffix = source_name.partition(".")
    if not separator or prefix not in CANONICAL_PREFIXES or not suffix:
        raise ValueError(f"unsupported Kokoro source tensor: {source_name}")
    if ".parametrizations." in source_name or source_name.endswith(("weight_g", "weight_v")):
        raise ValueError(f"unresolved weight parametrization: {source_name}")
    return f"{CANONICAL_PREFIXES[prefix]}.{suffix}"


def _align_up(value: int, alignment: int = CACHE_ALIGN) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _as_fp32(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected nonempty finite FP32 tensor")
    return np.ascontiguousarray(array, dtype="<f4")


def write_bundle(
    output_dir: Path,
    tensors: dict[str, np.ndarray],
    origins: dict[str, dict],
    config: dict,
    provenance: dict,
) -> dict:
    """Write caller supplied canonical FP32 tensors with checked BUMP offsets."""
    if not tensors or set(tensors) != set(origins):
        raise ValueError("tensor and origin sets must match and be nonempty")
    for name, origin in origins.items():
        if (set(origin) != {"source_name", "transform"} or
                not all(isinstance(origin[key], str) and origin[key] for key in origin)):
            raise ValueError(f"{name}: expected source_name and transform provenance")
    arrays = {name: _as_fp32(value, name) for name, value in tensors.items()}
    names = sorted(arrays)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "weights.bump"
    temporary_path = output_dir / "weights.bump.tmp"
    manifest_path = output_dir / "weights_manifest.json"
    dtype_bytes = bytes([CK_DT_FP32] * len(names))
    entries = []
    try:
        with temporary_path.open("w+b") as stream:
            stream.write(b"\0" * (HEADER_SIZE + EXT_METADATA_SIZE))
            data_digest = hashlib.sha256()

            def write_data(payload: bytes) -> None:
                stream.write(payload)
                data_digest.update(payload)

            write_data(struct.pack("<I", len(dtype_bytes)))
            write_data(dtype_bytes)
            offset = DATA_START + 4 + len(dtype_bytes)
            for name in names:
                array = arrays[name]
                next_offset = _align_up(offset)
                if next_offset > offset:
                    write_data(b"\0" * (next_offset - offset))
                payload = array.tobytes(order="C")
                write_data(payload)
                entries.append({
                    "name": name,
                    "dtype": "fp32",
                    "shape": list(array.shape),
                    "file_offset": next_offset,
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    **origins[name],
                })
                offset = next_offset + len(payload)

            manifest = {
                "schema_version": 1,
                "format": "BUMPWGT5",
                "architecture": "kokoro_v1",
                "artifact_scope": "canonical_weights_and_fixed_voice",
                "generated_waveform_status": "NOT_TESTED",
                "template": {"name": "kokoro_v1", "artifact_scope": "weights_only"},
                "config": config,
                "pin": PIN,
                "provenance": provenance,
                "entries": entries,
            }
            metadata = {
                "schema_version": 1,
                "format": "BUMPWGT5",
                "created_by": "version/v8/tts/export_kokoro_bump.py",
                "template": manifest["template"],
                "config": config,
                "quant_summary": {"all": "fp32"},
                "manifest_hash": calculate_manifest_hash(manifest),
            }
            metadata_bytes = _canonical_json_bytes(metadata)
            stream.seek(0)
            stream.write(b"BUMPWGT5")
            stream.write(struct.pack("<I", BUMP_VERSION_V5))
            stream.write(struct.pack("<I", 1))
            for value in (
                0, int(config["n_token"]), int(config["hidden_dim"]),
                int(config["plbert"]["intermediate_size"]),
                int(config["plbert"]["max_position_embeddings"]),
                int(config["plbert"]["num_attention_heads"]), 0, 0,
            ):
                stream.write(struct.pack("<I", value))
            for value in (
                int(config["hidden_dim"]), 0,
                int(config["plbert"]["intermediate_size"]),
                int(config["plbert"]["max_position_embeddings"]),
            ):
                stream.write(struct.pack("<Q", value))
            stream.write(struct.pack("<II", 0, 0))
            stream.write(data_digest.digest())
            stream.seek(0, os.SEEK_END)
            stream.write(metadata_bytes)
            write_bumpv5_footer(stream, len(metadata_bytes), calculate_metadata_hash(metadata))
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(weights_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        (output_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        return manifest
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_bundle(output_dir: Path) -> dict:
    """Validate the BUMP envelope, every manifest extent and every FP32 payload."""
    weights_path = output_dir / "weights.bump"
    manifest = json.loads((output_dir / "weights_manifest.json").read_text())
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("missing BUMP manifest entries")
    names = [entry["name"] for entry in entries]
    if len(set(names)) != len(names) or names != sorted(names):
        raise ValueError("duplicate or unordered canonical tensor names")
    with weights_path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        file_size = stream.tell()
        if file_size < DATA_START + 4 + 48:
            raise ValueError("truncated BUMP file")
        stream.seek(0)
        if stream.read(8) != b"BUMPWGT5":
            raise ValueError("invalid BUMP magic")
        if struct.unpack("<I", stream.read(4))[0] != BUMP_VERSION_V5:
            raise ValueError("unsupported BUMP version")
        stream.seek(88)
        expected_data_hash = stream.read(32)
        stream.seek(DATA_START)
        count = struct.unpack("<I", stream.read(4))[0]
        if count != len(entries) or stream.read(count) != bytes([CK_DT_FP32] * count):
            raise ValueError("BUMP dtype table does not match manifest")
        table_end = DATA_START + 4 + count
        stream.seek(file_size - 48)
        if stream.read(8) != BUMP_META_FOOTER_MAGIC:
            raise ValueError("invalid BUMP metadata footer")
        metadata_size = struct.unpack("<Q", stream.read(8))[0]
        expected_metadata_hash = stream.read(32)
        metadata_start = file_size - 48 - metadata_size
        if metadata_start < table_end:
            raise ValueError("invalid BUMP metadata extent")
        stream.seek(metadata_start)
        metadata_bytes = stream.read(metadata_size)
        if hashlib.sha256(metadata_bytes).digest() != expected_metadata_hash:
            raise ValueError("BUMP metadata hash mismatch")
        metadata = json.loads(metadata_bytes)
        if metadata.get("manifest_hash") != calculate_manifest_hash(manifest):
            raise ValueError("BUMP manifest hash mismatch")
        stream.seek(DATA_START)
        data_hash = hashlib.sha256()
        remaining = metadata_start - DATA_START
        while remaining:
            block = stream.read(min(1 << 20, remaining))
            if not block:
                raise ValueError("truncated BUMP data")
            data_hash.update(block)
            remaining -= len(block)
        if data_hash.digest() != expected_data_hash:
            raise ValueError("BUMP data hash mismatch")
        end = table_end
        for entry in entries:
            shape = entry["shape"]
            if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
                raise ValueError(f"{entry['name']}: invalid shape")
            elements = 1
            for dim in shape:
                elements *= dim
            start = entry["file_offset"]
            size = entry["size"]
            if (entry["dtype"] != "fp32" or size != elements * 4 or
                    start % CACHE_ALIGN or start < end or start + size > metadata_start):
                raise ValueError(f"{entry['name']}: invalid payload extent")
            stream.seek(start)
            payload_hash = hashlib.sha256()
            left = size
            while left:
                block = stream.read(min(1 << 20, left))
                if not block:
                    raise ValueError(f"{entry['name']}: truncated payload")
                payload_hash.update(block)
                left -= len(block)
            if payload_hash.hexdigest() != entry["sha256"]:
                raise ValueError(f"{entry['name']}: payload hash mismatch")
            end = start + size
    return manifest


def export(model_dir: Path, output_dir: Path) -> dict:
    import importlib.metadata
    import inspect
    import torch
    from torch.nn.utils.parametrize import remove_parametrizations
    from kokoro import KModel
    from kokoro.modules import AdaLayerNorm

    hashes = verify_pinned_assets(model_dir)
    actual_versions = {
        "kokoro": importlib.metadata.version("kokoro"),
        "torch": importlib.metadata.version("torch"),
    }
    for package, actual in actual_versions.items():
        expected = REFERENCE["environment"]["packages"][package]
        if actual != expected:
            raise ValueError(f"{package} {actual} != pinned oracle package {expected}")
    config = json.loads((model_dir / "config.json").read_text())
    raw_checkpoint = torch.load(
        model_dir / "kokoro-v1_0.pth", map_location="cpu", weights_only=True
    )
    raw_names = {
        f"{section}.{name.removeprefix('module.')}"
        for section, section_tensors in raw_checkpoint.items()
        for name in section_tensors
    }
    if len(raw_names) != 548:
        raise ValueError(f"unexpected raw Kokoro tensor inventory: {len(raw_names)}")
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    model = KModel(
        repo_id=PIN["model"]["repository"],
        config=str(model_dir / "config.json"),
        model=str(model_dir / "kokoro-v1_0.pth"),
    ).eval()
    transformed = set()
    for module_name, module in model.named_modules():
        if hasattr(module, "parametrizations") and hasattr(module.parametrizations, "weight"):
            remove_parametrizations(module, "weight", leave_parametrized=True)
            transformed.add(f"{module_name}.weight")
    state = model.state_dict()
    if len(transformed) != 89 or len(state) != 599:
        raise ValueError(f"unexpected effective Kokoro inventory: {len(transformed)} transforms, {len(state)} tensors")
    tensors = {}
    origins = {}
    consumed_raw = set()
    synthesized = 0
    for source_name, value in state.items():
        name = canonical_name(source_name)
        if name in tensors:
            raise ValueError(f"canonical tensor collision: {name}")
        tensors[name] = value.detach().cpu().numpy()
        if source_name in transformed:
            stem = source_name.removesuffix(".weight")
            raw_pair = (f"{stem}.weight_g", f"{stem}.weight_v")
            if not all(part in raw_names for part in raw_pair):
                raise ValueError(f"missing raw weight-norm pair for {source_name}")
            consumed_raw.update(raw_pair)
            origin = {
                "source_name": " + ".join(
                    f"{part.partition('.')[0]}:module.{part.partition('.')[2]}"
                    for part in raw_pair
                ),
                "transform": "PyTorch weight_norm effective weight; remove_parametrizations(leave_parametrized=True)",
            }
        elif source_name in raw_names:
            consumed_raw.add(source_name)
            section, _, suffix = source_name.partition(".")
            origin = {
                "source_name": f"{section}:module.{suffix}",
                "transform": "identity",
            }
        else:
            expected = 1.0 if source_name.endswith(".norm.weight") else 0.0
            if not source_name.endswith((".norm.weight", ".norm.bias")) or not source_name.startswith(
                ("predictor.", "decoder.")
            ) or not np.all(tensors[name] == expected):
                raise ValueError(f"unaccounted constructor tensor: {source_name}")
            synthesized += 1
            origin = {
                "source_name": "synthetic:torch.nn.InstanceNorm1d(affine=True)",
                "transform": f"pinned model constructor default {expected:g}",
            }
        origins[name] = origin
    if consumed_raw != raw_names or synthesized != 140:
        raise ValueError(
            f"incomplete checkpoint coverage: used {len(consumed_raw)}/{len(raw_names)} "
            f"raw tensors, synthesized {synthesized}/140"
        )
    voice_pack = torch.load(model_dir / "voices/af_heart.pt", map_location="cpu", weights_only=True)
    row_index = int(REFERENCE["voice_row_index"])
    expected_index = len(REFERENCE["phonemes"]) - 1
    if row_index != expected_index:
        raise ValueError("pinned voice-row selection mismatch")
    voice_row = voice_pack[row_index].detach().cpu().numpy().reshape(-1)
    if voice_row.shape != (256,):
        raise ValueError(f"unexpected selected voice shape: {voice_row.shape}")
    for name, data, transform in (
        ("voice.fixed.full", voice_row, "selected row"),
        ("voice.fixed.decoder", voice_row[:128], "selected row columns 0:128"),
        ("voice.fixed.predictor", voice_row[128:], "selected row columns 128:256"),
    ):
        tensors[name] = data
        origins[name] = {
            "source_name": f"voices/af_heart.pt[{row_index}]",
            "transform": transform,
        }
    provenance = {
        "source_asset_sha256": hashes,
        "reference_code": PIN["reference_code"],
        "oracle_packages": actual_versions,
        "oracle_source_sha256": {
            "kokoro.model": sha256_file(Path(inspect.getfile(KModel))),
            "kokoro.modules": sha256_file(Path(inspect.getfile(AdaLayerNorm))),
        },
        "package_source_commit_match": "NOT_TESTED",
        "voice_row_index": row_index,
        "effective_weight_transforms": len(transformed),
        "source_tensor_coverage": {
            "pass": True,
            "raw_checkpoint_tensors": len(raw_names),
            "consumed_raw_tensors": len(consumed_raw),
            "synthesized_instance_norm_affine_tensors": synthesized,
        },
    }
    write_bundle(output_dir, tensors, origins, config, provenance)
    return verify_bundle(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest = export(args.model_dir.resolve(), args.output_dir.resolve())
    print(f"exported {len(manifest['entries'])} canonical tensors to {args.output_dir / 'weights.bump'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
