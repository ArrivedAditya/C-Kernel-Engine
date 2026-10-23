#!/usr/bin/env python3
"""Capture pinned split-RVQ codebook reconstruction through official PyTorch.

This is an operation-level codec prefix oracle. It does not certify the
pre-transformer, generator, or waveform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_qwen3_tts_customvoice_oracle import (
    MODEL_REVISION, REFERENCE_REVISION, verify_model_dir,
    verify_reference_code,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/tts/qwen3_tts_0b6_customvoice"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(model_dir: Path, output_dir: Path) -> None:
    verify_model_dir(model_dir)
    import qwen_tts
    import torch
    from safetensors import safe_open
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        SplitResidualVectorQuantizer,
    )
    verify_reference_code(Path(qwen_tts.__file__).resolve().parent)
    torch.set_num_threads(4)
    with np.load(FIXTURES / "qwen3_tts_customvoice_reference.npz") as fixture:
        codes = fixture["codec_tokens"].astype(np.int64)
    module = SplitResidualVectorQuantizer(
        dimension=256, n_q=16, n_q_semantic=1, bins=2048,
        input_dimension=512, output_dimension=512,
    ).eval()
    prefix = "decoder.quantizer."
    with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                   framework="pt", device="cpu") as source:
        weights = {name[len(prefix):]: source.get_tensor(name)
                   for name in source.keys() if name.startswith(prefix)}
    module.load_state_dict(weights, strict=True)
    with torch.inference_mode():
        ids = torch.from_numpy(codes[:, 0]).reshape(1, 1, -1)
        raw = module.rvq_first.vq.layers[0]._codebook.decode(ids[:, 0])
        projected = module.rvq_first.decode(ids)
        all_codes = torch.from_numpy(codes.T.copy()).reshape(1, 16, -1)
        rvq = module.decode(all_codes)
    raw_np = raw[0].numpy().astype(np.float32)
    projected_np = projected[0].T.numpy().astype(np.float32)
    rvq_np = rvq[0].T.numpy().astype(np.float32)
    if raw_np.shape != (len(codes), 256) or projected_np.shape != (len(codes), 512):
        raise RuntimeError("unexpected semantic codec checkpoint geometry")
    if rvq_np.shape != (len(codes), 512):
        raise RuntimeError("unexpected complete RVQ checkpoint geometry")
    if (not np.isfinite(raw_np).all() or
            not np.isfinite(projected_np).all() or
            not np.isfinite(rvq_np).all()):
        raise RuntimeError("nonfinite semantic codec checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "qwen3_tts_codec_semantic.npz"
    np.savez_compressed(npz_path, semantic_ids=codes[:, 0].astype(np.int32),
                        semantic_raw=raw_np, semantic_projected=projected_np,
                        rvq_output=rvq_np)
    manifest = {
        "schema": 1,
        "scope": "official operation-level split-RVQ only; CKE and full codec NOT_TESTED",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "codec_source_sha256": sha256(
            Path(qwen_tts.__file__).resolve().parent
            / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"),
        "source_codes_sha256": sha256(FIXTURES / "qwen3_tts_customvoice_reference.npz"),
        "npz_sha256": sha256(npz_path),
        "frames": len(codes),
        "semantic_codebook": 0,
    }
    (output_dir / "qwen3_tts_codec_semantic.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(output_dir / "qwen3_tts_codec_semantic.json"),
                      "npz_sha256": manifest["npz_sha256"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=FIXTURES)
    args = parser.parse_args()
    capture(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
