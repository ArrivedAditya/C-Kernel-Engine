#!/usr/bin/env python3
"""Capture the pinned official causal pre-convolution after codec split-RVQ."""

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
        Qwen3TTSTokenizerV2CausalConvNet,
    )
    verify_reference_code(Path(qwen_tts.__file__).resolve().parent)
    torch.set_num_threads(4)
    rvq_path = FIXTURES / "qwen3_tts_codec_semantic.npz"
    with np.load(rvq_path) as fixture:
        rvq = np.ascontiguousarray(fixture["rvq_output"])
    module = Qwen3TTSTokenizerV2CausalConvNet(512, 1024, 3).eval()
    with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                   framework="pt", device="cpu") as source:
        prefix = "decoder.pre_conv."
        weights = {name[len(prefix):]: source.get_tensor(name)
                   for name in source.keys() if name.startswith(prefix)}
    module.load_state_dict(weights, strict=True)
    with torch.inference_mode():
        output = module(torch.from_numpy(rvq.T.copy()).unsqueeze(0))
    result = np.ascontiguousarray(output[0].numpy().astype(np.float32))
    if result.shape != (1024, len(rvq)) or not np.isfinite(result).all():
        raise RuntimeError("unexpected pre-convolution checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "qwen3_tts_codec_preconv.npz"
    np.savez_compressed(npz_path, preconv_output=result)
    manifest = {
        "schema": 1,
        "scope": "official operation-level pre-convolution given captured RVQ; transformer and waveform NOT_TESTED",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "codec_source_sha256": sha256(
            Path(qwen_tts.__file__).resolve().parent
            / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"),
        "source_rvq_sha256": sha256(rvq_path),
        "npz_sha256": sha256(npz_path),
        "frames": len(rvq),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
    }
    (output_dir / "qwen3_tts_codec_preconv.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(output_dir / "qwen3_tts_codec_preconv.json"),
                      "npz_sha256": manifest["npz_sha256"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=FIXTURES)
    args = parser.parse_args()
    capture(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
