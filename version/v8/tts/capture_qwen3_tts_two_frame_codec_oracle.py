#!/usr/bin/env python3
"""Capture the pinned Qwen codec decoding the first two generated code frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

if __package__:
    from .capture_qwen3_tts_customvoice_oracle import (
        MODEL_REVISION, REFERENCE_REVISION, verify_model_dir,
        verify_reference_code,
    )
else:
    from capture_qwen3_tts_customvoice_oracle import (
        MODEL_REVISION, REFERENCE_REVISION, verify_model_dir,
        verify_reference_code,
    )

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/tts/qwen3_tts_0b6_customvoice"


def capture(model_dir: Path, output_dir: Path) -> dict:
    model_hashes = verify_model_dir(model_dir)
    import numpy as np
    import torch
    import qwen_tts
    from safetensors import safe_open
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2DecoderConfig,
    )
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Decoder,
    )

    reference_hashes = verify_reference_code(Path(qwen_tts.__file__).resolve().parent)
    torch.set_num_threads(max(1, int(os.environ.get("CKE_QWEN3_TTS_TORCH_THREADS", "4"))))
    values = json.loads((model_dir / "speech_tokenizer/config.json").read_text())[
        "decoder_config"]
    config = Qwen3TTSTokenizerV2DecoderConfig(**values)
    config._attn_implementation = "eager"
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                   framework="pt", device="cpu") as source:
        decoder.load_state_dict({name.removeprefix("decoder."): source.get_tensor(name)
                                 for name in source.keys() if name.startswith("decoder.")},
                                strict=True)
    source_path = FIXTURES / "qwen3_tts_customvoice_reference.npz"
    with np.load(source_path) as source:
        code_ids = np.ascontiguousarray(source["codec_tokens"][:2], dtype=np.int32)
    first_path = FIXTURES / "qwen3_tts_first_frame_codec.npz"
    with np.load(first_path) as first:
        if not np.array_equal(code_ids[:1], first["code_ids"]):
            raise RuntimeError("first generated code frame differs from pinned codec IDs")
    continuation_path = FIXTURES / "qwen3_tts_talker_continuation.npz"
    with np.load(continuation_path) as continuation:
        second_first = int(np.argmax(continuation["step1_logits"]))
    if code_ids.shape != (2, 16) or int(code_ids[1, 0]) != second_first:
        raise RuntimeError("second talker code differs from pinned codec IDs")
    with torch.inference_mode():
        waveform = decoder(torch.from_numpy(code_ids.T[None].astype(np.int64)))
    samples = np.ascontiguousarray(waveform[0].detach().cpu().numpy(), dtype=np.float32)
    if samples.shape != (1, 3840) or not np.isfinite(samples).all():
        raise RuntimeError("two-frame reference output geometry or finiteness changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = output_dir / "qwen3_tts_two_frame_codec.npz"
    np.savez_compressed(fixture, code_ids=code_ids, waveform_output=samples)
    manifest = {
        "schema": 1,
        "scope": "direct pinned decoder of two generated 16-codebook frames",
        "model_revision": MODEL_REVISION,
        "model_files_sha256": model_hashes,
        "reference_revision": REFERENCE_REVISION,
        "reference_files_sha256": reference_hashes,
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "numpy": np.__version__, "machine": platform.machine(),
                    "device": "cpu", "dtype": "float32", "attention": "eager",
                    "threads": torch.get_num_threads()},
        "source_codes_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_first_frame_sha256": hashlib.sha256(first_path.read_bytes()).hexdigest(),
        "source_continuation_sha256": hashlib.sha256(continuation_path.read_bytes()).hexdigest(),
        "code_ids": code_ids.tolist(), "samples": int(samples.size),
        "sample_rate_hz": 24000,
        "npz_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    }
    (output_dir / "qwen3_tts_two_frame_codec.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=FIXTURES)
    args = parser.parse_args()
    print(json.dumps(capture(args.model_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
