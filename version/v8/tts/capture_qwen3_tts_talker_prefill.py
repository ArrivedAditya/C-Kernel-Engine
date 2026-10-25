#!/usr/bin/env python3
"""Capture every talker prefill layer and final logits from the pinned model.

This is independent reference tooling. It never runs in the CKE generated
speech path.
"""

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
        DEFAULT_LANGUAGE, DEFAULT_SPEAKER, DEFAULT_TEXT, MODEL_REVISION,
        REFERENCE_REVISION, verify_model_dir, verify_reference_code,
    )
else:
    from capture_qwen3_tts_customvoice_oracle import (
        DEFAULT_LANGUAGE, DEFAULT_SPEAKER, DEFAULT_TEXT, MODEL_REVISION,
        REFERENCE_REVISION, verify_model_dir, verify_reference_code,
    )


def capture(model_dir: Path, output_dir: Path) -> dict:
    model_hashes = verify_model_dir(model_dir)
    import numpy as np
    import torch
    import qwen_tts
    from qwen_tts import Qwen3TTSModel

    reference_hashes = verify_reference_code(Path(qwen_tts.__file__).resolve().parent)
    torch.manual_seed(20261022)
    torch.set_num_threads(max(1, int(os.environ.get("CKE_QWEN3_TTS_TORCH_THREADS", "4"))))
    model = Qwen3TTSModel.from_pretrained(
        str(model_dir), device_map="cpu", dtype=torch.float32,
        attn_implementation="eager")
    talker = model.model.talker
    arrays = {}
    handles = []

    def save(name, tensor):
        if tensor.ndim == 3 and tensor.shape[1] > 1 and name not in arrays:
            arrays[name] = tensor.detach().cpu().contiguous().numpy().copy()

    def before_first(_, inputs, __):
        save("prompt_input", inputs[0])

    handles.append(talker.model.layers[0].register_forward_pre_hook(
        before_first, with_kwargs=True))
    for index, layer in enumerate(talker.model.layers):
        def record(_, __, output, layer_index=index):
            save(f"layer{layer_index}_output", output[0])
        handles.append(layer.register_forward_hook(record))
    handles.append(talker.model.norm.register_forward_hook(
        lambda _, __, output: save("final_norm", output)))
    handles.append(talker.codec_head.register_forward_hook(
        lambda _, __, output: save("logits", output)))
    try:
        with torch.inference_mode():
            model.generate_custom_voice(
                text=DEFAULT_TEXT, speaker=DEFAULT_SPEAKER,
                language=DEFAULT_LANGUAGE, do_sample=False,
                subtalker_dosample=False, max_new_tokens=8,
                non_streaming_mode=True)
    finally:
        for handle in handles:
            handle.remove()
    expected = {"prompt_input", "final_norm", "logits"} | {
        f"layer{i}_output" for i in range(28)}
    if set(arrays) != expected:
        raise RuntimeError(f"incomplete prefill capture: missing {sorted(expected - set(arrays))}")
    if arrays["prompt_input"].shape != (1, 18, 1024) or \
       arrays["logits"].shape != (1, 18, 3072):
        raise RuntimeError("unexpected pinned prefill geometry")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("nonfinite prefill reference")

    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = output_dir / "qwen3_tts_talker_prefill.npz"
    np.savez_compressed(fixture, **arrays)
    manifest = {
        "schema": 1,
        "scope": "direct pinned PyTorch 28-layer talker prefill and logits; CKE execution NOT_TESTED",
        "model_revision": MODEL_REVISION,
        "model_files_sha256": model_hashes,
        "reference_revision": REFERENCE_REVISION,
        "reference_files_sha256": reference_hashes,
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "numpy": np.__version__, "machine": platform.machine(),
                    "device": "cpu", "dtype": "float32", "attention": "eager",
                    "threads": torch.get_num_threads()},
        "request": {"text": DEFAULT_TEXT, "speaker": DEFAULT_SPEAKER,
                    "language": DEFAULT_LANGUAGE, "seed": 20261022,
                    "do_sample": False, "subtalker_dosample": False,
                    "non_streaming_mode": True},
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
        "npz_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    }
    (output_dir / "qwen3_tts_talker_prefill.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.model_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
