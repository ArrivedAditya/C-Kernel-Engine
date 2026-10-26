#!/usr/bin/env python3
"""Capture one pinned greedy Qwen3-TTS code-predictor frame.

This records the first two-token predictor prefill and its fourteen one-token
continuations. It is oracle evidence for connected native bring-up, not an
input to a production synthesis path.
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
    predictor = model.model.talker.code_predictor
    arrays: dict[str, np.ndarray] = {}
    state = {"call": -1, "talker_step": -1}
    handles = []

    def save(name, tensor):
        if tensor is not None:
            arrays[name] = tensor.detach().cpu().contiguous().numpy().copy()

    def before_talker(_, args, kwargs):
        state["talker_step"] += 1
        if state["talker_step"] == 1:
            save("talker_first_code_id", kwargs.get("input_ids"))
            save("talker_past_hidden", kwargs.get("past_hidden"))

    def before_predictor(_, args, kwargs):
        state["call"] += 1
        index = state["call"]
        if index >= 15:
            return
        for name in ("inputs_embeds", "input_ids", "cache_position",
                     "attention_mask", "position_ids"):
            save(f"call{index}_{name}", kwargs.get(name))
        arrays[f"call{index}_generation_steps"] = np.array(
            [-1 if kwargs.get("generation_steps") is None else
             int(kwargs["generation_steps"])], dtype=np.int32)

    def after_predictor(_, args, kwargs, output):
        index = state["call"]
        if index < 15:
            save(f"call{index}_logits", output.logits)
            arrays[f"call{index}_selected_id"] = np.array(
                [int(output.logits[0, -1].argmax())], dtype=np.int32)

    def before_model(_, args, kwargs):
        index = state["call"]
        if index < 15:
            for name in ("inputs_embeds", "cache_position", "position_ids"):
                save(f"call{index}_model_{name}", kwargs.get(name))

    def after_model(_, args, kwargs, output):
        index = state["call"]
        if index >= 15:
            return
        save(f"call{index}_final_norm", output.last_hidden_state)
        rows = 2 if index == 0 else 1
        for layer, cache in enumerate(output.past_key_values.layers):
            for kind, tensor in (("k", cache.keys), ("v", cache.values)):
                suffix = "cache" if index == 0 else "new"
                save(f"call{index}_layer{layer}_{kind}_{suffix}",
                     tensor if index == 0 else tensor[:, :, -rows:, :])

    handles.append(model.model.talker.register_forward_pre_hook(
        before_talker, with_kwargs=True))
    handles.append(predictor.register_forward_pre_hook(
        before_predictor, with_kwargs=True))
    handles.append(predictor.register_forward_hook(
        after_predictor, with_kwargs=True))
    handles.append(predictor.model.register_forward_pre_hook(
        before_model, with_kwargs=True))
    handles.append(predictor.model.register_forward_hook(
        after_model, with_kwargs=True))
    for layer, module in enumerate(predictor.model.layers):
        def record(_, args, output, index=layer):
            if state["call"] < 15:
                save(f"call{state['call']}_layer{index}_output", output[0])
        handles.append(module.register_forward_hook(record))
    diagnostic = predictor.model.layers[2]
    for name in ("input_layernorm", "self_attn.q_proj", "self_attn.k_proj",
                 "self_attn.v_proj", "self_attn", "post_attention_layernorm",
                 "mlp.gate_proj", "mlp.up_proj", "mlp"):
        module = diagnostic.get_submodule(name)
        def record_stage(_, args, output, stage=name):
            if state["call"] == 0:
                save("call0_layer2_" + stage.replace(".", "_"),
                     output[0] if isinstance(output, tuple) else output)
        handles.append(module.register_forward_hook(record_stage))
    try:
        with torch.inference_mode():
            model.generate_custom_voice(
                text=DEFAULT_TEXT, speaker=DEFAULT_SPEAKER,
                language=DEFAULT_LANGUAGE, do_sample=False,
                subtalker_dosample=False, max_new_tokens=2,
                non_streaming_mode=True)
    finally:
        for handle in handles:
            handle.remove()
    if state["call"] != 14 or state["talker_step"] < 1:
        raise RuntimeError("expected one speech frame and 15 predictor calls")
    for index in range(15):
        if arrays[f"call{index}_logits"].shape != (1, 2 if index == 0 else 1, 2048):
            raise RuntimeError(f"unexpected predictor logits geometry at {index}")
        for layer in range(5):
            if f"call{index}_layer{layer}_output" not in arrays:
                raise RuntimeError(f"missing layer {layer} at predictor call {index}")
    if not all(np.isfinite(value).all() for value in arrays.values()
               if value.dtype.kind == "f"):
        raise RuntimeError("nonfinite code-predictor reference")

    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = output_dir / "qwen3_tts_code_predictor.npz"
    np.savez_compressed(fixture, **arrays)
    manifest = {
        "schema": 1,
        "scope": "pinned PyTorch first speech frame: two-token predictor prefill plus fourteen greedy continuation calls",
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
                    "max_new_tokens": 2, "non_streaming_mode": True},
        "selected_ids": [int(arrays[f"call{i}_selected_id"][0]) for i in range(15)],
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
        "npz_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    }
    (output_dir / "qwen3_tts_code_predictor.json").write_text(
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
