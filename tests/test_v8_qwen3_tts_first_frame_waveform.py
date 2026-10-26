"""One generated Qwen talker/predictor frame joined to the complete codec."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_prompt_first_block import (
    FIXTURE, compile_entry, pinned_prompt_ids, view,
)
from version.v8.tts.build_qwen3_tts_prompt_first_block_circuit import prompt_plan
from version.v8.tts.build_qwen3_tts_talker_predictor_frame_circuit import bridge_plan
from version.v8.tts.build_qwen3_tts_first_frame_waveform_circuit import (
    OUTPUT, SAMPLES, build_circuit,
)

CODEC_SOURCES = (
    "src/kernels/audio_zero_pad_checked.c",
    "src/kernels/audio_conv1d_checked.c",
    "src/kernels/audio_conv_transpose_dense_checked.c",
    "src/kernels/audio_conv1d_depthwise_checked.c",
    "src/kernels/layernorm_rows_checked.c",
    "src/kernels/layernorm_kernels.c",
    "src/kernels/gelu_rows_erf_checked.c",
    "src/kernels/gelu_kernels.c",
    "src/kernels/audio_snake_checked.c",
    "src/kernels/audio_scaled_sum_strided.c",
)
CAPTURE = FIXTURE / "qwen3_tts_first_frame_codec.npz"
MANIFEST = FIXTURE / "qwen3_tts_first_frame_codec.json"


def _weight_shapes(graph, main, codec):
    refs = {name for block in graph["sequence"]
            for op in graph["block_types"][block].get("body", {}).get("ops", [])
            for name in op.get("weight_refs", {}).values()}
    shapes = {}
    for name in refs:
        if name.startswith("codec."):
            if name.endswith("effective_embedding"):
                shape = (2048, 256)
            elif name.endswith("output_projection.weight"):
                shape = (512, 256)
            else:
                shape = codec.get_slice(name.removeprefix("codec.")).get_shape()
        else:
            shape = main.get_slice(name).get_shape()
        shapes[name] = tuple(shape)
    return shapes


def _codec_value(source, name):
    if name == "codec.semantic.effective_embedding":
        group, index = "rvq_first", 0
    elif name.startswith("codec.acoustic.") and name.endswith(".effective_embedding"):
        group, index = "rvq_rest", int(name.split(".")[2])
    else:
        group = None
    if group is not None:
        prefix = f"decoder.quantizer.{group}.vq.layers.{index}._codebook."
        usage = source.get_tensor(prefix + "cluster_usage").float().numpy()
        summed = source.get_tensor(prefix + "embedding_sum").float().numpy()
        return np.ascontiguousarray(
            (summed / np.maximum(usage, 1e-5)[:, None]).astype(np.float32))
    if name == "codec.semantic.output_projection.weight":
        source_name = "decoder.quantizer.rvq_first.output_proj.weight"
    elif name == "codec.acoustic.output_projection.weight":
        source_name = "decoder.quantizer.rvq_rest.output_proj.weight"
    else:
        source_name = name.removeprefix("codec.")
    return np.ascontiguousarray(source.get_tensor(source_name).float().numpy().reshape(
        (512, 256) if name.endswith("output_projection.weight") else
        source.get_slice(source_name).get_shape()))


class Qwen3TTSFirstFrameWaveformTest(unittest.TestCase):
    def test_graph_and_oracle_identity_without_assets(self):
        graph = build_circuit()
        self.assertEqual(OUTPUT.read_text(), json.dumps(graph, indent=2) + "\n")
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        self.assertEqual(hashlib.sha256((FIXTURE / "qwen3_tts_customvoice_reference.npz").read_bytes()).hexdigest(),
                         manifest["source_codes_sha256"])
        self.assertEqual(hashlib.sha256((FIXTURE / "qwen3_tts_code_predictor.npz").read_bytes()).hexdigest(),
                         manifest["source_predictor_sha256"])
        ids = [op["graph_slots"]["inputs"]["ids"]
               for op in graph["block_types"]["codec_codec_semantic"]["body"]["ops"]
               if op["id"] == "codec_semantic_lookup" or
               op["id"].startswith("codec_acoustic_lookup_")]
        self.assertEqual(ids, ["talker_first_code_id",
                               "pred_predictor_codebook0_selected_id"] +
                         [f"pred_step{i}_selected_id" for i in range(1, 15)])
        self.assertEqual(graph["contract"]["runtime_invariants"]["generated_sample_count"],
                         SAMPLES)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; connected PCM NOT_TESTED")
    def test_pinned_generated_first_frame_pcm(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        graph = build_circuit()
        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as main, \
             safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as codec, \
             np.load(FIXTURE / "qwen3_tts_customvoice_reference.npz") as request, \
             np.load(CAPTURE) as oracle, tempfile.TemporaryDirectory() as temp:
            shapes = _weight_shapes(graph, main, codec)
            fn, acts, weights, layout, calls, library = compile_entry(
                Path(temp), graph, shapes, extra_sources=CODEC_SOURCES)
            self.assertEqual(len(calls["operations"]), 2455)
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for name, item in weights.items():
                value = (_codec_value(codec, name) if name.startswith("codec.")
                         else np.ascontiguousarray(main.get_tensor(name).float().numpy()))
                self.assertEqual(value.shape, shapes[name], name)
                arena[item["abs_offset"]:item["abs_offset"] + value.nbytes] = value.tobytes()
            text_ids, codec_ids = pinned_prompt_ids(model_dir, request)
            view(arena, acts, "text_ids", np.int32, text_ids.shape)[:] = text_ids
            view(arena, acts, "codec_ids", np.int32, codec_ids.shape)[:] = codec_ids
            for name, indices in prompt_plan(7).items():
                view(arena, acts, name, np.int32, (18,))[:] = indices
            for name, indices in bridge_plan(18).items():
                view(arena, acts, "bridge_" + name, np.int32, (2,))[:] = indices
            view(arena, acts, "pred_feedback_step_index_input", np.int32, (1,))[0] = 0
            valid_codes = view(arena, acts, "pred_predictor_valid_code_count", np.int32, (1,))
            valid_samples = view(arena, acts, "waveform_valid_samples", np.int32, (1,))
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual((int(valid_codes[0]), int(valid_samples[0])), (15, SAMPLES))
            id_names = ["talker_first_code_id", "pred_predictor_codebook0_selected_id"] + [
                f"pred_step{i}_selected_id" for i in range(1, 15)]
            selected = [int(view(arena, acts, name, np.int32, (1,))[0])
                        for name in id_names]
            np.testing.assert_array_equal(selected, oracle["code_ids"][0])
            pcm = view(arena, acts, "codec_waveform_output", np.float32, (1, SAMPLES))
            self.assertTrue(np.isfinite(pcm).all())
            delta = pcm.astype(np.float64) - oracle["waveform_output"].astype(np.float64)
            maximum = float(np.max(np.abs(delta)))
            self.assertLessEqual(maximum, 3e-6)
            print("QWEN3_TTS_FIRST_FRAME_PCM " + json.dumps({
                "codes": selected, "samples": SAMPLES, "sample_rate_hz": 24000,
                "max_abs": maximum,
                "rmse": float(np.sqrt(np.mean(delta * delta))),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "complete_text_to_wav": "NOT_TESTED",
            }, sort_keys=True))

            baseline = pcm.copy()
            # Reject after the codes are produced, before waveform execution.
            table = np.ndarray((2048, 256), dtype=np.float32, buffer=arena,
                               offset=weights["codec.semantic.effective_embedding"]["abs_offset"])
            saved = float(table[selected[0], 0])
            table[selected[0], 0] = np.nan
            self.assertNotEqual(fn(arena, size), 0)
            self.assertEqual(int(valid_samples[0]), 0)
            np.testing.assert_array_equal(pcm, baseline)
            table[selected[0], 0] = saved
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(valid_samples[0]), SAMPLES)
            np.testing.assert_array_equal(pcm, baseline)


if __name__ == "__main__":
    unittest.main()
