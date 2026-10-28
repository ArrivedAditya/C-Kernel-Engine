"""Two CKE-generated Qwen speech-code frames joined to generated codec PCM."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_first_frame_waveform import (
    CODEC_SOURCES, _codec_value, _weight_shapes,
)
from tests.test_v8_qwen3_tts_prompt_first_block import (
    FIXTURE, compile_entry, pinned_prompt_ids, view,
)
from tests.test_v8_qwen3_tts_second_generated_frame import prepare_inputs
from version.v8.tts.build_qwen3_tts_two_frame_waveform_circuit import (
    OUTPUT, SAMPLES, build_circuit,
)

CAPTURE = FIXTURE / "qwen3_tts_two_frame_codec.npz"
MANIFEST = FIXTURE / "qwen3_tts_two_frame_codec.json"


class Qwen3TTSTwoFrameWaveformTest(unittest.TestCase):
    def test_graph_and_oracle_integrity_without_assets(self):
        graph = build_circuit()
        self.assertEqual(OUTPUT.read_text(), json.dumps(graph, indent=2) + "\n")
        self.assertNotIn("segments", graph["native_entry"])
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        for key, filename in (
            ("source_codes_sha256", "qwen3_tts_customvoice_reference.npz"),
            ("source_first_frame_sha256", "qwen3_tts_first_frame_codec.npz"),
            ("source_continuation_sha256", "qwen3_tts_talker_continuation.npz"),
        ):
            self.assertEqual(hashlib.sha256((FIXTURE / filename).read_bytes()).hexdigest(),
                             manifest[key])
        ops = graph["block_types"]["pack_generated_codec_frames"]["body"]["ops"]
        self.assertEqual(len(ops), 16)
        self.assertEqual(ops[0]["graph_slots"]["inputs"], {
            "left": "talker_first_code_id", "right": "second_talker_first_code_id"})
        self.assertEqual(ops[-1]["graph_slots"]["inputs"], {
            "left": "pred_step14_selected_id", "right": "second_step14_selected_id"})
        self.assertEqual(graph["contract"]["runtime_invariants"]["generated_sample_count"],
                         SAMPLES)
        self.assertFalse(graph["contract"]["runtime_invariants"]["complete_text_to_wav"])
        with np.load(CAPTURE) as oracle:
            self.assertEqual(oracle["code_ids"].shape, (2, 16))
            self.assertEqual(oracle["waveform_output"].shape, (1, SAMPLES))
            self.assertTrue(np.isfinite(oracle["waveform_output"]).all())

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; two-frame PCM NOT_TESTED")
    def test_pinned_generated_two_frame_pcm_and_recovery(self):
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
                Path(temp), graph, shapes,
                extra_sources=("src/kernels/concat_i32_checked.c",
                               "src/kernels/select_next_id_checked.c",
                               "src/kernels/repetition_penalty_checked.c") + CODEC_SOURCES)
            self.assertGreater(len(calls["operations"]), 3000)
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for name, item in weights.items():
                value = (_codec_value(codec, name) if name.startswith("codec.")
                         else np.ascontiguousarray(main.get_tensor(name).float().numpy()))
                self.assertEqual(value.shape, shapes[name], name)
                arena[item["abs_offset"]:item["abs_offset"] + value.nbytes] = value.tobytes()
            prepare_inputs(arena, acts)
            text_ids, codec_ids = pinned_prompt_ids(model_dir, request)
            view(arena, acts, "text_ids", np.int32, text_ids.shape)[:] = text_ids
            view(arena, acts, "codec_ids", np.int32, codec_ids.shape)[:] = codec_ids
            valid = view(arena, acts, "waveform_valid_samples", np.int32, (1,))
            pcm = view(arena, acts, "codec_waveform_output", np.float32, (1, SAMPLES))
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), SAMPLES)
            selected = np.empty((2, 16), dtype=np.int32)
            for codebook in range(16):
                name = ("packed_semantic_ids" if codebook == 0 else
                        f"packed_acoustic_ids_{codebook - 1}")
                selected[:, codebook] = view(arena, acts, name, np.int32, (2,))
            np.testing.assert_array_equal(selected, oracle["code_ids"])
            self.assertTrue(np.isfinite(pcm).all())
            delta = pcm.astype(np.float64) - oracle["waveform_output"].astype(np.float64)
            maximum = float(np.max(np.abs(delta)))
            self.assertLessEqual(maximum, 3e-6)
            print("QWEN3_TTS_TWO_FRAME_PCM " + json.dumps({
                "samples": SAMPLES, "sample_rate_hz": 24000,
                "max_abs": maximum,
                "rmse": float(np.sqrt(np.mean(delta * delta))),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "complete_text_to_wav": "NOT_TESTED",
            }, sort_keys=True))
            baseline = pcm.copy()
            # A rejected generated producer must not publish a prior sample extent.
            view(arena, acts, "second_bridge_left_indices", np.int32, (2,))[0] = 1
            self.assertNotEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 0)
            np.testing.assert_array_equal(pcm, baseline)
            view(arena, acts, "second_bridge_left_indices", np.int32, (2,))[0] = 0
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), SAMPLES)
            np.testing.assert_array_equal(pcm, baseline)


if __name__ == "__main__":
    unittest.main()
