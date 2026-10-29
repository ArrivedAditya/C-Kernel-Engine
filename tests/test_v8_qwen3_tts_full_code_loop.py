"""Pinned Qwen3-TTS natural-EOS code trajectory through generated entries."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from tests.test_v8_checked_frame_loop import Loop, run_function
from tests.test_v8_qwen3_tts_prompt_first_block import (
    FIXTURE, compile_entry, pinned_prompt_ids, view,
)
from tests.test_v8_qwen3_tts_second_generated_frame import prepare_inputs
from tests.test_v8_qwen3_tts_talker_prefill import weight_shapes
from tests.test_v8_qwen3_tts_two_phase_speech import SOURCES, segment_function
from version.v8.tts.build_qwen3_tts_second_generated_frame_circuit import build_circuit


class Qwen3TTSFullCodeLoopTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; natural EOS NOT_TESTED")
    def test_pinned_natural_eos_and_complete_codes(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        graph = build_circuit(max_generated_frames=64)
        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source, \
             np.load(FIXTURE / "qwen3_tts_customvoice_reference.npz") as oracle, \
             tempfile.TemporaryDirectory() as temp:
            _, acts, weights, layout, _, library = compile_entry(
                Path(temp), graph, weight_shapes(graph),
                extra_sources=(*SOURCES, "src/runtime/checked_frame_loop.c"))
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for name, item in weights.items():
                value = np.ascontiguousarray(source.get_tensor(name).float().numpy())
                arena[item["abs_offset"]:item["abs_offset"] + value.nbytes] = value.tobytes()
            text_ids, codec_ids = pinned_prompt_ids(model_dir, oracle)
            prepare_inputs(arena, acts)
            view(arena, acts, "text_ids", np.int32, text_ids.shape)[:] = text_ids
            view(arena, acts, "codec_ids", np.int32, codec_ids.shape)[:] = codec_ids
            first_names = ["talker_first_code_id", "pred_predictor_codebook0_selected_id"] + [
                f"pred_step{i}_selected_id" for i in range(1, 15)]
            frame_names = ["second_talker_first_code_id",
                           "second_predictor_codebook0_selected_id"] + [
                f"second_step{i}_selected_id" for i in range(1, 15)]
            first_offsets = (ctypes.c_size_t * 16)(
                *(acts[name]["abs_offset"] for name in first_names))
            frame_offsets = (ctypes.c_size_t * 16)(
                *(acts[name]["abs_offset"] for name in frame_names))
            staging = np.full(64 * 16, -901, np.int32)
            output = np.full(64 * 16, -901, np.int32)
            valid = ctypes.c_size_t(99)
            request = Loop(
                ctypes.cast(segment_function(library, graph, 0), ctypes.c_void_p),
                ctypes.cast(segment_function(library, graph, 1), ctypes.c_void_p),
                arena, size, acts["talker_step_index_input"]["abs_offset"],
                acts["pred_predictor_valid_code_count"]["abs_offset"],
                acts["second_predictor_valid_code_count"]["abs_offset"],
                first_offsets, frame_offsets, 16, 64,
                staging.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 64 * 16,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 64 * 16,
                ctypes.pointer(valid))
            run = run_function(library)
            request.arena_bytes = size - 1
            self.assertEqual(run(ctypes.byref(request)), -2)
            self.assertEqual(valid.value, 0)
            np.testing.assert_array_equal(output, -901)
            request.arena_bytes = size
            start = time.monotonic()
            self.assertEqual(run(ctypes.byref(request)), 0)
            seconds = time.monotonic() - start
            reference = oracle["codec_tokens"].astype(np.int32)
            self.assertEqual(reference.shape, (46, 16))
            self.assertEqual(valid.value, len(reference))
            np.testing.assert_array_equal(output[:reference.size].reshape(reference.shape),
                                          reference)
            np.testing.assert_array_equal(output[reference.size:], -901)
            self.assertEqual(int(view(arena, acts, "second_talker_first_code_id",
                                      np.int32, (1,))[0]), 2150)
            self.assertEqual(int(view(arena, acts, "generation_state_valid",
                                      np.int32, (1,))[0]), 0)
            print("QWEN3_TTS_NATURAL_EOS " + json.dumps({
                "frames": int(valid.value), "codebooks": 16,
                "first_code_mismatch": None, "generation_seconds": round(seconds, 2),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "complete_text_to_wav": False,
            }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
