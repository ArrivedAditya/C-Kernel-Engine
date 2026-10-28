"""Qwen's first two speech frames across two generated entry functions."""

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
from tests.test_v8_qwen3_tts_talker_prefill import weight_shapes
from tests.test_v8_qwen3_tts_second_generated_frame import (
    prepare_inputs,
)
from version.v8.tts.build_qwen3_tts_second_generated_frame_circuit import (
    OUTPUT, build_circuit,
)

SOURCES = ("src/kernels/select_next_id_checked.c",
           "src/kernels/repetition_penalty_checked.c")


def segment_function(library: Path, graph: dict, index: int):
    native = ctypes.CDLL(str(library))
    fn = getattr(native, graph["native_entry"]["segments"][index]["function"])
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    fn.restype = ctypes.c_int
    return fn


class Qwen3TTSTwoPhaseSpeechTest(unittest.TestCase):
    def test_generated_state_and_recovery_without_assets(self):
        graph = build_circuit(talker_layers=1, continuation_steps=2)
        with tempfile.TemporaryDirectory() as temp:
            full, acts, _, layout, calls, library = compile_entry(
                Path(temp), graph, weight_shapes(graph), extra_sources=SOURCES)
            self.assertEqual(calls["entry"]["segments"],
                             graph["native_entry"]["segments"])
            prefill = segment_function(library, graph, 0)
            frame = segment_function(library, graph, 1)
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            prepare_inputs(arena, acts)
            first_valid = view(arena, acts, "talker_continuation_valid",
                               np.int32, (1,))
            second_valid = view(arena, acts, "second_predictor_valid_code_count",
                                np.int32, (1,))
            self.assertNotEqual(frame(arena, size), 0)
            self.assertEqual(int(second_valid[0]), 0)
            for _ in range(2):
                self.assertEqual(prefill(arena, size), 0)
                self.assertEqual(int(first_valid[0]), 1)
                self.assertEqual(int(second_valid[0]), 0)
                self.assertEqual(frame(arena, size), 0)
                self.assertEqual(int(second_valid[0]), 3)
            bridge = view(arena, acts, "second_bridge_left_indices",
                          np.int32, (2,))
            bridge[0] = 1
            self.assertNotEqual(frame(arena, size), 0)
            self.assertEqual(int(second_valid[0]), 0)
            bridge[0] = 0
            self.assertEqual(frame(arena, size), 0)
            self.assertEqual(int(second_valid[0]), 3)
            self.assertEqual(full(arena, size), 0)  # original ABI composes both
            self.assertEqual(int(second_valid[0]), 3)
            step = view(arena, acts, "talker_step_index_input", np.int32, (1,))
            step[0] = 3
            self.assertNotEqual(prefill(arena, size), 0)
            self.assertEqual(int(first_valid[0]), 0)
            self.assertEqual(int(second_valid[0]), 0)
            self.assertNotEqual(frame(arena, size), 0)
            self.assertEqual(int(second_valid[0]), 0)
            self.assertNotEqual(full(arena, size), 0)
            self.assertEqual(int(first_valid[0]), 0)
            self.assertEqual(int(second_valid[0]), 0)
            step[0] = 0
            self.assertEqual(prefill(arena, size), 0)
            self.assertEqual(frame(arena, size), 0)
            self.assertEqual(int(second_valid[0]), 3)

    def test_canonical_graph_and_capture_integrity(self):
        graph = build_circuit()
        self.assertEqual(OUTPUT.read_text(), json.dumps(graph, indent=2) + "\n")
        manifest = json.loads((FIXTURE / "qwen3_tts_talker_continuation.json").read_text())
        capture = FIXTURE / "qwen3_tts_talker_continuation.npz"
        self.assertEqual(hashlib.sha256(capture.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        self.assertFalse(graph["contract"]["runtime_invariants"]
                         ["repeatable_generation_step"])

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; two-phase parity NOT_TESTED")
    def test_pinned_second_frame_across_entries(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        graph = build_circuit()
        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source, \
             np.load(FIXTURE / "qwen3_tts_customvoice_reference.npz") as request, \
             tempfile.TemporaryDirectory() as temp:
            _, acts, weights, layout, calls, library = compile_entry(
                Path(temp), graph, weight_shapes(graph), extra_sources=SOURCES)
            prefill = segment_function(library, graph, 0)
            frame = segment_function(library, graph, 1)
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for name, item in weights.items():
                value = np.ascontiguousarray(source.get_tensor(name).float().numpy())
                arena[item["abs_offset"]:item["abs_offset"] + value.nbytes] = value.tobytes()
            text_ids, codec_ids = pinned_prompt_ids(model_dir, request)
            view(arena, acts, "text_ids", np.int32, text_ids.shape)[:] = text_ids
            view(arena, acts, "codec_ids", np.int32, codec_ids.shape)[:] = codec_ids
            prepare_inputs(arena, acts)
            # prepare_inputs initializes integer inputs to zeros; restore the
            # pinned prompt after setting bridge and step-control indices.
            view(arena, acts, "text_ids", np.int32, text_ids.shape)[:] = text_ids
            view(arena, acts, "codec_ids", np.int32, codec_ids.shape)[:] = codec_ids
            self.assertEqual(prefill(arena, size), 0)
            self.assertEqual(frame(arena, size), 0)
            self.assertEqual(int(view(arena, acts, "talker_continuation_valid",
                                      np.int32, (1,))[0]), 1)
            self.assertEqual(int(view(arena, acts,
                                      "second_predictor_valid_code_count",
                                      np.int32, (1,))[0]), 15)
            names = ["second_talker_first_code_id",
                     "second_predictor_codebook0_selected_id"] + [
                f"second_step{i}_selected_id" for i in range(1, 15)]
            codes = [int(view(arena, acts, name, np.int32, (1,))[0])
                     for name in names]
            np.testing.assert_array_equal(codes, request["codec_tokens"][1])
            print("QWEN3_TTS_TWO_PHASE_FRAME " + json.dumps({
                "codes": codes, "arena_bytes": size,
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "repeatable_generation_step": "NOT_TESTED",
            }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
