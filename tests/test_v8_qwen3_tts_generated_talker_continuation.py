"""Generated Qwen codebook feedback into the first talker continuation."""

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
from version.v8.tts.build_qwen3_tts_generated_talker_continuation_circuit import (
    OUTPUT, build_circuit,
)
from version.v8.tts.build_qwen3_tts_prompt_first_block_circuit import prompt_plan
from version.v8.tts.build_qwen3_tts_talker_predictor_frame_circuit import bridge_plan

CAPTURE = FIXTURE / "qwen3_tts_talker_continuation.npz"
MANIFEST = FIXTURE / "qwen3_tts_talker_continuation.json"


class Qwen3TTSGeneratedTalkerContinuationTest(unittest.TestCase):
    def test_declared_edges_and_capture_integrity_without_assets(self):
        graph = build_circuit()
        self.assertEqual(OUTPUT.read_text(), json.dumps(graph, indent=2) + "\n")
        self.assertEqual(hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
                         json.loads(MANIFEST.read_text())["npz_sha256"])
        self.assertLess(graph["sequence"].index("talker_cache_seed"),
                        graph["sequence"].index("talker_predictor_bridge"))
        self.assertLess(graph["sequence"].index("talker_code_feedback"),
                        graph["sequence"].index("cont_first_block"))
        ops = graph["block_types"]["talker_code_feedback"]["body"]["ops"]
        self.assertEqual(sum(op["op"] == "add_rows_checked" for op in ops), 15)
        self.assertEqual(ops[0]["weight_refs"]["table"],
                         "talker.code_predictor.model.codec_embedding.14.weight")
        self.assertEqual(ops[-1]["graph_slots"]["inputs"]["right"],
                         "projection_output")
        self.assertEqual(graph["block_types"]["cont_first_block"]["body"]["ops"][0]
                         ["graph_slots"]["inputs"]["input"], "feedback_next_input")
        self.assertEqual(graph["runtime_lengths"]["talker_decode_position"]["capacity"], 20)
        for block_name in ("cont_first_block", "cont_talker_head"):
            for op in graph["block_types"][block_name]["body"]["ops"]:
                if op["op"] == "rope_qk_bounded":
                    self.assertEqual(op["runtime_scalar_bindings"],
                                     {"pos_offset": "talker_decode_position"})
                    self.assertNotIn("pos_offset", op["params"]["call_constants"])
                if op["op"] == "attention_decode_append_checked":
                    self.assertEqual(op["runtime_scalar_bindings"],
                                     {"position": "talker_decode_position"})
                    self.assertNotIn("position", op["params"]["call_constants"])

    def test_asset_free_generated_feedback_and_recovery(self):
        graph = build_circuit(talker_layers=1, continuation_steps=2)
        with tempfile.TemporaryDirectory() as temp:
            fn, acts, weights, layout, _, _ = compile_entry(
                Path(temp), graph, weight_shapes(graph))
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for name, value in (
                ("talker.model.codec_embedding.weight", 0.1),
                ("talker.code_predictor.model.codec_embedding.0.weight", 0.2),
                ("talker.code_predictor.model.codec_embedding.1.weight", 0.3),
                ("talker.code_predictor.model.codec_embedding.2.weight", 0.4),
            ):
                table = np.ndarray(weight_shapes(graph)[name], dtype=np.float32,
                                   buffer=arena, offset=weights[name]["abs_offset"])
                table[0, 0] = value
            view(arena, acts, "text_ids", np.int32, (13,))[:] = 0
            view(arena, acts, "codec_ids", np.int32, (7,))[:] = 0
            for name, indices in prompt_plan(7).items():
                view(arena, acts, name, np.int32, (18,))[:] = indices
            for name, indices in bridge_plan(18).items():
                view(arena, acts, "bridge_" + name, np.int32, (2,))[:] = indices
            view(arena, acts, "pred_feedback_step_index_input", np.int32, (1,))[0] = 0
            view(arena, acts, "feedback_left_indices", np.int32, (1,))[0] = 0
            view(arena, acts, "feedback_right_indices", np.int32, (1,))[0] = 3
            step = view(arena, acts, "talker_step_index_input", np.int32, (1,))
            step[0] = 0
            valid = view(arena, acts, "talker_continuation_valid", np.int32, (1,))
            logits = view(arena, acts, "cont_talker_logits", np.float32, (1, 3072))
            cache_k = view(arena, acts, "talker_layer0_cache_k", np.float32,
                           (8, 21, 128))
            cache_k[:, 18:, :] = 7
            logits[:] = -719
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 1)
            self.assertFalse(np.all(cache_k[:, 18, :] == 7))
            np.testing.assert_array_equal(cache_k[:, 19:, :], 7)
            self.assertAlmostEqual(float(view(
                arena, acts, "feedback_next_input", np.float32, (1, 1024))[0, 0]),
                1.0, places=6)
            baseline = logits.copy()
            for index in (1, 2):
                step[0] = index
                self.assertEqual(fn(arena, size), 0)
                self.assertFalse(np.all(cache_k[:, 18 + index, :] == 7))
                if index < 2:
                    np.testing.assert_array_equal(cache_k[:, 19 + index:, :], 7)
                self.assertEqual(int(valid[0]), 1)
                self.assertTrue(np.isfinite(logits).all())
            previous = logits.copy()
            for invalid in (-1, 3):
                step[0] = invalid
                self.assertNotEqual(fn(arena, size), 0)
                self.assertEqual(int(valid[0]), 0)
                np.testing.assert_array_equal(logits, previous)
            step[0] = 0
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 1)
            np.testing.assert_array_equal(logits, baseline)
            table = np.ndarray((2048, 1024), dtype=np.float32, buffer=arena,
                               offset=weights[
                                   "talker.code_predictor.model.codec_embedding.2.weight"
                               ]["abs_offset"])
            table[0, 0] = np.nan
            self.assertNotEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 0)
            np.testing.assert_array_equal(logits, baseline)
            table[0, 0] = 0.4
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 1)
            np.testing.assert_array_equal(logits, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; generated continuation NOT_TESTED")
    def test_pinned_generated_first_continuation_and_recovery(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        graph = build_circuit()
        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source, \
             np.load(FIXTURE / "qwen3_tts_customvoice_reference.npz") as request, \
             np.load(CAPTURE) as oracle, tempfile.TemporaryDirectory() as temp:
            fn, acts, weights, layout, calls, library = compile_entry(
                Path(temp), graph, weight_shapes(graph))
            self.assertGreater(len(calls["operations"]), 2700)
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for name, item in weights.items():
                value = np.ascontiguousarray(source.get_tensor(name).float().numpy())
                self.assertEqual(value.shape, weight_shapes(graph)[name], name)
                arena[item["abs_offset"]:item["abs_offset"] + value.nbytes] = value.tobytes()
            text_ids, codec_ids = pinned_prompt_ids(model_dir, request)
            view(arena, acts, "text_ids", np.int32, text_ids.shape)[:] = text_ids
            view(arena, acts, "codec_ids", np.int32, codec_ids.shape)[:] = codec_ids
            for name, indices in prompt_plan(7).items():
                view(arena, acts, name, np.int32, (18,))[:] = indices
            for name, indices in bridge_plan(18).items():
                view(arena, acts, "bridge_" + name, np.int32, (2,))[:] = indices
            view(arena, acts, "pred_feedback_step_index_input", np.int32, (1,))[0] = 0
            view(arena, acts, "feedback_left_indices", np.int32, (1,))[0] = 0
            view(arena, acts, "feedback_right_indices", np.int32, (1,))[0] = 3
            view(arena, acts, "talker_step_index_input", np.int32, (1,))[0] = 0
            self.assertEqual(fn(arena, size), 0)
            valid = view(arena, acts, "talker_continuation_valid", np.int32, (1,))
            self.assertEqual(int(valid[0]), 1)
            selected = ["talker_first_code_id", "pred_predictor_codebook0_selected_id"] + [
                f"pred_step{i}_selected_id" for i in range(1, 15)]
            ids = [int(view(arena, acts, name, np.int32, (1,))[0]) for name in selected]
            np.testing.assert_array_equal(ids, request["codec_tokens"][0])
            next_input = view(arena, acts, "feedback_next_input", np.float32, (1, 1024))
            input_delta = next_input.astype(np.float64) - oracle["step1_input"][0].astype(np.float64)
            input_error = float(np.max(np.abs(input_delta)))
            self.assertLessEqual(input_error, 4e-5)
            logits = view(arena, acts, "cont_talker_logits", np.float32, (1, 3072))
            logit_delta = logits.astype(np.float64) - oracle["step1_logits"][0].astype(np.float64)
            logit_error = float(np.max(np.abs(logit_delta)))
            self.assertLessEqual(logit_error, 2e-4)
            self.assertEqual(int(np.argmax(logits)),
                             int(np.argmax(oracle["step1_logits"])))
            baseline = logits.copy()
            # Failure in the generated feedback must prevent continuation and
            # leave the previously valid output intact.
            table = np.ndarray((2048, 1024), dtype=np.float32, buffer=arena,
                               offset=weights[
                                   "talker.code_predictor.model.codec_embedding.14.weight"
                               ]["abs_offset"])
            saved = float(table[ids[-1], 0])
            table[ids[-1], 0] = np.nan
            self.assertNotEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 0)
            np.testing.assert_array_equal(logits, baseline)
            table[ids[-1], 0] = saved
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 1)
            np.testing.assert_array_equal(logits, baseline)
            print("QWEN3_TTS_GENERATED_TALKER_CONTINUATION " + json.dumps({
                "generated_codes": ids, "input_max_abs": input_error,
                "logits_max_abs": logit_error,
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "complete_text_to_wav": "NOT_TESTED",
            }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
