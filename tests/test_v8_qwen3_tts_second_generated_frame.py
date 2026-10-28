"""Second Qwen speech-code frame from CKE's first generated frame."""

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
from version.v8.tts.build_qwen3_tts_prompt_first_block_circuit import prompt_plan
from version.v8.tts.build_qwen3_tts_talker_predictor_frame_circuit import bridge_plan
from version.v8.tts.build_qwen3_tts_second_generated_frame_circuit import (
    OUTPUT, build_circuit,
)

CAPTURE = FIXTURE / "qwen3_tts_talker_continuation.npz"
MANIFEST = FIXTURE / "qwen3_tts_talker_continuation.json"


def prepare_inputs(arena, acts):
    view(arena, acts, "text_ids", np.int32, (13,))[:] = 0
    view(arena, acts, "codec_ids", np.int32, (7,))[:] = 0
    for name, indices in prompt_plan(7).items():
        view(arena, acts, name, np.int32, (18,))[:] = indices
    for name, indices in bridge_plan(18).items():
        view(arena, acts, "bridge_" + name, np.int32, (2,))[:] = indices
    view(arena, acts, "pred_feedback_step_index_input", np.int32, (1,))[0] = 0
    view(arena, acts, "feedback_left_indices", np.int32, (1,))[0] = 0
    view(arena, acts, "feedback_right_indices", np.int32, (1,))[0] = 3
    view(arena, acts, "talker_step_index_input", np.int32, (1,))[0] = 0
    view(arena, acts, "second_bridge_left_indices", np.int32, (2,))[:] = [0, -1]
    view(arena, acts, "second_bridge_right_indices", np.int32, (2,))[:] = [-1, 0]
    view(arena, acts, "second_feedback_step_index_input", np.int32, (1,))[0] = 0


class Qwen3TTSSecondGeneratedFrameTest(unittest.TestCase):
    def test_canonical_edges_and_capture_integrity(self):
        graph = build_circuit()
        self.assertEqual(OUTPUT.read_text(), json.dumps(graph, indent=2) + "\n")
        self.assertEqual(hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
                         json.loads(MANIFEST.read_text())["npz_sha256"])
        blocks = graph["sequence"]
        self.assertLess(blocks.index("publish_talker_continuation"),
                        blocks.index("second_talker_predictor_bridge"))
        self.assertLess(blocks.index("second_talker_predictor_bridge"),
                        blocks.index("second_first_block"))
        bridge = graph["block_types"]["second_talker_predictor_bridge"]["body"]["ops"]
        self.assertEqual(bridge[0]["kernel"],
                         "repetition_penalty_last_row_checked_f32")
        self.assertEqual(bridge[0]["graph_slots"]["inputs"],
                         {"values": "cont_talker_logits",
                          "history": "external:talker_history_ids"})
        self.assertEqual(bridge[0]["runtime_scalar_bindings"],
                         {"history_count": "history_valid_count"})
        self.assertEqual(bridge[0]["params"]["call_constants"]
                         ["penalty_numerator"], 105)
        self.assertEqual(bridge[0]["params"]["call_constants"]
                         ["penalty_denominator"], 100)
        self.assertEqual(bridge[1]["kernel"], "select_next_id_checked_f32")
        self.assertEqual(bridge[1]["runtime_scalar_bindings"],
                         {"emitted_frames": "history_valid_count"})
        self.assertEqual({name: bridge[1]["params"]["call_constants"][name]
                          for name in ("min_frames", "max_frames",
                                       "stop_id", "suppressed_begin", "suppressed_end")},
                         {"min_frames": 2, "max_frames": 64,
                          "stop_id": 2150, "suppressed_begin": 2048,
                          "suppressed_end": 3072})
        self.assertEqual(bridge[1]["graph_slots"]["inputs"]["values"],
                         "second_talker_adjusted_logits")
        self.assertEqual(bridge[-1]["graph_slots"]["inputs"]["left"],
                         "cont_talker_final_norm")
        self.assertEqual(graph["runtime_lengths"]["second_decode_position"]["producer"],
                         "second_feedback_check_position")
        self.assertEqual(graph["contract"]["runtime_invariants"]["complete_text_to_wav"],
                         False)

    def test_generated_second_frame_rejection_and_recovery_without_assets(self):
        graph = build_circuit(talker_layers=1, continuation_steps=2)
        with tempfile.TemporaryDirectory() as temp:
            fn, acts, _, layout, _, _ = compile_entry(
                Path(temp), graph, weight_shapes(graph),
                extra_sources=("src/kernels/select_next_id_checked.c",
                               "src/kernels/repetition_penalty_checked.c",
                               "src/kernels/copy_i32_span_checked.c",
                               "src/kernels/equal_i32_scalar_checked.c"))
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            prepare_inputs(arena, acts)
            first_valid = view(arena, acts, "talker_continuation_valid", np.int32, (1,))
            second_valid = view(arena, acts, "second_predictor_valid_code_count",
                                np.int32, (1,))
            second_id = view(arena, acts, "second_talker_first_code_id", np.int32, (1,))
            last_id = view(arena, acts, "second_step2_selected_id", np.int32, (1,))
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(first_valid[0]), 1)
            self.assertEqual(int(second_valid[0]), 3)
            self.assertEqual(int(second_id[0]), 0)
            self.assertEqual(int(last_id[0]), 0)
            bridge = view(arena, acts, "second_bridge_left_indices", np.int32, (2,))
            last_id[0] = 73
            bridge[0] = 1  # one-row talker output cannot supply row one
            self.assertNotEqual(fn(arena, size), 0)
            self.assertEqual(int(second_valid[0]), 0)
            self.assertEqual(int(last_id[0]), 73)
            bridge[0] = 0
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(second_valid[0]), 3)
            self.assertEqual(int(last_id[0]), 0)
            step = view(arena, acts, "talker_step_index_input", np.int32, (1,))
            step[0] = 3
            self.assertNotEqual(fn(arena, size), 0)
            self.assertEqual(int(second_valid[0]), 0)
            step[0] = 0
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(second_valid[0]), 3)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; second frame NOT_TESTED")
    def test_pinned_second_generated_frame(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        pinned_talker = json.loads((model_dir / "config.json").read_text())["talker_config"]
        self.assertEqual((pinned_talker["vocab_size"],
                          pinned_talker["codec_eos_token_id"]), (3072, 2150))
        graph = build_circuit()
        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source, \
             np.load(FIXTURE / "qwen3_tts_customvoice_reference.npz") as request, \
             np.load(CAPTURE) as oracle, tempfile.TemporaryDirectory() as temp:
            fn, acts, weights, layout, calls, library = compile_entry(
                Path(temp), graph, weight_shapes(graph),
                extra_sources=("src/kernels/select_next_id_checked.c",
                               "src/kernels/repetition_penalty_checked.c",
                               "src/kernels/copy_i32_span_checked.c",
                               "src/kernels/equal_i32_scalar_checked.c"))
            self.assertGreater(len(calls["operations"]), 3000)
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
            for name, value in (("pred_feedback_step_index_input", 0),
                                ("feedback_left_indices", 0),
                                ("feedback_right_indices", 3),
                                ("talker_step_index_input", 0),
                                ("second_feedback_step_index_input", 0)):
                view(arena, acts, name, np.int32, (1,))[0] = value
            view(arena, acts, "second_bridge_left_indices", np.int32, (2,))[:] = [0, -1]
            view(arena, acts, "second_bridge_right_indices", np.int32, (2,))[:] = [-1, 0]
            self.assertEqual(fn(arena, size), 0)
            second_id = int(view(arena, acts, "second_talker_first_code_id",
                                 np.int32, (1,))[0])
            self.assertEqual(second_id, 215)
            self.assertEqual(second_id, int(np.argmax(oracle["step1_logits"])))
            adjusted = view(arena, acts, "second_talker_adjusted_logits",
                            np.float32, (3072,))
            expected_adjusted = oracle["step1_logits"].astype(np.float32).reshape(-1).copy()
            first_id = int(view(arena, acts, "talker_first_code_id",
                                np.int32, (1,))[0])
            self.assertEqual(first_id, int(request["codec_tokens"][0, 0]))
            expected_adjusted[first_id] = (expected_adjusted[first_id] *
                np.float32(1.05) if expected_adjusted[first_id] < 0 else
                expected_adjusted[first_id] / np.float32(1.05))
            adjusted_error = float(np.max(np.abs(adjusted - expected_adjusted)))
            self.assertLessEqual(adjusted_error, 1e-4)
            selected = ["second_talker_first_code_id",
                        "second_predictor_codebook0_selected_id"] + [
                f"second_step{i}_selected_id" for i in range(1, 15)]
            codes = [int(view(arena, acts, name, np.int32, (1,))[0])
                     for name in selected]
            np.testing.assert_array_equal(codes, request["codec_tokens"][1])
            self.assertEqual(int(view(arena, acts,
                                      "second_predictor_valid_code_count",
                                      np.int32, (1,))[0]), 15)
            expected = np.vstack((
                oracle["step1_norm"].reshape(1, 1024),
                source.get_tensor("talker.model.codec_embedding.weight")
                      [second_id].float().numpy().reshape(1, 1024),
            ))
            actual = view(arena, acts, "second_pred_block_input", np.float32,
                          (2, 1024))
            bridge_error = float(np.max(np.abs(actual.astype(np.float64) -
                                               expected.astype(np.float64))))
            self.assertLessEqual(bridge_error, 1e-4)
            print("QWEN3_TTS_SECOND_GENERATED_FRAME " + json.dumps({
                "second_codes": codes, "bridge_max_abs": bridge_error,
                "adjusted_logits_max_abs": adjusted_error,
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "complete_text_to_wav": "NOT_TESTED",
            }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
