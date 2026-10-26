"""Connected generated talker-to-predictor first-frame evidence."""

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
from version.v8.tts.build_qwen3_tts_talker_predictor_frame_circuit import (
    OUTPUT, bridge_plan, build_circuit,
)


class Qwen3TTSTalkerPredictorFrameTest(unittest.TestCase):
    def test_canonical_graph_and_independent_captures(self):
        graph = build_circuit()
        self.assertEqual(OUTPUT.read_text(), json.dumps(graph, indent=2) + "\n")
        self.assertEqual(bridge_plan(18), {"left_indices": [17, -1],
                                           "right_indices": [-1, 0]})
        bridge = graph["block_types"]["talker_predictor_bridge"]["body"]["ops"]
        self.assertEqual(bridge[0]["graph_slots"]["inputs"]["values"],
                         "talker_logits")
        self.assertEqual(bridge[1]["weight_refs"]["table"],
                         "talker.model.codec_embedding.weight")
        self.assertEqual(bridge[2]["graph_slots"]["outputs"]["output"],
                         "pred_block_input")
        self.assertEqual(graph["block_types"]["pred_first_block"]["body"]["ops"][0]
                         ["graph_slots"]["inputs"]["input"], "pred_block_input")
        self.assertEqual(len(weight_shapes(graph)), 401)
        identities = []
        for name in ("qwen3_tts_code_predictor", "qwen3_tts_talker_continuation",
                     "qwen3_tts_talker_prefill"):
            manifest = json.loads((FIXTURE / f"{name}.json").read_text())
            self.assertEqual(hashlib.sha256((FIXTURE / f"{name}.npz").read_bytes()).hexdigest(),
                             manifest["npz_sha256"])
            identities.append((manifest["model_revision"], manifest["reference_revision"],
                               manifest["model_files_sha256"],
                               manifest["reference_files_sha256"]))
        self.assertTrue(all(item == identities[0] for item in identities))

    def test_generated_synthetic_bridge_and_recovery(self):
        graph = build_circuit(context_tokens=1, talker_layers=1,
                              continuation_steps=2)
        with tempfile.TemporaryDirectory() as temp:
            fn, acts, weights, layout, calls, _ = compile_entry(
                Path(temp), graph, weight_shapes(graph))
            self.assertEqual(len(calls["operations"]), 354)
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)

            def weight(name, shape):
                return np.ndarray(shape, dtype=np.float32, buffer=arena,
                                  offset=weights[name]["abs_offset"])

            text = weight("talker.model.text_embedding.weight", (151936, 2048))
            text[:7, 0] = np.arange(1, 8) / 10
            codec = weight("talker.model.codec_embedding.weight", (3072, 1024))
            codec[:7, 0] = np.arange(1, 8) / 20
            codec[2, 0] = 0.75
            weight("talker.text_projection.linear_fc1.weight", (2048, 2048))[0, 0] = 1
            weight("talker.text_projection.linear_fc2.weight", (1024, 2048))[0, 0] = 1
            for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                weight("talker.model.layers.0." + suffix, (1024,))[:] = 1
            for suffix in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                weight("talker.model.layers.0." + suffix, (128,))[:] = 1
            weight("talker.model.norm.weight", (1024,))[:] = 1
            weight("talker.codec_head.weight", (3072, 1024))[2, 0] = 1
            for layer in range(5):
                for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                    weight(f"talker.code_predictor.model.layers.{layer}." + suffix,
                           (1024,))[:] = 1
                for suffix in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                    weight(f"talker.code_predictor.model.layers.{layer}." + suffix,
                           (128,))[:] = 1
            weight("talker.code_predictor.model.norm.weight", (1024,))[:] = 1
            view(arena, acts, "text_ids", np.int32, (7,))[:] = np.arange(7)
            codec_ids = view(arena, acts, "codec_ids", np.int32, (7,))
            codec_ids[:] = np.arange(7)
            for name, indices in prompt_plan(1).items():
                view(arena, acts, name, np.int32, (12,))[:] = indices
            for name, indices in bridge_plan(12).items():
                view(arena, acts, "bridge_" + name, np.int32, (2,))[:] = indices
            view(arena, acts, "pred_feedback_step_index_input", np.int32, (1,))[0] = 0
            valid = view(arena, acts, "pred_predictor_valid_code_count", np.int32, (1,))
            first = view(arena, acts, "talker_first_code_id", np.int32, (1,))
            seed = view(arena, acts, "pred_block_input", np.float32, (2, 1024))

            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(first[0]), 2)
            self.assertEqual(int(valid[0]), 3)
            self.assertEqual(float(seed[1, 0]), 0.75)
            np.testing.assert_array_equal(seed[0], view(
                arena, acts, "talker_final_norm", np.float32, (12, 1024))[-1])
            baseline = seed.copy()
            codec_ids[6] = 5
            self.assertEqual(fn(arena, size), 0)
            self.assertGreater(float(np.max(np.abs(seed - baseline))), 1e-5)
            codec_ids[6] = 6
            self.assertEqual(fn(arena, size), 0)
            np.testing.assert_array_equal(seed, baseline)
            codec_ids[6] = 3072
            self.assertNotEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 0)
            np.testing.assert_array_equal(seed, baseline)
            codec_ids[6] = 6
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 3)
            np.testing.assert_array_equal(seed, baseline)
            left = view(arena, acts, "bridge_left_indices", np.int32, (2,))
            left[0] = 10
            self.assertEqual(fn(arena, size), 0)
            self.assertGreater(float(np.max(np.abs(seed - baseline))), 1e-5)
            left[0] = 11
            self.assertEqual(fn(arena, size), 0)
            np.testing.assert_array_equal(seed, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; connected frame NOT_TESTED")
    def test_pinned_generated_talker_to_predictor_frame(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        graph = build_circuit()
        shapes = weight_shapes(graph)
        with np.load(FIXTURE / "qwen3_tts_customvoice_reference.npz") as request, \
             np.load(FIXTURE / "qwen3_tts_code_predictor.npz") as predictor, \
             np.load(FIXTURE / "qwen3_tts_talker_continuation.npz") as talker, \
             tempfile.TemporaryDirectory() as temp:
            fn, acts, weights, layout, calls, library = compile_entry(
                Path(temp), graph, shapes)
            self.assertEqual(len(calls["operations"]), 2142)
            functions = [call["function"] for call in calls["operations"]]
            self.assertEqual(functions.count("audio_argmax_last_row_checked_f32"), 16)
            self.assertEqual(functions.count("audio_text_embedding_token_major_f32"), 15)
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source:
                for name, item in weights.items():
                    tensor = source.get_tensor(name).float().numpy()
                    self.assertEqual(tensor.shape, shapes[name])
                    arena[item["abs_offset"]:item["abs_offset"] + tensor.nbytes] = tensor.tobytes()
            text_ids, codec_ids = pinned_prompt_ids(model_dir, request)
            view(arena, acts, "text_ids", np.int32, text_ids.shape)[:] = text_ids
            view(arena, acts, "codec_ids", np.int32, codec_ids.shape)[:] = codec_ids
            for name, indices in prompt_plan(7).items():
                view(arena, acts, name, np.int32, (18,))[:] = indices
            for name, indices in bridge_plan(18).items():
                view(arena, acts, "bridge_" + name, np.int32, (2,))[:] = indices
            view(arena, acts, "pred_feedback_step_index_input", np.int32, (1,))[0] = 0
            valid = view(arena, acts, "pred_predictor_valid_code_count", np.int32, (1,))
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 15)
            first = view(arena, acts, "talker_first_code_id", np.int32, (1,))
            self.assertEqual(int(first[0]), int(predictor["talker_first_code_id"][0, 0]))
            self.assertEqual(int(first[0]), int(np.argmax(talker["step0_logits"][0, -1])))
            seed = view(arena, acts, "pred_block_input", np.float32, (2, 1024))
            expected_seed = predictor["call0_model_inputs_embeds"][0]
            seed_error = float(np.max(np.abs(seed - expected_seed)))
            self.assertLessEqual(seed_error, 3e-4)
            names = ["pred_predictor_codebook0_selected_id"] + [
                f"pred_step{i}_selected_id" for i in range(1, 15)]
            selected = [int(view(arena, acts, name, np.int32, (1,))[0])
                        for name in names]
            expected = [int(predictor[f"call{i}_selected_id"][0])
                        for i in range(15)]
            self.assertEqual(selected, expected)
            self.assertEqual(int(valid[0]), 15)
            print("QWEN3_TTS_TALKER_PREDICTOR_FRAME " + json.dumps({
                "first_code_id": int(first[0]), "selected_ids": selected,
                "seed_max_abs": seed_error, "valid_codes": int(valid[0]),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "codec_to_wav": "NOT_TESTED", "text_to_wav": "NOT_TESTED",
            }, sort_keys=True))

            # A malformed row plan rejects before the predictor and must not
            # publish a complete frame; recovery uses the same loaded entry.
            left = view(arena, acts, "bridge_left_indices", np.int32, (2,))
            left[0] = 18
            self.assertNotEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 0)
            left[0] = 17
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 15)
            self.assertEqual([int(view(arena, acts, name, np.int32, (1,))[0])
                              for name in names], selected)


if __name__ == "__main__":
    unittest.main()
