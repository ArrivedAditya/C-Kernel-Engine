"""Checked generic feedback providers and generated Qwen predictor connection."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_code_predictor_prefill import _arena, CAPTURE, MANIFEST
from tests.test_v8_qwen3_tts_prompt_first_block import compile_entry, view
from tests.test_v8_qwen3_tts_talker_prefill import weight_shapes
from version.v8.tts.build_qwen3_tts_predictor_feedback_circuit import (
    OUTPUT, build_circuit,
)
from version.v8.tts.build_qwen3_tts_predictor_full_frame_circuit import (
    OUTPUT as FULL_OUTPUT, build_circuit as full_frame_circuit,
)

ROOT = Path(__file__).resolve().parents[1]


def _direct_library(directory: Path):
    library = directory / "feedback_providers.so"
    subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                    "-I", str(ROOT / "include"),
                    str(ROOT / "src/kernels/audio_text_embedding.c"),
                    "-o", str(library)], check=True, capture_output=True, text=True)
    loaded = ctypes.CDLL(str(library))
    pfloat = ctypes.POINTER(ctypes.c_float)
    pint = ctypes.POINTER(ctypes.c_int32)
    size = ctypes.c_size_t
    argmax = loaded.audio_argmax_last_row_checked_f32
    argmax.argtypes = [pfloat, size, size, size, size, pint, size]
    argmax.restype = ctypes.c_int
    lookup = loaded.audio_text_embedding_token_major_f32
    lookup.argtypes = [pint, size, pfloat, size, size, size, size,
                       pfloat, size, size, size]
    lookup.restype = ctypes.c_int
    return argmax, lookup


def _ptr(array, dtype):
    return array.ctypes.data_as(ctypes.POINTER(dtype))


class Qwen3TTSPredictorFeedbackTest(unittest.TestCase):
    def test_primitive_boundaries_and_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            argmax, lookup = _direct_library(Path(temp))
            values = np.array([[40, 41, 42, -701, -701],
                               [3, 8, 8, -701, -701]], dtype=np.float32)
            selected = np.array([-731], dtype=np.int32)

            def choose(rows=2, channels=3, stride=5, capacity=10):
                return argmax(_ptr(values, ctypes.c_float), capacity,
                              rows, channels, stride,
                              _ptr(selected, ctypes.c_int32), 1)

            self.assertEqual(choose(), 0)
            self.assertEqual(int(selected[0]), 1)  # first tie in the last row
            for changes, arguments in (
                ({(1, 1): np.nan}, {}),
                ({}, {"capacity": 7}),
                ({}, {"rows": 0}),
            ):
                with self.subTest(changes=changes, arguments=arguments):
                    values[1, 1] = 8
                    for index, value in changes.items():
                        values[index] = value
                    selected[0] = -731
                    self.assertNotEqual(choose(**arguments), 0)
                    self.assertEqual(int(selected[0]), -731)
            values[1, 1] = 8
            self.assertEqual(choose(), 0)
            self.assertEqual(int(selected[0]), 1)

            table = np.array([[0, 1, 2, 3, -907, -907],
                              [4, 5, 6, 7, -907, -907],
                              [8, 9, 10, 11, -907, -907]], dtype=np.float32)
            ids = np.array([2, 0], dtype=np.int32)
            output = np.full((2, 5), -719, dtype=np.float32)

            def embed(id_capacity=2, table_capacity=18, output_capacity=10):
                return lookup(_ptr(ids, ctypes.c_int32), id_capacity,
                              _ptr(table, ctypes.c_float), table_capacity,
                              3, 4, 6, _ptr(output, ctypes.c_float),
                              output_capacity, 2, 5)

            self.assertEqual(embed(), 0)
            np.testing.assert_array_equal(output[:, :4], table[ids, :4])
            np.testing.assert_array_equal(output[:, 4], [-719, -719])
            for change, capacities in (
                ("invalid_id", {}), ("nonfinite_row", {}),
                ("short_ids", {"id_capacity": 1}),
                ("short_table", {"table_capacity": 15}),
                ("short_output", {"output_capacity": 8}),
            ):
                with self.subTest(change=change):
                    output[:] = -719
                    ids[:] = [2, 0]
                    table[2, 1] = 9
                    if change == "invalid_id":
                        ids[1] = 3
                    if change == "nonfinite_row":
                        table[2, 1] = np.nan
                    self.assertNotEqual(embed(**capacities), 0)
                    self.assertTrue(np.all(output == -719))
            ids[:] = [2, 0]
            table[2, 1] = 9
            self.assertEqual(embed(), 0)
            np.testing.assert_array_equal(output[:, :4], table[ids, :4])

    def test_canonical_graph_and_capture_without_assets(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        self.assertEqual(OUTPUT.read_text(), json.dumps(build_circuit(), indent=2) + "\n")
        graph = build_circuit()
        self.assertEqual(len(weight_shapes(graph)), 59)
        ops = [op for block in graph["block_types"].values()
               for op in block.get("body", {}).get("ops", [])]
        select = next(op for op in ops if op["id"] == "predictor_feedback_argmax")
        lookup = next(op for op in ops if op["id"] == "predictor_feedback_embedding")
        self.assertEqual(select["graph_slots"]["inputs"]["values"],
                         "predictor_codebook0_logits")
        self.assertEqual(lookup["graph_slots"]["inputs"]["ids"],
                         "predictor_codebook0_selected_id")
        self.assertEqual(lookup["weight_refs"]["table"],
                         "talker.code_predictor.model.codec_embedding.0.weight")
        first_decode = graph["block_types"]["feedback_first_block"]["body"]["ops"]
        self.assertEqual(first_decode[0]["graph_slots"]["inputs"]["input"],
                         "feedback_embedding")
        self.assertEqual(FULL_OUTPUT.read_text(),
                         json.dumps(full_frame_circuit(), indent=2) + "\n")
        self.assertEqual(len(weight_shapes(full_frame_circuit())), 85)

    def test_asset_free_generated_feedback_routes_different_ids(self):
        graph = build_circuit()
        with tempfile.TemporaryDirectory() as temp:
            fn, acts, weights, layout, _, _ = compile_entry(
                Path(temp), graph, weight_shapes(graph))
            size = layout["memory"]["arena"]["total_size"]
            arena = _arena(size)
            for name, item in weights.items():
                if name.endswith("norm.weight"):
                    np.ndarray((1024,), dtype=np.float32, buffer=arena,
                               offset=item["abs_offset"])[:] = 1
            head0 = np.ndarray((2048, 1024), dtype=np.float32, buffer=arena,
                               offset=weights["talker.code_predictor.lm_head.0.weight"]["abs_offset"])
            head1 = np.ndarray((2048, 1024), dtype=np.float32, buffer=arena,
                               offset=weights["talker.code_predictor.lm_head.1.weight"]["abs_offset"])
            table = np.ndarray((2048, 1024), dtype=np.float32, buffer=arena,
                               offset=weights["talker.code_predictor.model.codec_embedding.0.weight"]["abs_offset"])
            head0[2, 0] = 2
            head1[1, 0], head1[2, 0] = 1, -1
            table[0, 0], table[2, 0] = -0.5, 0.5
            view(arena, acts, "feedback_step_index_input", np.int32, (1,))[0] = 0
            first = view(arena, acts, "block_input", np.float32, (2, 1024))
            selected = view(arena, acts, "predictor_codebook0_selected_id", np.int32, (1,))
            embedding = view(arena, acts, "feedback_embedding", np.float32, (1, 1024))
            logits = view(arena, acts, "feedback_predictor_logits", np.float32, (1, 2048))
            observed = []
            for value in (0.1, -0.1, 0.1):
                first[:] = 0
                first[1, 0] = value
                self.assertEqual(fn(arena, size), 0)
                observed.append((int(selected[0]), float(embedding[0, 0]),
                                 int(np.argmax(logits[-1]))))
            self.assertEqual(observed, [(2, 0.5, 1), (0, -0.5, 2),
                                        (2, 0.5, 1)])
            logits[:] = -719
            self.assertNotEqual(fn(arena, size - 1), 0)
            self.assertTrue(np.all(logits == -719))
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(np.argmax(logits[-1])), 1)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; connected feedback NOT_TESTED")
    def test_pinned_generated_first_feedback_edge(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        graph = build_circuit()
        with np.load(CAPTURE) as oracle, tempfile.TemporaryDirectory() as temp:
            fn, acts, weights, layout, calls, library = compile_entry(
                Path(temp), graph, weight_shapes(graph))
            self.assertEqual(len(calls["operations"]), 213)
            selected_functions = [op["function"] for op in calls["operations"]]
            self.assertEqual(selected_functions.count(
                "audio_argmax_last_row_checked_f32"), 1)
            self.assertEqual(selected_functions.count(
                "audio_text_embedding_token_major_f32"), 1)
            size = layout["memory"]["arena"]["total_size"]
            arena = _arena(size)
            with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source:
                for name, item in weights.items():
                    tensor = source.get_tensor(name).float().numpy()
                    arena[item["abs_offset"]:item["abs_offset"] + tensor.nbytes] = tensor.tobytes()
            view(arena, acts, "block_input", np.float32, (2, 1024))[:] = (
                oracle["call0_model_inputs_embeds"][0])
            step = view(arena, acts, "feedback_step_index_input", np.int32, (1,))
            step[0] = 0
            self.assertEqual(fn(arena, size), 0)
            code0 = view(arena, acts, "predictor_codebook0_selected_id", np.int32, (1,))
            embedding = view(arena, acts, "feedback_embedding", np.float32, (1, 1024))
            logits = view(arena, acts, "feedback_predictor_logits", np.float32, (1, 2048))
            self.assertEqual(int(code0[0]), int(oracle["call0_selected_id"][0]))
            np.testing.assert_array_equal(embedding, oracle["call1_model_inputs_embeds"][0])
            self.assertLessEqual(float(np.max(np.abs(logits - oracle["call1_logits"][0]))), 1e-4)
            self.assertEqual(int(np.argmax(logits[-1])), int(oracle["call1_selected_id"][0]))
            for layer in range(5):
                for kind in ("k", "v"):
                    cache = view(arena, acts, f"predictor_layer{layer}_cache_{kind}",
                                 np.float32, (8, 16, 128))
                    expected = oracle[f"call1_layer{layer}_{kind}_new"][0]
                    self.assertLessEqual(float(np.max(np.abs(cache[:, 2:3] - expected))), 5e-5)
            baseline = logits.copy()
            logits[:] = -719
            step[0] = 14
            self.assertNotEqual(fn(arena, size), 0)
            self.assertTrue(np.all(logits == -719))
            step[0] = 0
            self.assertEqual(fn(arena, size), 0)
            np.testing.assert_array_equal(logits, baseline)
            print("QWEN3_TTS_PREDICTOR_FIRST_FEEDBACK " + json.dumps({
                "selected_ids": [int(code0[0]), int(np.argmax(logits[-1]))],
                "embedding_max_abs": float(np.max(np.abs(
                    embedding - oracle["call1_model_inputs_embeds"][0]))),
                "logits_max_abs": float(np.max(np.abs(
                    logits - oracle["call1_logits"][0]))),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "complete_frame": "NOT_TESTED", "text_to_wav": "NOT_TESTED"},
                sort_keys=True))

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; full predictor frame NOT_TESTED")
    def test_pinned_generated_full_predictor_frame(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        graph = full_frame_circuit()
        with np.load(CAPTURE) as oracle, tempfile.TemporaryDirectory() as temp:
            fn, acts, weights, layout, calls, library = compile_entry(
                Path(temp), graph, weight_shapes(graph))
            self.assertEqual(len(calls["operations"]), 1568)
            selected_functions = [op["function"] for op in calls["operations"]]
            self.assertEqual(selected_functions.count(
                "audio_argmax_last_row_checked_f32"), 15)
            self.assertEqual(selected_functions.count(
                "audio_text_embedding_token_major_f32"), 14)
            size = layout["memory"]["arena"]["total_size"]
            arena = _arena(size)
            with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source:
                for name, item in weights.items():
                    tensor = source.get_tensor(name).float().numpy()
                    arena[item["abs_offset"]:item["abs_offset"] + tensor.nbytes] = tensor.tobytes()
            first = view(arena, acts, "block_input", np.float32, (2, 1024))
            first[:] = oracle["call0_model_inputs_embeds"][0]
            step = view(arena, acts, "feedback_step_index_input", np.int32, (1,))
            step[0] = 0
            valid = view(arena, acts, "predictor_valid_code_count", np.int32, (1,))
            valid[0] = 0
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 15)
            names = ["predictor_codebook0_selected_id"] + [
                f"step{i}_selected_id" for i in range(1, 15)]
            selected = [int(view(arena, acts, name, np.int32, (1,))[0])
                        for name in names]
            expected = [int(oracle[f"call{i}_selected_id"][0])
                        for i in range(15)]
            self.assertEqual(selected, expected)
            logits_max = 0.0
            for index in range(1, 15):
                name = ("feedback_predictor_logits" if index == 1 else
                        f"step{index}_predictor_logits")
                logits = view(arena, acts, name, np.float32, (1, 2048))
                logits_max = max(logits_max, float(np.max(np.abs(
                    logits - oracle[f"call{index}_logits"][0]))))
            self.assertLessEqual(logits_max, 1e-4)
            for layer in range(5):
                for kind in ("k", "v"):
                    cache = view(arena, acts, f"predictor_layer{layer}_cache_{kind}",
                                 np.float32, (8, 16, 128))
                    expected_new = oracle[f"call14_layer{layer}_{kind}_new"][0]
                    self.assertLessEqual(float(np.max(np.abs(
                        cache[:, 15:16] - expected_new))), 5e-5)
            baseline = selected.copy()
            # A different request must not inherit the previous frame's KV rows.
            first[:] = 0
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 15)
            first[:] = oracle["call0_model_inputs_embeds"][0]
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual([int(view(arena, acts, name, np.int32, (1,))[0])
                              for name in names], baseline)
            # Reject near the end of a frame. Earlier IDs may exist, but the
            # complete-frame validity scalar must remain unset.
            last_table = np.ndarray((2048, 1024), dtype=np.float32, buffer=arena,
                                    offset=weights[
                                        "talker.code_predictor.model.codec_embedding.13.weight"
                                    ]["abs_offset"])
            saved = float(last_table[expected[13], 0])
            last_table[expected[13], 0] = np.nan
            self.assertNotEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 0)
            last_table[expected[13], 0] = saved
            self.assertEqual(fn(arena, size), 0)
            self.assertEqual(int(valid[0]), 15)
            self.assertEqual([int(view(arena, acts, name, np.int32, (1,))[0])
                              for name in names], baseline)
            print("QWEN3_TTS_PREDICTOR_FULL_FRAME " + json.dumps({
                "selected_ids": baseline, "logits_max_abs": logits_max,
                "valid_codes": int(valid[0]), "library_sha256":
                    hashlib.sha256(library.read_bytes()).hexdigest(),
                "oracle_fed_initial_embeddings": True,
                "text_to_wav": "NOT_TESTED"}, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
