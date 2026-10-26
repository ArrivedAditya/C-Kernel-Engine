"""Connected generated five-layer Qwen3-TTS code-predictor prefill."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_prompt_first_block import FIXTURE, compile_entry, view
from tests.test_v8_qwen3_tts_talker_prefill import weight_shapes
from version.v8.tts.build_qwen3_tts_code_predictor_prefill_circuit import (
    OUTPUT, build_circuit,
)
from version.v8.tts.build_qwen3_tts_code_predictor_decode_circuit import (
    OUTPUT as DECODE_OUTPUT, build_circuit as decode_circuit,
)

CAPTURE = FIXTURE / "qwen3_tts_code_predictor.npz"
MANIFEST = FIXTURE / "qwen3_tts_code_predictor.json"


def _arena(size):
    raw = (ctypes.c_uint8 * (size + 63))()
    return (ctypes.c_uint8 * size).from_buffer(raw, (-ctypes.addressof(raw)) & 63)


class Qwen3TTSCodePredictorPrefillTest(unittest.TestCase):
    def test_capture_integrity_and_canonical_graph(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        self.assertEqual(manifest["selected_ids"],
                         [117, 1114, 1723, 1443, 1543, 1751, 240, 1242,
                          329, 293, 1347, 747, 2027, 180, 1187])
        self.assertEqual(OUTPUT.read_text(), json.dumps(build_circuit(), indent=2) + "\n")
        self.assertEqual(DECODE_OUTPUT.read_text(),
                         json.dumps(decode_circuit(), indent=2) + "\n")
        graph = build_circuit()
        self.assertEqual(len(graph["sequence"]), 7)
        self.assertEqual(len(weight_shapes(graph)), 57)
        for layer in range(5):
            ops = graph["block_types"]["first_block" if layer == 0 else
                                       f"predictor_layer_{layer}"]["body"]["ops"]
            self.assertEqual(ops[0]["weight_refs"]["gamma"],
                             f"talker.code_predictor.model.layers.{layer}.input_layernorm.weight")
        with np.load(CAPTURE) as oracle:
            self.assertEqual(oracle["talker_first_code_id"].tolist(), [[1995]])
            self.assertEqual(oracle["call0_model_inputs_embeds"].shape, (1, 2, 1024))
            self.assertEqual(oracle["call0_logits"].shape, (1, 2, 2048))

    def test_decode_position_rejection_preserves_logits_and_cache(self):
        for bad in (-1, 14, 2**31 - 1):
            with self.subTest(step=bad), tempfile.TemporaryDirectory() as temp:
                graph = decode_circuit(layers=2)
                fn, acts, _, layout, _, _ = compile_entry(
                    Path(temp), graph, weight_shapes(graph))
                size = layout["memory"]["arena"]["total_size"]
                arena = _arena(size)
                logits = view(arena, acts, "predictor_logits", np.float32, (1, 2048))
                logits[:] = -731
                caches = [view(arena, acts, f"predictor_layer{layer}_cache_{kind}",
                               np.float32, (8, 16, 128)) for layer in range(2)
                          for kind in ("k", "v")]
                for cache in caches:
                    cache[:] = 0.25
                view(arena, acts, "step_index_input", np.int32, (1,))[0] = bad
                self.assertNotEqual(fn(arena, size), 0)
                self.assertTrue(np.all(logits == -731))
                for cache in caches:
                    self.assertTrue(np.all(cache == 0.25))
                view(arena, acts, "step_index_input", np.int32, (1,))[0] = 0
                self.assertEqual(fn(arena, size), 0)
                self.assertTrue(np.isfinite(logits).all())
                baseline = logits.copy()
                saved_caches = [cache.copy() for cache in caches]
                logits[:] = -731
                self.assertNotEqual(fn(arena, size - 1), 0)
                self.assertTrue(np.all(logits == -731))
                for cache, saved in zip(caches, saved_caches):
                    np.testing.assert_array_equal(cache, saved)
                self.assertEqual(fn(arena, size), 0)
                np.testing.assert_array_equal(logits, baseline)
        for bad in (0, 15, True, 1.0):
            with self.subTest(codebook=bad):
                with self.assertRaises(ValueError):
                    decode_circuit(codebook_index=bad)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; predictor prefill NOT_TESTED")
    def test_pinned_generated_predictor_prefill(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        graph = build_circuit()
        shapes = weight_shapes(graph)
        with np.load(CAPTURE) as oracle, tempfile.TemporaryDirectory() as temp:
            fn, acts, weights, layout, calls, library = compile_entry(
                Path(temp), graph, shapes)
            self.assertEqual(len(calls["operations"]), 107)
            size = layout["memory"]["arena"]["total_size"]
            arena = _arena(size)
            with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source:
                for name, item in weights.items():
                    tensor = source.get_tensor(name).float().numpy()
                    self.assertEqual(tensor.shape, shapes[name])
                    arena[item["abs_offset"]:item["abs_offset"] + tensor.nbytes] = tensor.tobytes()
            prompt = view(arena, acts, "block_input", np.float32, (2, 1024))
            prompt[:] = oracle["call0_model_inputs_embeds"][0]
            self.assertEqual(fn(arena, size), 0)
            measurements = []
            for layer in range(5):
                output = "block_output" if layer == 0 else f"layer{layer}_block_output"
                actual = view(arena, acts, output, np.float32, (2, 1024))
                expected = oracle[f"call0_layer{layer}_output"][0]
                delta = actual.astype(np.float64) - expected.astype(np.float64)
                # The pinned predictor has isolated activations near 50,000;
                # retain an elementwise scale-aware bound instead of treating
                # their FP32 ULPs as a global absolute tolerance.
                self.assertTrue(np.all(np.abs(delta) <=
                                       1e-4 + 1e-6 * np.abs(expected)), layer)
                measurements.append({"layer": layer,
                                     "max_abs": float(np.max(np.abs(delta))),
                                     "rmse": float(np.sqrt(np.mean(delta * delta)))})
                for kind in ("k", "v"):
                    cache = view(arena, acts, f"predictor_layer{layer}_cache_{kind}",
                                 np.float32, (8, 16, 128))
                    expected_cache = oracle[f"call0_layer{layer}_{kind}_cache"][0]
                    self.assertTrue(np.isfinite(cache[:, :2]).all())
                    measurements[-1][f"{kind}_cache_max_abs"] = float(np.max(
                        np.abs(cache[:, :2] - expected_cache)))
                    self.assertLessEqual(measurements[-1][f"{kind}_cache_max_abs"], 5e-5)
            norm = view(arena, acts, "predictor_final_norm", np.float32, (2, 1024))
            logits = view(arena, acts, "predictor_codebook0_logits", np.float32, (2, 2048))
            norm_error = float(np.max(np.abs(norm - oracle["call0_final_norm"][0])))
            logit_error = float(np.max(np.abs(logits - oracle["call0_logits"][0])))
            self.assertLessEqual(norm_error, 5e-5)
            self.assertLessEqual(logit_error, 5e-5)
            diagnostics = []
            for name, cke_name, width in (
                ("input_layernorm", "input_norm", 1024),
                ("self_attn_q_proj", "q_proj", 2048),
                ("self_attn_k_proj", "k_proj", 1024),
                ("self_attn_v_proj", "v_proj", 1024),
                ("self_attn", "attention_proj", 1024),
                ("post_attention_layernorm", "post_attention_norm", 1024),
                ("mlp_gate_proj", "gate_proj", 3072),
                ("mlp_up_proj", "up_proj", 3072),
                ("mlp", "down_proj", 1024),
            ):
                actual = view(arena, acts, "layer2_" + cke_name,
                              np.float32, (2, width))
                expected = oracle["call0_layer2_" + name][0].reshape(2, width)
                delta = actual.astype(np.float64) - expected.astype(np.float64)
                index = np.unravel_index(int(np.argmax(np.abs(delta))), delta.shape)
                diagnostics.append({"stage": name, "max_abs": float(np.max(np.abs(delta))),
                                    "rmse": float(np.sqrt(np.mean(delta * delta))),
                                    "worst_index": [int(i) for i in index],
                                    "actual": float(actual[index]),
                                    "reference": float(expected[index])})
            print("QWEN3_TTS_PREDICTOR_PREFILL " + json.dumps({
                "layers": measurements, "norm_max_abs": norm_error,
                "logits_max_abs": logit_error,
                "layer2_diagnostics": diagnostics,
                "selected_id": int(np.argmax(logits[-1])),
                "reference_selected_id": int(oracle["call0_selected_id"][0]),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "autonomous_codes": "NOT_TESTED"}, sort_keys=True))

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned CustomVoice weights unavailable; predictor decode NOT_TESTED")
    def test_pinned_generated_all_teacher_forced_codebooks(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        manifest = json.loads(MANIFEST.read_text())
        with np.load(CAPTURE) as oracle, tempfile.TemporaryDirectory() as temp:
            prefill = build_circuit()
            pf, pa, pw, pl, _, _ = compile_entry(
                Path(temp) / "prefill", prefill, weight_shapes(prefill))
            psize = pl["memory"]["arena"]["total_size"]
            p_arena = _arena(psize)
            with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source:
                for name, item in pw.items():
                    tensor = source.get_tensor(name).float().numpy()
                    p_arena[item["abs_offset"]:item["abs_offset"] + tensor.nbytes] = tensor.tobytes()
                view(p_arena, pa, "block_input", np.float32, (2, 1024))[:] = (
                    oracle["call0_model_inputs_embeds"][0])
                self.assertEqual(pf(p_arena, psize), 0)
                caches = {}
                for layer in range(5):
                    for kind in ("k", "v"):
                        name = f"predictor_layer{layer}_cache_{kind}"
                        caches[name] = view(p_arena, pa, name, np.float32,
                                            (8, 16, 128)).copy()
                selected = [int(np.argmax(view(p_arena, pa,
                            "predictor_codebook0_logits", np.float32, (2, 2048))[-1]))]
                layer_max, logits_max, cache_max = 0.0, 0.0, 0.0
                for step in range(1, 15):
                    decode = decode_circuit(codebook_index=step)
                    df, da, dw, dl, _, _ = compile_entry(
                        Path(temp) / f"decode{step}", decode, weight_shapes(decode))
                    dsize = dl["memory"]["arena"]["total_size"]
                    d_arena = _arena(dsize)
                    for name, item in dw.items():
                        tensor = source.get_tensor(name).float().numpy()
                        d_arena[item["abs_offset"]:item["abs_offset"] + tensor.nbytes] = tensor.tobytes()
                    for name, previous in caches.items():
                        view(d_arena, da, name, np.float32, (8, 16, 128))[:] = previous
                    view(d_arena, da, "step_index_input", np.int32, (1,))[0] = step - 1
                    view(d_arena, da, "block_input", np.float32, (1, 1024))[:] = (
                        oracle[f"call{step}_model_inputs_embeds"][0])
                    self.assertEqual(df(d_arena, dsize), 0, step)
                    for layer in range(5):
                        name = "block_output" if layer == 0 else f"layer{layer}_block_output"
                        actual = view(d_arena, da, name, np.float32, (1, 1024))
                        expected = oracle[f"call{step}_layer{layer}_output"][0]
                        layer_max = max(layer_max, float(np.max(np.abs(actual - expected))))
                        for kind in ("k", "v"):
                            name = f"predictor_layer{layer}_cache_{kind}"
                            cache = view(d_arena, da, name, np.float32, (8, 16, 128))
                            expected_new = oracle[f"call{step}_layer{layer}_{kind}_new"][0]
                            cache_max = max(cache_max, float(np.max(np.abs(
                                cache[:, step + 1:step + 2] - expected_new))))
                            caches[name] = cache.copy()
                    logits = view(d_arena, da, "predictor_logits", np.float32,
                                  (1, 2048))
                    logits_max = max(logits_max, float(np.max(np.abs(
                        logits - oracle[f"call{step}_logits"][0]))))
                    selected.append(int(np.argmax(logits[-1])))
                self.assertEqual(selected, manifest["selected_ids"])
                self.assertLessEqual(layer_max, 2e-4)
                self.assertLessEqual(logits_max, 0.001)
                self.assertLessEqual(cache_max, 0.001)
                print("QWEN3_TTS_PREDICTOR_TEACHER_FORCED " + json.dumps({
                    "layer_max_abs": layer_max, "logits_max_abs": logits_max,
                    "cache_new_max_abs": cache_max, "selected_ids": selected,
                    "scope": "oracle-fed embeddings; CKE five-layer prefill and fourteen decode calls",
                    "autonomous_codes": "NOT_TESTED"}, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
