"""Generated Qwen3-TTS teacher-forced continuation with checked KV state."""

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
from version.v8.tts.build_qwen3_tts_talker_prefill_cache_circuit import (
    OUTPUT as PREFILL_OUTPUT, build_circuit as prefill_circuit,
)
from version.v8.tts.build_qwen3_tts_talker_decode_circuit import (
    OUTPUT as DECODE_OUTPUT, build_circuit as decode_circuit,
)

CAPTURE = FIXTURE / "qwen3_tts_talker_continuation.npz"
MANIFEST = FIXTURE / "qwen3_tts_talker_continuation.json"


def _arena(size):
    raw = (ctypes.c_uint8 * (size + 63))()
    return (ctypes.c_uint8 * size).from_buffer(raw, (-ctypes.addressof(raw)) & 63)


class Qwen3TTSTalkerContinuationTest(unittest.TestCase):
    def test_capture_and_circuit_integrity_without_model_assets(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        self.assertEqual(PREFILL_OUTPUT.read_text(),
                         json.dumps(prefill_circuit(), indent=2) + "\n")
        self.assertEqual(DECODE_OUTPUT.read_text(),
                         json.dumps(decode_circuit(), indent=2) + "\n")
        self.assertEqual(manifest["model_revision"],
                         "85e237c12c027371202489a0ec509ded67b5e4b5")
        with np.load(CAPTURE) as oracle:
            for step in range(4):
                self.assertIn(f"step{step}_logits", oracle)
                for layer in range(28):
                    self.assertIn(f"step{step}_layer{layer}_output", oracle)
                    suffix = "cache" if step == 0 else "new"
                    self.assertIn(f"step{step}_layer{layer}_k_{suffix}", oracle)
                    self.assertIn(f"step{step}_layer{layer}_v_{suffix}", oracle)

    def test_generated_two_layer_rejection_and_recovery_without_assets(self):
        graph = decode_circuit(layers=2)
        with tempfile.TemporaryDirectory() as temp:
            fn, acts, weights, layout, calls, _ = compile_entry(
                Path(temp), graph, weight_shapes(graph))
            self.assertEqual(len(calls["operations"]), 44)
            size = layout["memory"]["arena"]["total_size"]
            arena = _arena(size)
            view(arena, acts, "step_index_input", np.int32, (1,))[0] = 0
            input_row = view(arena, acts, "block_input", np.float32, (1, 1024))
            input_row[:] = 0
            input_row[0, 0] = 0.1
            for layer in range(2):
                for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                    name = f"talker.model.layers.{layer}.{suffix}"
                    np.ndarray((1024,), dtype=np.float32, buffer=arena,
                               offset=weights[name]["abs_offset"])[:] = 1
            np.ndarray((1024,), dtype=np.float32, buffer=arena,
                       offset=weights["talker.model.norm.weight"]["abs_offset"])[:] = 1
            np.ndarray((3072, 1024), dtype=np.float32, buffer=arena,
                       offset=weights["talker.codec_head.weight"]["abs_offset"])[0, 0] = 1
            logits = view(arena, acts, "talker_logits", np.float32, (1, 3072))
            caches = [view(arena, acts, f"talker_layer{layer}_cache_{kind}",
                           np.float32, (8, 21, 128)) for layer in range(2)
                      for kind in ("k", "v")]
            logits[:] = -719
            self.assertEqual(fn(arena, size), 0)
            baseline = logits.copy()
            self.assertGreater(float(logits[0, 0]), 0)
            logits[:] = -719
            cached = [item.copy() for item in caches]
            view(arena, acts, "step_index_input", np.int32, (1,))[0] = 3
            self.assertNotEqual(fn(arena, size), 0)
            self.assertTrue(np.all(logits == -719))
            for actual, expected in zip(caches, cached):
                np.testing.assert_array_equal(actual, expected)
            view(arena, acts, "step_index_input", np.int32, (1,))[0] = 0
            self.assertEqual(fn(arena, size), 0)
            np.testing.assert_array_equal(logits, baseline)
            input_row[0, 0] = -0.2
            self.assertEqual(fn(arena, size), 0)
            self.assertLess(float(logits[0, 0]), 0)
            input_row[0, 0] = 0.1
            self.assertEqual(fn(arena, size), 0)
            np.testing.assert_array_equal(logits, baseline)
            # A later layer can fail after an earlier layer appended its row.
            # The request is invalid then: reset caller-owned state before reuse.
            q_weight = np.ndarray((2048, 1024), dtype=np.float32, buffer=arena,
                                  offset=weights["talker.model.layers.1.self_attn.q_proj.weight"]["abs_offset"])
            saved_cache = [item.copy() for item in caches]
            q_weight[0, 0] = np.nan
            logits[:] = -719
            self.assertNotEqual(fn(arena, size), 0)
            self.assertTrue(np.all(logits == -719))
            q_weight[0, 0] = 0
            for actual, saved in zip(caches, saved_cache):
                actual[:] = saved
            self.assertEqual(fn(arena, size), 0)
            np.testing.assert_array_equal(logits, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned Qwen3-TTS weights unavailable; continuation NOT_TESTED")
    def test_pinned_full_prefill_and_three_teacher_forced_steps(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        with np.load(CAPTURE) as oracle, \
             np.load(FIXTURE / "qwen3_tts_customvoice_reference.npz") as request, \
             tempfile.TemporaryDirectory() as temp:
            prefill = prefill_circuit()
            decode = decode_circuit()
            pf, pa, pw, pl, pc, _ = compile_entry(
                Path(temp) / "prefill", prefill, weight_shapes(prefill))
            df, da, dw, dl, dc, _ = compile_entry(
                Path(temp) / "decode", decode, weight_shapes(decode))
            self.assertEqual(len(pc["operations"]), 598)
            self.assertEqual(len(dc["operations"]), 564)
            psize = pl["memory"]["arena"]["total_size"]
            dsize = dl["memory"]["arena"]["total_size"]
            p_arena, d_arena = _arena(psize), _arena(dsize)
            with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source:
                for arena, entries, graph in ((p_arena, pw, prefill), (d_arena, dw, decode)):
                    shapes = weight_shapes(graph)
                    for name, item in entries.items():
                        tensor = source.get_tensor(name).float().numpy()
                        self.assertEqual(tensor.shape, shapes[name])
                        arena[item["abs_offset"]:item["abs_offset"] + tensor.nbytes] = tensor.tobytes()
            text_ids, codec_ids = pinned_prompt_ids(model_dir, request)
            view(p_arena, pa, "text_ids", np.int32, text_ids.shape)[:] = text_ids
            view(p_arena, pa, "codec_ids", np.int32, codec_ids.shape)[:] = codec_ids
            for name, values in prompt_plan(7).items():
                view(p_arena, pa, name, np.int32, (18,))[:] = values
            self.assertEqual(pf(p_arena, psize), 0)
            worst_prefill_cache = 0.0
            for layer in range(28):
                for kind in ("k", "v"):
                    name = f"talker_layer{layer}_cache_{kind}"
                    source = view(p_arena, pa, name, np.float32, (8, 21, 128))
                    target = view(d_arena, da, name, np.float32, (8, 21, 128))
                    expected = oracle[f"step0_layer{layer}_{kind}_cache"][0]
                    cache_error = float(np.max(np.abs(source[:, :18] - expected)))
                    worst_prefill_cache = max(worst_prefill_cache, cache_error)
                    target[:] = source
            # The cache-copy provider is bitwise; this bound covers upstream
            # prefill projection drift measured separately from the copy.
            self.assertLessEqual(worst_prefill_cache, 8e-5)
            worst_layer = 0.0
            worst_cache_row = 0.0
            worst_logits = 0.0
            for step in range(1, 4):
                view(d_arena, da, "step_index_input", np.int32, (1,))[0] = step - 1
                view(d_arena, da, "block_input", np.float32, (1, 1024))[:] = (
                    oracle[f"step{step}_input"][0])
                self.assertEqual(df(d_arena, dsize), 0, step)
                for layer in range(28):
                    output = "block_output" if layer == 0 else f"layer{layer}_block_output"
                    actual = view(d_arena, da, output, np.float32, (1, 1024))
                    expected = oracle[f"step{step}_layer{layer}_output"][0]
                    delta = actual.astype(np.float64) - expected.astype(np.float64)
                    worst_layer = max(worst_layer, float(np.max(np.abs(delta))))
                    self.assertLessEqual(float(np.max(np.abs(delta))), 8e-5,
                                         (step, layer, float(np.max(np.abs(delta)))))
                    for kind in ("k", "v"):
                        cache = view(d_arena, da, f"talker_layer{layer}_cache_{kind}",
                                     np.float32, (8, 21, 128))
                        expected_new = oracle[f"step{step}_layer{layer}_{kind}_new"][0]
                        worst_cache_row = max(worst_cache_row, float(np.max(
                            np.abs(cache[:, 17 + step:18 + step] - expected_new))))
                        np.testing.assert_allclose(cache[:, 17 + step:18 + step], expected_new,
                                                   atol=4e-5, rtol=4e-5)
                logits = view(d_arena, da, "talker_logits", np.float32, (1, 3072))
                expected = oracle[f"step{step}_logits"][0]
                delta = logits.astype(np.float64) - expected.astype(np.float64)
                worst_logits = max(worst_logits, float(np.max(np.abs(delta))))
                self.assertLessEqual(float(np.max(np.abs(delta))), 7e-5,
                                     (step, float(np.max(np.abs(delta)))))
                self.assertEqual(int(np.argmax(logits[-1])), int(np.argmax(expected[-1])))
            print("QWEN3_TTS_CONTINUATION " + json.dumps({
                "scope": "oracle-fed embeddings; generated 28-layer prefill and three decode steps",
                "prefill_cache_max_abs": worst_prefill_cache,
                "decode_layer_max_abs": worst_layer,
                "decode_new_cache_max_abs": worst_cache_row,
                "decode_logits_max_abs": worst_logits,
                "autonomous_codes": "NOT_TESTED", "waveform": "NOT_TESTED",
            }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
