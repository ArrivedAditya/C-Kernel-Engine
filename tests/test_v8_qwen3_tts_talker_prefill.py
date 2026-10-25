"""Connected generated 28-layer Qwen3-TTS talker prefill against pinned PyTorch."""

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
from version.v8.tts.build_qwen3_tts_talker_prefill_circuit import (
    OUTPUT, build_circuit,
)
from version.v8.tts.build_qwen3_tts_prompt_first_block_circuit import prompt_plan


def weight_shapes(graph: dict) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for block in graph["sequence"]:
        for op in graph["block_types"][block]["body"]["ops"]:
            for port, name in op.get("weight_refs", {}).items():
                if port == "table":
                    shape = (op["params"]["V"], op["params"]["C"])
                elif port == "gamma":
                    shape = (op["params"]["C"],)
                elif port == "weight":
                    shape = (op["params"]["N"], op["params"]["K"])
                elif port == "bias":
                    shape = (op["params"]["N"],)
                else:
                    raise AssertionError((op["id"], port))
                shape = tuple(shape)
                if name in shapes and shapes[name] != shape:
                    raise AssertionError((name, shapes[name], shape))
                shapes[name] = shape
    return shapes


class Qwen3TTSTalkerPrefillTest(unittest.TestCase):
    # Pinned PyTorch 2.14.1 CPU capture. Layer 1 begins with activations above
    # 1,100, where one FP32 ULP is about 1.22e-4. Keep the observed peak and
    # RMSE beside these bounds rather than applying a global talker tolerance.
    LAYER_ABS_CEILINGS = (2e-5,) + (2.5e-4,) * 13 + (3e-4,) * 14
    LAYER_RMSE_CEILING = 1.2e-5
    FINAL_NORM_ABS_CEILING = 2e-4
    FINAL_NORM_RMSE_CEILING = 2e-5
    LOGITS_ABS_CEILING = 1.5e-4
    LOGITS_RMSE_CEILING = 1.2e-5

    def test_pinned_capture_integrity_without_model_assets(self):
        manifest = json.loads((FIXTURE / "qwen3_tts_talker_prefill.json").read_text())
        capture = FIXTURE / "qwen3_tts_talker_prefill.npz"
        self.assertEqual(hashlib.sha256(capture.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        self.assertEqual(manifest["model_revision"],
                         "85e237c12c027371202489a0ec509ded67b5e4b5")
        self.assertEqual(manifest["reference_revision"],
                         "022e286b98fbec7e1e916cb940cdf532cd9f488e")
        self.assertEqual(manifest["runtime"]["attention"], "eager")
        with np.load(capture) as oracle:
            self.assertEqual(set(oracle.files),
                             {"prompt_input", "final_norm", "logits"} |
                             {f"layer{i}_output" for i in range(28)})
            self.assertEqual(oracle["logits"].shape, (1, 18, 3072))

    def test_canonical_full_graph_and_weight_ownership(self):
        graph = build_circuit()
        self.assertEqual(OUTPUT.read_text(), json.dumps(graph, indent=2) + "\n")
        self.assertEqual(len(graph["sequence"]), 30)
        self.assertEqual(len(weight_shapes(graph)), 316)
        operations = [op for block in graph["sequence"]
                      for op in graph["block_types"][block]["body"]["ops"]]
        self.assertEqual(len(operations), 570)
        self.assertEqual(len({op["id"] for op in operations}), 570)
        for index in range(1, 28):
            first = graph["block_types"][f"talker_layer_{index}"]["body"]["ops"][0]
            self.assertEqual(first["graph_slots"]["inputs"]["input"],
                             "block_output" if index == 1 else
                             f"layer{index-1}_block_output")
            self.assertEqual(first["weight_refs"]["gamma"],
                             f"talker.model.layers.{index}.input_layernorm.weight")
        self.assertEqual(operations[-1]["id"], "talker_logits")
        for invalid in (0, 29, True, 1.5):
            with self.assertRaises(ValueError):
                build_circuit(layers=invalid)

    def test_two_layer_builder_preserves_distinct_edges(self):
        graph = build_circuit(layers=2)
        self.assertEqual(len(graph["sequence"]), 4)
        self.assertEqual(len(weight_shapes(graph)), 30)
        self.assertFalse(graph["contract"]["runtime_invariants"]["complete_talker_prefill"])

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned Qwen3-TTS weights unavailable; complete talker prefill NOT_TESTED")
    def test_pinned_generated_full_prefill(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        manifest = json.loads((FIXTURE / "qwen3_tts_talker_prefill.json").read_text())
        capture_path = FIXTURE / "qwen3_tts_talker_prefill.npz"
        self.assertEqual(hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        graph = build_circuit()
        shapes = weight_shapes(graph)
        with np.load(FIXTURE / "qwen3_tts_customvoice_reference.npz") as request, \
             np.load(capture_path) as oracle, \
             tempfile.TemporaryDirectory() as temp:
            fn, acts, weights, layout, calls, library = compile_entry(Path(temp), graph, shapes)
            self.assertEqual(len(calls["operations"]), 570)
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(raw, (-ctypes.addressof(raw)) & 63)
            with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as source:
                for name, item in weights.items():
                    tensor = source.get_tensor(name).float().numpy()
                    self.assertEqual(tensor.shape, shapes[name])
                    self.assertEqual(tensor.dtype, np.float32)
                    self.assertTrue(tensor.flags.c_contiguous)
                    arena[item["abs_offset"]:item["abs_offset"] + tensor.nbytes] = tensor.tobytes()
            text_ids, codec_ids = pinned_prompt_ids(model_dir, request)
            view(arena, acts, "text_ids", np.int32, text_ids.shape)[:] = text_ids
            view(arena, acts, "codec_ids", np.int32, codec_ids.shape)[:] = codec_ids
            for name, indices in prompt_plan(7).items():
                view(arena, acts, name, np.int32, (18,))[:] = indices
            self.assertEqual(fn(arena, len(arena)), 0)

            measurements = []
            prompt = view(arena, acts, "block_input", np.float32, (18, 1024))
            self.assertTrue(np.isfinite(prompt).all())
            self.assertLessEqual(float(np.max(np.abs(prompt - oracle["prompt_input"][0]))), 5e-7)
            for layer in range(28):
                name = "block_output" if layer == 0 else f"layer{layer}_block_output"
                actual = view(arena, acts, name, np.float32, (18, 1024))
                expected = oracle[f"layer{layer}_output"][0]
                self.assertTrue(np.isfinite(actual).all(), name)
                difference = actual.astype(np.float64) - expected.astype(np.float64)
                measurements.append({
                    "layer": layer,
                    "max_abs": float(np.max(np.abs(difference))),
                    "rmse": float(np.sqrt(np.mean(difference * difference))),
                    "reference_peak": float(np.max(np.abs(expected))),
                })
                self.assertLessEqual(measurements[-1]["max_abs"],
                                     self.LAYER_ABS_CEILINGS[layer], measurements[-1])
                self.assertLessEqual(measurements[-1]["rmse"],
                                     self.LAYER_RMSE_CEILING, measurements[-1])
            outputs = {}
            for name, fixture_name, width in (
                ("talker_final_norm", "final_norm", 1024),
                ("talker_logits", "logits", 3072),
            ):
                actual = view(arena, acts, name, np.float32, (18, width))
                expected = oracle[fixture_name][0]
                self.assertTrue(np.isfinite(actual).all(), name)
                delta = actual.astype(np.float64) - expected.astype(np.float64)
                outputs[name] = {
                    "max_abs": float(np.max(np.abs(delta))),
                    "rmse": float(np.sqrt(np.mean(delta * delta))),
                    "reference_peak": float(np.max(np.abs(expected))),
                }
            self.assertLessEqual(outputs["talker_final_norm"]["max_abs"],
                                 self.FINAL_NORM_ABS_CEILING)
            self.assertLessEqual(outputs["talker_final_norm"]["rmse"],
                                 self.FINAL_NORM_RMSE_CEILING)
            self.assertLessEqual(outputs["talker_logits"]["max_abs"],
                                 self.LOGITS_ABS_CEILING)
            self.assertLessEqual(outputs["talker_logits"]["rmse"],
                                 self.LOGITS_RMSE_CEILING)
            logits = view(arena, acts, "talker_logits", np.float32, (18, 3072))
            last_logits = oracle["logits"][0, -1]
            expected_id = int(np.argmax(last_logits))
            actual_id = int(np.argmax(logits[-1]))
            self.assertEqual(actual_id, expected_id)
            margin = float(last_logits[expected_id] - np.partition(last_logits, -2)[-2])
            self.assertGreater(margin, 2 * outputs["talker_logits"]["max_abs"])
            pristine = logits.copy()
            print("QWEN3_TTS_FULL_PREFILL " + json.dumps({
                "scope": "connected prompt through 28-layer prefill and logits; no cache or codes",
                "layers": measurements, "outputs": outputs,
                "last_token_top1": expected_id, "last_token_top1_margin": margin,
                "arena_bytes": size,
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "autonomous_speech": "NOT_TESTED"}, sort_keys=True))

            logits[:] = -777
            view(arena, acts, "codec_ids", np.int32, (7,))[0] = 3072
            self.assertNotEqual(fn(arena, len(arena)), 0)
            self.assertTrue(np.all(logits == -777))
            view(arena, acts, "codec_ids", np.int32, (7,))[0] = codec_ids[0]
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(logits, pristine)

            # Bounded A→B→A replay on one loaded generated entry. This is
            # prefill state isolation, not a continuation-cache reset claim.
            ids = view(arena, acts, "text_ids", np.int32, text_ids.shape)
            ids[-1] = int(text_ids[-1]) + 1
            self.assertEqual(fn(arena, len(arena)), 0)
            self.assertGreater(float(np.max(np.abs(logits - pristine))), 1e-6)
            ids[-1] = text_ids[-1]
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(logits, pristine)

            logits[:] = -777
            self.assertNotEqual(fn(arena, len(arena) - 1), 0)
            self.assertTrue(np.all(logits == -777))
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(logits, pristine)

            # A late-layer weight mutation must fail the matching checkpoint.
            late = next(item for item in weights.values()
                        if item["name"] == "talker.model.layers.27.mlp.down_proj.weight")
            weight = np.ndarray((1024, 3072), dtype=np.float32,
                                buffer=arena, offset=late["abs_offset"])
            original = float(weight[0, 0])
            weight[0, 0] = original + 10.0
            self.assertEqual(fn(arena, len(arena)), 0)
            late_output = view(arena, acts, "layer27_block_output", np.float32,
                               (18, 1024))
            self.assertGreater(float(np.max(np.abs(
                late_output - oracle["layer27_output"][0]))),
                self.LAYER_ABS_CEILINGS[27])
            weight[0, 0] = original
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(logits, pristine)


if __name__ == "__main__":
    unittest.main()
