"""Generated full Qwen3-TTS codec pre-transformer and per-layer oracle parity."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_codec_semantic import MODEL, generated_entry, view
from tests.test_v8_qwen3_tts_codec_preconv import fill_ids
from tests.test_v8_qwen3_tts_codec_qkv import PREFIX
from tests.test_v8_qwen3_tts_codec_attention import synthetic_weights, pinned_weights
from tests.test_v8_qwen3_tts_codec_first_block import add_synthetic_block_weights
from version.v8.tts.build_qwen3_tts_codec_pretransformer_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_pretransformer.npz")
NAMES = ("first_block_output", *(f"layer_{i}_block_output" for i in range(1, 8)),
         "decoder_final_norm", "decoder_output")


def add_synthetic_weights(arrays, rng, layers):
    add_synthetic_block_weights(arrays, rng)
    for layer in range(1, layers):
        for name, value in list(arrays.items()):
            if name.startswith(f"{PREFIX}.layers.0."):
                arrays[name.replace(".layers.0.", f".layers.{layer}.")] = (
                    rng.normal(size=value.shape).astype(np.float32) * .1)
    arrays[f"{PREFIX}.norm.weight"] = np.ones(4, dtype=np.float32)
    arrays[f"{PREFIX}.output_proj.weight"] = (
        rng.normal(size=(3, 4)).astype(np.float32) * .1)
    arrays[f"{PREFIX}.output_proj.bias"] = np.zeros(3, dtype=np.float32)


class Qwen3TTSCodecPretransformerTest(unittest.TestCase):
    def test_authored_circuit_and_fixture_integrity(self):
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_pretransformer_bounded.json"
        self.assertEqual(json.loads(path.read_text()), build_circuit())
        for layers in (0, 9, True):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                build_circuit(layers=layers)
        graph = build_circuit()
        ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
        self.assertEqual(len({op["id"] for op in ops}), len(ops))
        self.assertEqual(ops[-2]["id"], "decoder_final_norm")
        self.assertEqual(ops[-1]["id"], "decoder_output")
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        for source, field in (("preconv", "source_preconv_sha256"),
                              ("first_block", "source_first_block_sha256")):
            self.assertEqual(hashlib.sha256(MODEL.with_name(
                f"qwen3_tts_codec_{source}.npz").read_bytes()).hexdigest(),
                manifest[field])
        self.assertEqual((manifest["frames"], manifest["layers"],
                          manifest["hidden_width"], manifest["latent_width"]),
                         (46, 8, 512, 1024))
        with np.load(CHECKPOINT) as oracle:
            self.assertEqual(set(oracle.files), set(NAMES))
            for name in NAMES:
                width = 1024 if name == "decoder_output" else 512
                self.assertEqual(oracle[name].shape, (46, width))
                self.assertTrue(np.isfinite(oracle[name]).all())

    def test_generated_two_blocks_and_failure_recovery(self):
        rng = np.random.default_rng(20261024)
        arrays = synthetic_weights(rng, vocab=12)
        add_synthetic_weights(arrays, rng, 2)
        with tempfile.TemporaryDirectory() as temp:
            fn, arena, acts, calls, _ = generated_entry(
                Path(temp), arrays, 5, acoustic_codebooks=2,
                pre_transformer_complete=True, transformer_layers=2,
                qkv_heads=2, qkv_head_dim=2, sliding_window=3)
            codes = np.stack((np.arange(1, 6), np.arange(2, 7),
                              np.arange(3, 8)), axis=1).astype(np.int32)
            fill_ids(arena, acts, codes)
            self.assertEqual(fn(arena, len(arena)), 0)
            first = view(arena, acts, "first_block_output", np.float32, (5, 4)).copy()
            second = view(arena, acts, "layer_1_block_output", np.float32, (5, 4))
            final_norm = view(arena, acts, "decoder_final_norm", np.float32, (5, 4))
            output = view(arena, acts, "decoder_output", np.float32, (5, 3))
            baseline = output.copy()
            self.assertTrue(np.isfinite(baseline).all())
            expected_norm = second * np.reciprocal(np.sqrt(
                np.mean(second.astype(np.float64) ** 2, axis=1, keepdims=True) + 1e-5))
            np.testing.assert_allclose(final_norm, expected_norm, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(output, final_norm @ arrays[
                f"{PREFIX}.output_proj.weight"].T, atol=1e-6, rtol=1e-6)
            layer_weight = next(entry for entry in calls["memory"]["weights"]["entries"]
                                if entry["name"] ==
                                f"{PREFIX}.layers.1.mlp_layer_scale.scale")
            bound = np.ndarray((4,), dtype=np.float32, buffer=arena,
                               offset=layer_weight["abs_offset"])
            saved = bound.copy()
            bound[:] *= 2
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(view(
                arena, acts, "first_block_output", np.float32, (5, 4)), first)
            self.assertGreater(float(np.max(np.abs(output - baseline))), 1e-9)
            bound[:] = saved
            bound[0] = np.nan
            output[:] = -123
            self.assertNotEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, -123)
            bound[:] = saved
            ids = view(arena, acts, "acoustic_ids_1", np.int32, (5,))
            ids[0] = 12
            self.assertNotEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, -123)
            ids[0] = codes[0, 2]
            self.assertNotEqual(fn(arena, len(arena) - 1), 0)
            np.testing.assert_array_equal(output, -123)
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, baseline)

    def test_generated_all_eight_blocks_without_model_assets(self):
        rng = np.random.default_rng(20261025)
        arrays = synthetic_weights(rng, vocab=12)
        add_synthetic_weights(arrays, rng, 8)
        frames = 2
        with tempfile.TemporaryDirectory() as temp:
            fn, arena, acts, calls, _ = generated_entry(
                Path(temp), arrays, frames, acoustic_codebooks=2,
                pre_transformer_complete=True, qkv_heads=2,
                qkv_head_dim=2, sliding_window=3)
            codes = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
            fill_ids(arena, acts, codes)
            self.assertEqual(fn(arena, len(arena)), 0)
            output = view(arena, acts, "decoder_output", np.float32, (frames, 3))
            baseline = output.copy()
            self.assertTrue(np.isfinite(baseline).all())
            self.assertEqual(sum(op["function"] ==
                                 "attention_sliding_bounded_checked_f32"
                                 for op in calls["operations"]), 8)
            self.assertEqual(sum(op["function"] == "silu_mul_rows_checked_f32"
                                 for op in calls["operations"]), 8)
            final_weight = next(entry for entry in calls["memory"]["weights"]["entries"]
                                if entry["name"] ==
                                f"{PREFIX}.layers.7.mlp_layer_scale.scale")
            bound = np.ndarray((4,), dtype=np.float32, buffer=arena,
                               offset=final_weight["abs_offset"])
            saved = bound.copy()
            bound[:] += 1
            self.assertEqual(fn(arena, len(arena)), 0)
            self.assertGreater(float(np.max(np.abs(output - baseline))), 1e-9)
            bound[:] = saved
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned weights unavailable; complete codec transformer parity NOT_TESTED")
    def test_pinned_connected_eight_layers(self):
        from safetensors import safe_open
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        arrays = pinned_weights(model_dir)
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            for layer in range(8):
                for name in ("input_layernorm.weight", "self_attn.q_proj.weight",
                             "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                             "self_attn.o_proj.weight", "self_attn_layer_scale.scale",
                             "post_attention_layernorm.weight", "mlp.gate_proj.weight",
                             "mlp.up_proj.weight", "mlp.down_proj.weight",
                             "mlp_layer_scale.scale"):
                    key = f"{PREFIX}.layers.{layer}.{name}"
                    arrays[key] = np.ascontiguousarray(
                        source.get_tensor(key.removeprefix("codec.")).float().numpy())
            for name in ("norm.weight", "output_proj.weight", "output_proj.bias"):
                key = f"{PREFIX}.{name}"
                arrays[key] = np.ascontiguousarray(
                    source.get_tensor(key.removeprefix("codec.")).float().numpy())
        with np.load(MODEL) as fixture, np.load(CHECKPOINT) as oracle:
            codes = fixture["codec_tokens"].astype(np.int32)
            with tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, _, library = generated_entry(
                    Path(temp), arrays, len(codes), acoustic_codebooks=15,
                    pre_transformer_complete=True)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                errors, rmses, worst = {}, {}, {}
                for name in NAMES:
                    expected = oracle[name]
                    actual = view(arena, acts, name, np.float32, expected.shape)
                    diff = np.abs(actual - expected)
                    errors[name] = float(np.max(diff))
                    rmses[name] = float(np.sqrt(np.mean(np.square(diff))))
                    index = np.unravel_index(int(np.argmax(diff)), diff.shape)
                    worst[name] = {"index": [int(i) for i in index],
                                   "actual": float(actual[index]),
                                   "reference": float(expected[index])}
                    self.assertTrue(np.isfinite(actual).all(), name)
                print("QWEN3_TTS_CODEC_PRETRANSFORMER " + json.dumps({
                    "max_abs_errors": errors, "rmse": rmses, "worst": worst,
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "scope": "reference codec codes through all eight decoder transformer blocks and output projection; waveform NOT_TESTED",
                }, sort_keys=True))
                for name, limit in LIMITS.items():
                    self.assertLessEqual(errors[name], limit, name)


# Per-checkpoint ceilings include modest cross-host rounding headroom over the
# direct pinned capture. They are not a single whole-model tolerance.
LIMITS = {
    "first_block_output": 1e-6,
    "layer_1_block_output": 1e-6,
    "layer_2_block_output": 6e-6,
    "layer_3_block_output": 2e-6,
    "layer_4_block_output": 1e-6,
    "layer_5_block_output": 1.5e-6,
    "layer_6_block_output": 1.5e-6,
    "layer_7_block_output": 2e-6,
    "decoder_final_norm": 1e-6,
    "decoder_output": 1e-6,
}


if __name__ == "__main__":
    unittest.main()
