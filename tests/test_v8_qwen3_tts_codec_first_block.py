"""Generated Qwen3-TTS codec decoder through its first complete transformer block."""

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
from tests.test_v8_attention_sliding_bounded_oracle import reference as sliding_reference
from version.v8.tts.build_qwen3_tts_codec_first_block_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_first_block.npz")
NAMES = (
    "first_attention_token", "first_attention_proj", "first_attention_scaled",
    "first_post_attention", "first_post_attention_norm", "first_gate",
    "first_up", "first_gated_mlp", "first_down", "first_mlp_scaled",
    "first_block_output",
)
LIMITS = {
    "first_attention_token": 2e-6,
    "first_attention_proj": 3e-6,
    "first_attention_scaled": 1e-6,
    "first_post_attention": 2e-6,
    "first_post_attention_norm": 4e-6,
    "first_gate": 4e-6,
    "first_up": 4e-6,
    "first_gated_mlp": 1e-5,
    "first_down": 5e-6,
    "first_mlp_scaled": 2e-6,
    "first_block_output": 3e-6,
}


def add_synthetic_block_weights(arrays, rng, *, hidden=4, heads=2, dim=2,
                                intermediate=6):
    for name, shape in (
        ("self_attn.o_proj.weight", (hidden, heads * dim)),
        ("mlp.gate_proj.weight", (intermediate, hidden)),
        ("mlp.up_proj.weight", (intermediate, hidden)),
        ("mlp.down_proj.weight", (hidden, intermediate)),
    ):
        arrays[f"{PREFIX}.layers.0.{name}"] = rng.normal(
            size=shape).astype(np.float32) * .1
    for name in ("self_attn_layer_scale.scale", "mlp_layer_scale.scale",
                 "post_attention_layernorm.weight"):
        arrays[f"{PREFIX}.layers.0.{name}"] = rng.normal(
            size=hidden).astype(np.float32)


class Qwen3TTSCodecFirstBlockTest(unittest.TestCase):
    def test_authored_circuit_and_fixture_integrity(self):
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_first_block_bounded.json"
        self.assertEqual(json.loads(path.read_text()), build_circuit())
        with self.assertRaises(ValueError):
            build_circuit(intermediate_width=0)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        for name, field in (("preconv", "source_preconv_sha256"),
                            ("attention", "source_attention_sha256")):
            path = MODEL.with_name(f"qwen3_tts_codec_{name}.npz")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             manifest[field])
        self.assertEqual(manifest["model_revision"],
                         "85e237c12c027371202489a0ec509ded67b5e4b5")
        self.assertEqual(manifest["reference_revision"],
                         "022e286b98fbec7e1e916cb940cdf532cd9f488e")
        self.assertEqual((manifest["frames"], manifest["hidden_width"],
                          manifest["intermediate_width"],
                          manifest["sliding_window"]), (46, 512, 1024, 72))
        with np.load(CHECKPOINT) as values:
            self.assertEqual(set(values.files), set(NAMES))
            for name in NAMES:
                width = 1024 if name in ("first_attention_token", "first_gate",
                                          "first_up", "first_gated_mlp") else 512
                self.assertEqual(values[name].shape, (46, width))
                self.assertTrue(np.isfinite(values[name]).all())

    def test_generated_nonzero_block_and_recovery(self):
        rng = np.random.default_rng(20261024)
        arrays = synthetic_weights(rng, vocab=12)
        add_synthetic_block_weights(arrays, rng)
        frames, hidden, intermediate = 5, 4, 6
        with tempfile.TemporaryDirectory() as temp:
            fn, arena, acts, calls, _ = generated_entry(
                Path(temp), arrays, frames, acoustic_codebooks=2,
                pre_transformer_first_block=True,
                qkv_heads=2, qkv_head_dim=2, sliding_window=3)
            codes = np.stack((np.arange(1, frames + 1),
                              np.arange(2, frames + 2),
                              np.arange(3, frames + 3)), axis=1).astype(np.int32)
            fill_ids(arena, acts, codes)
            self.assertEqual(fn(arena, len(arena)), 0)
            results = {
                name: view(arena, acts, name, np.float32,
                           (frames, intermediate if name in (
                               "first_gate", "first_up", "first_gated_mlp") else hidden))
                for name in NAMES if name != "first_attention_token"
            }
            results["first_attention_token"] = view(
                arena, acts, "first_attention_token", np.float32, (frames, hidden))
            head = view(arena, acts, "first_attention_head", np.float32,
                        (2, frames, 2))
            expected = head.transpose(1, 0, 2).reshape(frames, hidden)
            np.testing.assert_array_equal(results["first_attention_token"], expected)
            def weight(name):
                return arrays[f"{PREFIX}.layers.0.{name}"]
            projected = expected @ weight("self_attn.o_proj.weight").T
            post = view(arena, acts, "transformer_input", np.float32,
                        (frames, hidden)) + projected * weight(
                            "self_attn_layer_scale.scale")
            norm = post * np.reciprocal(np.sqrt(
                np.mean(np.square(post), axis=1, keepdims=True) + 1e-5))
            norm *= weight("post_attention_layernorm.weight")
            gate = norm @ weight("mlp.gate_proj.weight").T
            up = norm @ weight("mlp.up_proj.weight").T
            gated = (gate / (1 + np.exp(-gate))) * up
            down = gated @ weight("mlp.down_proj.weight").T
            expected_by_name = {
                "first_attention_proj": projected,
                "first_attention_scaled": projected * weight("self_attn_layer_scale.scale"),
                "first_post_attention": post,
                "first_post_attention_norm": norm,
                "first_gate": gate, "first_up": up,
                "first_gated_mlp": gated, "first_down": down,
                "first_mlp_scaled": down * weight("mlp_layer_scale.scale"),
                "first_block_output": post + down * weight("mlp_layer_scale.scale"),
            }
            for name, wanted in expected_by_name.items():
                np.testing.assert_allclose(results[name], wanted,
                                           atol=1e-6, rtol=1e-6, err_msg=name)
            baseline = results["first_block_output"].copy()
            scales = next(entry for entry in calls["memory"]["weights"]["entries"]
                          if entry["name"] ==
                          f"{PREFIX}.layers.0.mlp_layer_scale.scale")
            bound = np.ndarray((hidden,), dtype=np.float32, buffer=arena,
                               offset=scales["abs_offset"])
            original = bound.copy()
            bound[:] *= 2
            self.assertEqual(fn(arena, len(arena)), 0)
            self.assertGreater(float(np.max(np.abs(
                results["first_block_output"] - baseline))), 1e-8)
            bound[:] = original
            ids = view(arena, acts, "acoustic_ids_1", np.int32, (frames,))
            ids[-1] = (int(ids[-1]) + 1) % 12
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(results["first_block_output"][:-1],
                                          baseline[:-1])
            self.assertGreater(float(np.max(np.abs(
                results["first_block_output"][-1] - baseline[-1]))), 1e-9)
            ids[-1] = codes[-1, 2]
            ids[0] = 12
            results["first_block_output"][:] = -123
            self.assertNotEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(results["first_block_output"], -123)
            ids[0] = codes[0, 2]
            self.assertNotEqual(fn(arena, len(arena) - 1), 0)
            np.testing.assert_array_equal(results["first_block_output"], -123)
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(results["first_block_output"], baseline)

    def test_generated_block_crosses_sliding_window(self):
        rng = np.random.default_rng(20261025)
        arrays = synthetic_weights(rng, vocab=83)
        add_synthetic_block_weights(arrays, rng)
        frames = 73
        with tempfile.TemporaryDirectory() as temp:
            fn, arena, acts, _, _ = generated_entry(
                Path(temp), arrays, frames, acoustic_codebooks=2,
                pre_transformer_first_block=True,
                qkv_heads=2, qkv_head_dim=2, sliding_window=72)
            codes = np.stack((np.arange(frames), np.arange(1, frames + 1),
                              np.arange(2, frames + 2)), axis=1).astype(np.int32)
            fill_ids(arena, acts, codes)
            self.assertEqual(fn(arena, len(arena)), 0)
            shape = (2, frames, 2)
            actual = view(arena, acts, "first_attention_head", np.float32, shape)
            q = view(arena, acts, "first_q_head", np.float32, shape)
            k = view(arena, acts, "first_k_head", np.float32, shape)
            v = view(arena, acts, "first_v_head", np.float32, shape)
            np.testing.assert_allclose(actual, sliding_reference(q, k, v, 72),
                                       atol=1e-6, rtol=1e-6)
            self.assertTrue(np.isfinite(view(
                arena, acts, "first_block_output", np.float32,
                (frames, 4))).all())

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned weights unavailable; complete codec block parity NOT_TESTED")
    def test_pinned_connected_first_block(self):
        from safetensors import safe_open
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        arrays = pinned_weights(model_dir)
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            for name in (
                "self_attn.o_proj.weight",
                "self_attn_layer_scale.scale",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight",
                "mlp.down_proj.weight", "mlp_layer_scale.scale",
            ):
                key = f"{PREFIX}.layers.0.{name}"
                arrays[key] = np.ascontiguousarray(
                    source.get_tensor(key.removeprefix("codec.")).float().numpy())
        with np.load(MODEL) as fixture, np.load(CHECKPOINT) as oracle:
            codes = fixture["codec_tokens"].astype(np.int32)
            frames = len(codes)
            with tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, library = generated_entry(
                    Path(temp), arrays, frames, acoustic_codebooks=15,
                    pre_transformer_first_block=True)
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
                    self.assertLessEqual(errors[name], LIMITS[name], name)
                print("QWEN3_TTS_CODEC_FIRST_BLOCK " + json.dumps({
                    "max_abs_errors": errors, "rmse": rmses,
                    "worst": worst, "limits": LIMITS,
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "scope": "reference codec codes through first complete decoder transformer block; waveform NOT_TESTED",
                }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
