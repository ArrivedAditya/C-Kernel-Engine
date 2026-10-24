"""Connected Qwen3-TTS reference codes through first sliding attention."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_attention_sliding_bounded_oracle import reference
from tests.test_v8_qwen3_tts_codec_semantic import (
    TABLE, PROJECTION, ACOUSTIC_PROJECTION, MODEL, generated_entry, view,
)
from tests.test_v8_qwen3_tts_codec_preconv import CONV_WEIGHT, CONV_BIAS, fill_ids
from tests.test_v8_qwen3_tts_codec_qkv import PREFIX
from version.v8.tts.build_qwen3_tts_codec_attention_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_attention.npz")


def synthetic_weights(rng, *, vocab=11, width=4, output_width=7,
                      latent=3, hidden=4):
    arrays = {
        TABLE: rng.normal(size=(vocab, width)).astype(np.float32),
        PROJECTION: rng.normal(size=(output_width, width)).astype(np.float32) * .1,
        ACOUSTIC_PROJECTION: rng.normal(size=(output_width, width)).astype(np.float32) * .1,
        CONV_WEIGHT: rng.normal(size=(latent, output_width, 3)).astype(np.float32) * .1,
        CONV_BIAS: rng.normal(size=latent).astype(np.float32) * .1,
        f"{PREFIX}.input_proj.weight": rng.normal(size=(hidden, latent)).astype(np.float32) * .1,
        f"{PREFIX}.input_proj.bias": rng.normal(size=hidden).astype(np.float32) * .1,
        f"{PREFIX}.layers.0.input_layernorm.weight": rng.normal(size=hidden).astype(np.float32),
    }
    for index in range(2):
        arrays[f"codec.acoustic.{index}.effective_embedding"] = rng.normal(
            size=(vocab, width)).astype(np.float32)
    for name in ("q", "k", "v"):
        arrays[f"{PREFIX}.layers.0.self_attn.{name}_proj.weight"] = rng.normal(
            size=(hidden, hidden)).astype(np.float32) * .1
    return arrays


def pinned_weights(model_dir):
    from safetensors import safe_open
    from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir
    verify_model_dir(model_dir)
    arrays = {}
    with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                   framework="pt", device="cpu") as source:
        for group, count in (("rvq_first", 1), ("rvq_rest", 15)):
            prefix = f"decoder.quantizer.{group}."
            for index in range(count):
                layer = f"{prefix}vq.layers.{index}._codebook."
                usage = source.get_tensor(layer + "cluster_usage").float().numpy()
                summed = source.get_tensor(layer + "embedding_sum").float().numpy()
                name = (TABLE if group == "rvq_first" else
                        f"codec.acoustic.{index}.effective_embedding")
                arrays[name] = np.ascontiguousarray(
                    (summed / np.maximum(usage, 1e-5)[:, None]).astype(np.float32))
            weight = source.get_tensor(prefix + "output_proj.weight")
            arrays[PROJECTION if group == "rvq_first" else
                   ACOUSTIC_PROJECTION] = np.ascontiguousarray(
                weight.float().numpy().reshape(512, 256))
        for name in (CONV_WEIGHT, CONV_BIAS):
            arrays[name] = np.ascontiguousarray(
                source.get_tensor(name.removeprefix("codec.")).float().numpy())
        for name in ("input_proj.weight", "input_proj.bias",
                     "layers.0.input_layernorm.weight",
                     "layers.0.self_attn.q_proj.weight",
                     "layers.0.self_attn.k_proj.weight",
                     "layers.0.self_attn.v_proj.weight"):
            arrays[f"{PREFIX}.{name}"] = np.ascontiguousarray(
                source.get_tensor(f"decoder.pre_transformer.{name}").float().numpy())
    return arrays


class Qwen3TTSCodecAttentionTest(unittest.TestCase):
    def test_authored_circuit_and_fixture_integrity(self):
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_attention_bounded.json"
        self.assertEqual(json.loads(path.read_text()), build_circuit())
        for bad in (0, True, 8001):
            with self.subTest(window=bad), self.assertRaises(ValueError):
                build_circuit(sliding_window=bad)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        self.assertEqual(hashlib.sha256(MODEL.with_name(
            "qwen3_tts_codec_preconv.npz").read_bytes()).hexdigest(),
                         manifest["source_preconv_sha256"])
        self.assertEqual(hashlib.sha256(MODEL.with_name(
            "qwen3_tts_codec_qkv.npz").read_bytes()).hexdigest(),
                         manifest["source_qkv_sha256"])
        self.assertEqual(manifest["model_revision"],
                         "85e237c12c027371202489a0ec509ded67b5e4b5")
        self.assertEqual(manifest["reference_revision"],
                         "022e286b98fbec7e1e916cb940cdf532cd9f488e")
        self.assertEqual((manifest["frames"], manifest["sliding_window"]), (46, 72))
        with np.load(CHECKPOINT) as values:
            self.assertEqual(values["first_attention_head"].shape, (16, 46, 64))
            self.assertTrue(np.isfinite(values["first_attention_head"]).all())

    def test_generated_nonzero_sliding_and_failure_recovery(self):
        rng = np.random.default_rng(20261025)
        frames, heads, dim, vocab = 73, 2, 2, 83
        arrays = synthetic_weights(rng, vocab=vocab)
        with tempfile.TemporaryDirectory() as temp:
            fn, arena, acts, calls, _ = generated_entry(
                Path(temp), arrays, frames, acoustic_codebooks=2,
                pre_transformer_attention=True, qkv_heads=heads,
                qkv_head_dim=dim, sliding_window=72)
            codes = np.stack((np.arange(frames), np.arange(1, frames + 1),
                              np.arange(2, frames + 2)), axis=1).astype(np.int32)
            fill_ids(arena, acts, codes)
            self.assertEqual(fn(arena, len(arena)), 0)
            actual = view(arena, acts, "first_attention_head", np.float32,
                          (heads, frames, dim))
            q = view(arena, acts, "first_q_head", np.float32, actual.shape)
            k = view(arena, acts, "first_k_head", np.float32, actual.shape)
            v = view(arena, acts, "first_v_head", np.float32, actual.shape)
            np.testing.assert_allclose(actual, reference(q, k, v, 72),
                                       atol=1e-6, rtol=1e-6)
            baseline = actual.copy()
            late = view(arena, acts, "acoustic_ids_1", np.int32, (frames,))
            late[-1] = (int(late[-1]) + 1) % vocab
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(actual[:, :-1], baseline[:, :-1])
            self.assertGreater(float(np.max(np.abs(actual[:, -1] - baseline[:, -1]))), 1e-9)
            late[-1] = vocab
            actual[:] = -123
            self.assertNotEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(actual, -123)
            late[-1] = codes[-1, 2]
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(actual, baseline)
            self.assertEqual(calls["operations"][-1]["function"],
                             "attention_sliding_bounded_checked_f32")

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned weights unavailable; connected attention parity NOT_TESTED")
    def test_pinned_connected_attention(self):
        arrays = pinned_weights(Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"]))
        with np.load(MODEL) as fixture, np.load(CHECKPOINT) as oracle:
            codes = fixture["codec_tokens"].astype(np.int32)
            with tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, _, library = generated_entry(
                    Path(temp), arrays, len(codes), acoustic_codebooks=15,
                    pre_transformer_attention=True)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                actual = view(arena, acts, "first_attention_head", np.float32,
                              (16, len(codes), 64))
                expected = oracle["first_attention_head"]
                diff = np.abs(actual - expected)
                metrics = {
                    "max_abs": float(np.max(diff)),
                    "rmse": float(np.sqrt(np.mean(np.square(diff)))),
                    "worst_index": [int(i) for i in np.unravel_index(
                        int(np.argmax(diff)), diff.shape)],
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "scope": "reference codes through first sliding attention; waveform NOT_TESTED",
                }
                print("QWEN3_TTS_CODEC_ATTENTION " + json.dumps(metrics, sort_keys=True))
                self.assertTrue(np.isfinite(actual).all())
                self.assertLessEqual(metrics["max_abs"], 1e-5)


if __name__ == "__main__":
    unittest.main()
