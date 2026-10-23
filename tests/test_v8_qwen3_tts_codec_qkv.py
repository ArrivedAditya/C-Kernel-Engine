"""Connected split-RVQ → pre-convolution → first codec Q/K/V and RoPE."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_codec_semantic import (
    TABLE, PROJECTION, ACOUSTIC_PROJECTION, MODEL, generated_entry, view,
)
from tests.test_v8_qwen3_tts_codec_preconv import (
    CONV_WEIGHT, CONV_BIAS, fill_ids,
)
from version.v8.tts.build_qwen3_tts_codec_qkv_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "codec.decoder.pre_transformer"
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_qkv.npz")
LIMITS = {
    "transformer_input": 1e-6,
    "first_input_norm": 3e-6,
    "first_q": 5e-6,
    "first_k": 5e-6,
    "first_v": 2e-6,
    "first_v_head": 2e-6,
    "first_rope_q": 8e-6,
    "first_rope_k": 8e-6,
}


class Qwen3TTSCodecQKVTest(unittest.TestCase):
    def test_authored_circuit_and_fixture_integrity(self):
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_qkv_bounded.json"
        self.assertEqual(json.loads(path.read_text()), build_circuit())
        with self.assertRaises(ValueError):
            build_circuit(heads=16, head_dim=63)
        with self.assertRaises(ValueError):
            build_circuit(hidden_width=0)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        self.assertEqual(hashlib.sha256(MODEL.with_name(
            "qwen3_tts_codec_preconv.npz").read_bytes()).hexdigest(),
                         manifest["source_preconv_sha256"])
        self.assertEqual(manifest["model_revision"],
                         "85e237c12c027371202489a0ec509ded67b5e4b5")
        self.assertEqual(manifest["reference_revision"],
                         "022e286b98fbec7e1e916cb940cdf532cd9f488e")
        self.assertEqual((manifest["frames"], manifest["heads"],
                          manifest["head_dim"]), (46, 16, 64))
        with np.load(CHECKPOINT) as oracle:
            for name in ("transformer_input", "first_input_norm"):
                self.assertEqual(oracle[name].shape, (46, 512))
                self.assertTrue(np.isfinite(oracle[name]).all())
            for name in ("first_q", "first_k", "first_v"):
                self.assertEqual(oracle[name].shape, (46, 1024))
                self.assertTrue(np.isfinite(oracle[name]).all())
            for name in ("first_rope_q", "first_rope_k", "first_v_head"):
                self.assertEqual(oracle[name].shape, (16, 46, 64))
                self.assertTrue(np.isfinite(oracle[name]).all())

    def test_generated_nonzero_projection_and_causality(self):
        rng = np.random.default_rng(20261025)
        frames, vocab, width, output_width, latent = 5, 11, 4, 7, 3
        heads, head_dim, hidden = 2, 2, 4
        arrays = {
            TABLE: rng.normal(size=(vocab, width)).astype(np.float32),
            PROJECTION: rng.normal(size=(output_width, width)).astype(np.float32) * .1,
            ACOUSTIC_PROJECTION: rng.normal(
                size=(output_width, width)).astype(np.float32) * .1,
            CONV_WEIGHT: rng.normal(
                size=(latent, output_width, 3)).astype(np.float32) * .1,
            CONV_BIAS: rng.normal(size=latent).astype(np.float32) * .1,
            f"{PREFIX}.input_proj.weight": rng.normal(
                size=(hidden, latent)).astype(np.float32) * .1,
            f"{PREFIX}.input_proj.bias": rng.normal(size=hidden).astype(np.float32) * .1,
            f"{PREFIX}.layers.0.input_layernorm.weight": rng.normal(
                size=hidden).astype(np.float32),
        }
        for index in range(2):
            arrays[f"codec.acoustic.{index}.effective_embedding"] = rng.normal(
                size=(vocab, width)).astype(np.float32)
        for name in ("q", "k", "v"):
            arrays[f"{PREFIX}.layers.0.self_attn.{name}_proj.weight"] = rng.normal(
                size=(hidden, hidden)).astype(np.float32) * .1
        with tempfile.TemporaryDirectory() as temp:
            fn, arena, acts, calls, _ = generated_entry(
                Path(temp), arrays, frames, acoustic_codebooks=2,
                pre_transformer_qkv=True, qkv_heads=heads,
                qkv_head_dim=head_dim)
            codes = np.stack((np.arange(1, frames + 1),
                              np.arange(2, frames + 2),
                              np.arange(3, frames + 3)), axis=1).astype(np.int32)
            fill_ids(arena, acts, codes)
            self.assertEqual(fn(arena, len(arena)), 0)
            preconv = view(arena, acts, "preconv_output", np.float32,
                           (latent, frames))
            projected = view(arena, acts, "transformer_input", np.float32,
                             (frames, hidden))
            expected = (preconv.T @ arrays[f"{PREFIX}.input_proj.weight"].T
                        + arrays[f"{PREFIX}.input_proj.bias"])
            np.testing.assert_allclose(projected, expected, atol=1e-6, rtol=1e-6)
            normalized = view(arena, acts, "first_input_norm", np.float32,
                              (frames, hidden))
            expected_norm = expected * (
                np.mean(np.square(expected), axis=1, keepdims=True)
                + 1e-5) ** -.5
            expected_norm *= arrays[f"{PREFIX}.layers.0.input_layernorm.weight"]
            np.testing.assert_allclose(normalized, expected_norm, atol=2e-6, rtol=2e-6)
            for name in ("q", "k", "v"):
                actual = view(arena, acts, f"first_{name}", np.float32,
                              (frames, hidden))
                expected_rows = normalized @ arrays[
                    f"{PREFIX}.layers.0.self_attn.{name}_proj.weight"].T
                np.testing.assert_allclose(actual, expected_rows,
                                           atol=1e-6, rtol=1e-6)
            rope_q = view(arena, acts, "first_q_head", np.float32,
                          (heads, frames, head_dim))
            rope_k = view(arena, acts, "first_k_head", np.float32,
                          (heads, frames, head_dim))
            original_q = rope_q.copy()
            original_k = rope_k.copy()
            q_weight_name = f"{PREFIX}.layers.0.self_attn.q_proj.weight"
            q_entry = next(entry for entry in calls["memory"]["weights"]["entries"]
                           if entry["name"] == q_weight_name)
            q_weight = np.ndarray(arrays[q_weight_name].shape, dtype=np.float32,
                                  buffer=arena, offset=q_entry["abs_offset"])
            original_weight = float(q_weight[0, 0])
            q_weight[0, 0] = original_weight + 2.
            self.assertEqual(fn(arena, len(arena)), 0)
            self.assertGreater(float(np.max(np.abs(rope_q - original_q))), 1e-4)
            np.testing.assert_array_equal(rope_k, original_k)
            q_weight[0, 0] = original_weight
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(rope_q, original_q)
            late = view(arena, acts, "acoustic_ids_1", np.int32, (frames,))
            late[-1] = (int(late[-1]) + 1) % vocab
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(rope_q[:, :-1], original_q[:, :-1])
            self.assertGreater(float(np.max(np.abs(rope_q[:, -1] -
                                                   original_q[:, -1]))), 1e-6)
            late[-1] = vocab
            rope_q[:] = -123
            self.assertNotEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(rope_q, -123)
            late[-1] = codes[-1, 2]
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(rope_q, original_q)
            self.assertEqual(calls["operations"][-1]["function"],
                             "rope_qk_bounded_checked_f32")

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned model weights unavailable; connected Q/K/V parity NOT_TESTED")
    def test_pinned_connected_qkv(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
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
        with np.load(MODEL) as fixture, np.load(CHECKPOINT) as oracle:
            codes = fixture["codec_tokens"].astype(np.int32)
            for frames in (3, len(codes)):
                with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                    fn, arena, acts, calls, library = generated_entry(
                        Path(temp), arrays, frames, acoustic_codebooks=15,
                        pre_transformer_qkv=True)
                    fill_ids(arena, acts, codes[:frames])
                    self.assertEqual(fn(arena, len(arena)), 0)
                    errors = {}
                    rmses = {}
                    worst_indices = {}
                    for name, actual_name in (
                        ("transformer_input", "transformer_input"),
                        ("first_input_norm", "first_input_norm"),
                        ("first_q", "first_q"), ("first_k", "first_k"),
                        ("first_v", "first_v"),
                        ("first_v_head", "first_v_head"),
                        ("first_rope_q", "first_q_head"),
                        ("first_rope_k", "first_k_head"),
                    ):
                        shape = ((16, frames, 64) if name.endswith("_head")
                                 or name.startswith("first_rope") else
                                 (frames, 1024) if name in ("first_q", "first_k", "first_v")
                                 else (frames, 512))
                        result = view(arena, acts, actual_name, np.float32, shape)
                        expected = (oracle[name][..., :frames, :] if len(shape) == 3
                                    else oracle[name][:frames])
                        difference = np.abs(result - expected)
                        errors[name] = float(np.max(difference))
                        rmses[name] = float(np.sqrt(np.mean(np.square(difference))))
                        worst_indices[name] = [int(index) for index in
                                               np.unravel_index(
                            int(np.argmax(difference)), difference.shape)]
                        self.assertTrue(np.isfinite(result).all())
                        self.assertLessEqual(errors[name], LIMITS[name], name)
                    print("QWEN3_TTS_CODEC_QKV " + json.dumps({
                        "frames": frames, "max_abs_errors": errors,
                        "rmse": rmses, "worst_indices": worst_indices,
                        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                        "scope": "reference-code inputs through first Q/K/V RoPE; attention and waveform NOT_TESTED",
                    }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
