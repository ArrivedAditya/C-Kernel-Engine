"""Generated Qwen3-TTS codec through the first decoder SnakeBeta."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_codec_semantic import MODEL, generated_entry, view
from tests.test_v8_qwen3_tts_codec_preconv import fill_ids
from tests.test_v8_qwen3_tts_codec_attention import synthetic_weights
from tests.test_v8_qwen3_tts_codec_pretransformer import (
    add_synthetic_weights, pinned_pretransformer_weights,
)
from tests.test_v8_qwen3_tts_codec_convnext_complete import (
    WEIGHT_NAMES as FIRST_WEIGHT_NAMES, UPSAMPLE_WEIGHT, UPSAMPLE_BIAS,
    DEPTHWISE_WEIGHT, DEPTHWISE_BIAS, add_synthetic_convnext_weights,
)
from tests.test_v8_qwen3_tts_codec_second_upsample import (
    WEIGHT as SECOND_UPSAMPLE_WEIGHT, BIAS as SECOND_UPSAMPLE_BIAS,
)
from tests.test_v8_qwen3_tts_codec_second_convnext import (
    SECOND_WEIGHT_NAMES, add_synthetic_second_weights,
)
from tests.test_v8_qwen3_tts_codec_decoder_ingress import CONV_WEIGHT, CONV_BIAS
from version.v8.tts.build_qwen3_tts_codec_snake_beta_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_snake_beta.npz")
ALPHA = "codec.decoder.decoder.1.block.0.alpha"
BETA = "codec.decoder.decoder.1.block.0.beta"


class Qwen3TTSCodecSnakeBetaTest(unittest.TestCase):
    def test_circuit_and_fixture_integrity(self):
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_snake_beta_bounded.json"
        graph = build_circuit()
        self.assertEqual(json.loads(path.read_text()), graph)
        op = graph["block_types"]["codec_semantic"]["body"]["ops"][-1]
        self.assertEqual(op["id"], "decoder_snake_beta_output")
        self.assertEqual(op["graph_slots"]["inputs"]["input"],
                         "decoder_ingress_output")
        self.assertEqual(op["weight_refs"], {"alpha": ALPHA, "beta": BETA})
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                build_circuit(decoder_width=bad)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        prior = MODEL.with_name("qwen3_tts_codec_decoder_ingress.npz")
        self.assertEqual(hashlib.sha256(prior.read_bytes()).hexdigest(),
                         manifest["source_decoder_ingress_sha256"])
        prior_manifest = json.loads(prior.with_suffix(".json").read_text())
        for identity in ("model_revision", "reference_revision", "codec_source_sha256"):
            self.assertEqual(manifest[identity], prior_manifest[identity], identity)
        self.assertLessEqual(manifest["source_decoder_ingress_max_abs"], 1e-7)
        with np.load(CHECKPOINT) as oracle:
            self.assertEqual(set(oracle.files),
                             {"decoder_ingress_output", "decoder_snake_beta_output"})
            for name in oracle.files:
                self.assertEqual(oracle[name].shape, (1536, 184))
                self.assertTrue(np.isfinite(oracle[name]).all())

    def test_small_generated_two_parameter_routing_and_recovery(self):
        rng = np.random.default_rng(20261024)
        arrays = synthetic_weights(rng, vocab=12)
        add_synthetic_weights(arrays, rng, 2)
        add_synthetic_convnext_weights(arrays, rng)
        arrays[SECOND_UPSAMPLE_WEIGHT] = (rng.normal(size=(3, 3, 2)) * .1).astype(np.float32)
        arrays[SECOND_UPSAMPLE_BIAS] = (rng.normal(size=3) * .1).astype(np.float32)
        add_synthetic_second_weights(arrays)
        arrays[CONV_WEIGHT] = (rng.normal(size=(4, 3, 7)) * .1).astype(np.float32)
        arrays[CONV_BIAS] = (rng.normal(size=4) * .1).astype(np.float32)
        arrays[ALPHA] = np.array([-.5, 0, .3, .7], dtype=np.float32)
        arrays[BETA] = np.array([.7, -.2, .4, -.6], dtype=np.float32)
        for frames in (1, 5):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, _ = generated_entry(
                    Path(temp), arrays, frames, acoustic_codebooks=2,
                    decoder_snake_beta=True, convnext_expansion=2,
                    transformer_layers=2, qkv_heads=2, qkv_head_dim=2,
                    sliding_window=3)
                self.assertEqual(calls["operations"][-1]["function"],
                                 "audio_snake_beta_strided_f32_checked")
                codes = np.stack((np.arange(1, frames + 1),
                                  np.arange(2, frames + 2),
                                  np.arange(3, frames + 3)), axis=1).astype(np.int32)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                length = frames * 4
                source = view(arena, acts, "decoder_ingress_output", np.float32,
                              (4, length))
                output = view(arena, acts, "decoder_snake_beta_output", np.float32,
                              (4, length))
                expected = (source + np.sin(
                    source * np.exp(arrays[ALPHA])[:, None]) ** 2 /
                    (np.exp(arrays[BETA])[:, None] + np.float32(1e-9)))
                np.testing.assert_allclose(output, expected, atol=3e-6, rtol=3e-6)
                baseline = output.copy()
                weights = {e["name"]: e for e in calls["memory"]["weights"]["entries"]}
                beta = np.ndarray((4,), np.float32, buffer=arena,
                                  offset=weights[BETA]["abs_offset"])
                saved = beta.copy()
                beta[0] += 1.0
                self.assertEqual(fn(arena, len(arena)), 0)
                self.assertGreater(float(np.max(np.abs(output - baseline))), 1e-5)
                beta[0] = np.nan
                output[:] = -123
                self.assertNotEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, -123)
                beta[:] = saved
                self.assertNotEqual(fn(arena, len(arena) - 1), 0)
                np.testing.assert_array_equal(output, -123)
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned weights unavailable; connected SnakeBeta NOT_TESTED")
    def test_pinned_connected_snake_beta(self):
        from safetensors import safe_open
        import torch

        torch.set_num_threads(4)
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        arrays = pinned_pretransformer_weights(model_dir)
        names = (UPSAMPLE_WEIGHT, UPSAMPLE_BIAS, DEPTHWISE_WEIGHT,
                 DEPTHWISE_BIAS, *FIRST_WEIGHT_NAMES,
                 SECOND_UPSAMPLE_WEIGHT, SECOND_UPSAMPLE_BIAS,
                 *SECOND_WEIGHT_NAMES, CONV_WEIGHT, CONV_BIAS, ALPHA, BETA)
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            for name in names:
                arrays[name] = np.ascontiguousarray(
                    source.get_tensor(name.removeprefix("codec.")).float().numpy())
        with np.load(MODEL) as fixture, np.load(CHECKPOINT) as oracle:
            codes = fixture["codec_tokens"].astype(np.int32)
            with tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, _, library = generated_entry(
                    Path(temp), arrays, len(codes), acoustic_codebooks=15,
                    decoder_snake_beta=True)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                input_value = view(arena, acts, "decoder_ingress_output", np.float32,
                                   oracle["decoder_ingress_output"].shape)
                actual = view(arena, acts, "decoder_snake_beta_output", np.float32,
                              oracle["decoder_snake_beta_output"].shape)
                reference = oracle["decoder_snake_beta_output"]
                error = np.abs(actual - reference)
                worst_index = np.unravel_index(int(np.argmax(error)), error.shape)
                with torch.inference_mode():
                    x = torch.from_numpy(input_value.copy()[None])
                    a = torch.from_numpy(arrays[ALPHA].copy())[None, :, None]
                    b = torch.from_numpy(arrays[BETA].copy())[None, :, None]
                    same_input = (x + torch.sin(x * torch.exp(a)).pow(2) /
                                  (torch.exp(b) + 1e-9))[0].numpy()
                metrics = {
                    "input_max_abs": float(np.max(np.abs(
                        input_value - oracle["decoder_ingress_output"]))),
                    "connected_max_abs": float(np.max(error)),
                    "connected_rmse": float(np.sqrt(np.mean(np.square(
                        actual.astype(np.float64) - reference.astype(np.float64))))),
                    "same_input_local_max_abs": float(np.max(np.abs(actual - same_input))),
                    "worst": {"index": list(map(int, worst_index)),
                              "actual": float(actual[worst_index]),
                              "reference": float(reference[worst_index])},
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "scope": "reference codes through first decoder SnakeBeta; waveform NOT_TESTED",
                }
                print("QWEN3_TTS_CODEC_SNAKE_BETA " + json.dumps(metrics, sort_keys=True))
                self.assertTrue(np.isfinite(actual).all())
                self.assertLessEqual(metrics["input_max_abs"], 2e-5)
                self.assertLessEqual(metrics["connected_max_abs"], 3e-5)
                self.assertLessEqual(metrics["same_input_local_max_abs"], 5e-6)


if __name__ == "__main__":
    unittest.main()
