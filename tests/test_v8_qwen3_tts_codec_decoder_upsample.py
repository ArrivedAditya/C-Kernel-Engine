"""Generated reference-code-fed Qwen3-TTS first causal decoder upsample."""

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
from tests.test_v8_qwen3_tts_codec_snake_beta import ALPHA, BETA
from version.v8.tts.build_qwen3_tts_codec_decoder_upsample_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_decoder_upsample.npz")
WEIGHT = "codec.decoder.decoder.1.block.1.conv.weight"
BIAS = "codec.decoder.decoder.1.block.1.conv.bias"
FUNCTION = "audio_conv_transpose1d_dense_crop_right_channel_major_f32_checked"


class Qwen3TTSCodecDecoderUpsampleTest(unittest.TestCase):
    def test_circuit_and_fixture_integrity(self):
        graph = build_circuit()
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_decoder_upsample_bounded.json"
        self.assertEqual(json.loads(path.read_text()), graph)
        op = graph["block_types"]["codec_semantic"]["body"]["ops"][-1]
        self.assertEqual(op["id"], "decoder_upsample_output")
        self.assertEqual(op["graph_slots"]["inputs"]["input"],
                         "decoder_snake_beta_output")
        self.assertEqual(op["weight_refs"], {"weight": WEIGHT, "bias": BIAS})
        self.assertEqual(op["params"]["call_constants"]["cropped_deconv_right_crop"], 8)
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                build_circuit(decoder_upsample_rate=bad)
        for bad in (0, 3, True, 1.5):
            with self.assertRaises(ValueError):
                build_circuit(decoder_width=bad)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        prior = MODEL.with_name("qwen3_tts_codec_snake_beta.npz")
        self.assertEqual(hashlib.sha256(prior.read_bytes()).hexdigest(),
                         manifest["source_decoder_snake_beta_sha256"])
        prior_manifest = json.loads(prior.with_suffix(".json").read_text())
        for identity in ("model_revision", "reference_revision", "codec_source_sha256"):
            self.assertEqual(manifest[identity], prior_manifest[identity], identity)
        self.assertLessEqual(manifest["source_decoder_snake_beta_max_abs"], 1e-7)
        self.assertEqual(manifest["upsample_rate"], 8)
        self.assertEqual(manifest["right_crop"], 8)
        with np.load(CHECKPOINT) as oracle:
            self.assertEqual(set(oracle.files),
                             {"decoder_snake_beta_output", "decoder_upsample_output"})
            self.assertEqual(oracle["decoder_snake_beta_output"].shape, (1536, 184))
            self.assertEqual(oracle["decoder_upsample_output"].shape, (768, 1472))
            self.assertTrue(np.isfinite(oracle["decoder_upsample_output"]).all())

    @staticmethod
    def synthetic_arrays(rng):
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
        arrays[WEIGHT] = (rng.normal(size=(4, 2, 4)) * .1).astype(np.float32)
        arrays[BIAS] = (rng.normal(size=2) * .1).astype(np.float32)
        return arrays

    def test_small_generated_crop_routing_and_recovery(self):
        rng = np.random.default_rng(20261024)
        arrays = self.synthetic_arrays(rng)
        for frames in (1, 5):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, _ = generated_entry(
                    Path(temp), arrays, frames, acoustic_codebooks=2,
                    decoder_upsample=True, decoder_upsample_rate=2,
                    convnext_expansion=2, transformer_layers=2,
                    qkv_heads=2, qkv_head_dim=2, sliding_window=3)
                self.assertEqual(calls["operations"][-1]["function"], FUNCTION)
                codes = np.stack((np.arange(1, frames + 1),
                                  np.arange(2, frames + 2),
                                  np.arange(3, frames + 3)), axis=1).astype(np.int32)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                source = view(arena, acts, "decoder_snake_beta_output", np.float32,
                              (4, frames * 4))
                output = view(arena, acts, "decoder_upsample_output", np.float32,
                              (2, frames * 8))
                expected = np.tile(arrays[BIAS][:, None],
                                   (1, frames * 8)).astype(np.float64)
                for ic in range(4):
                    for frame in range(frames * 4):
                        for tap in range(4):
                            pos = frame * 2 + tap
                            if pos < frames * 8:
                                expected[:, pos] += (float(source[ic, frame]) *
                                                     arrays[WEIGHT][ic, :, tap])
                np.testing.assert_allclose(output, expected, atol=3e-6, rtol=3e-6)
                baseline = output.copy()
                entries = {e["name"]: e for e in calls["memory"]["weights"]["entries"]}
                bound = np.ndarray(arrays[WEIGHT].shape, np.float32, buffer=arena,
                                   offset=entries[WEIGHT]["abs_offset"])
                saved = bound.copy()
                bound[0, 0, 0] += 2
                self.assertEqual(fn(arena, len(arena)), 0)
                self.assertGreater(float(np.max(np.abs(output - baseline))), 1e-5)
                bound[0, 0, 0] = np.nan
                output[:] = -123
                self.assertNotEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, -123)
                bound[:] = saved
                self.assertNotEqual(fn(arena, len(arena) - 1), 0)
                np.testing.assert_array_equal(output, -123)
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned weights unavailable; connected decoder upsample NOT_TESTED")
    def test_pinned_connected_decoder_upsample(self):
        from safetensors import safe_open
        import torch

        torch.set_num_threads(4)
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        arrays = pinned_pretransformer_weights(model_dir)
        names = (UPSAMPLE_WEIGHT, UPSAMPLE_BIAS, DEPTHWISE_WEIGHT,
                 DEPTHWISE_BIAS, *FIRST_WEIGHT_NAMES,
                 SECOND_UPSAMPLE_WEIGHT, SECOND_UPSAMPLE_BIAS,
                 *SECOND_WEIGHT_NAMES, CONV_WEIGHT, CONV_BIAS,
                 ALPHA, BETA, WEIGHT, BIAS)
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
                    decoder_upsample=True)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                input_value = view(arena, acts, "decoder_snake_beta_output",
                                   np.float32, oracle["decoder_snake_beta_output"].shape)
                actual = view(arena, acts, "decoder_upsample_output", np.float32,
                              oracle["decoder_upsample_output"].shape)
                reference = oracle["decoder_upsample_output"]
                error = np.abs(actual - reference)
                worst_index = np.unravel_index(int(np.argmax(error)), error.shape)
                with torch.inference_mode():
                    same_input = torch.nn.functional.conv_transpose1d(
                        torch.from_numpy(input_value.copy()[None]),
                        torch.from_numpy(arrays[WEIGHT]),
                        torch.from_numpy(arrays[BIAS]), stride=8)[0, :, :-8].numpy()
                metrics = {
                    "input_max_abs": float(np.max(np.abs(
                        input_value - oracle["decoder_snake_beta_output"]))),
                    "connected_max_abs": float(np.max(error)),
                    "connected_rmse": float(np.sqrt(np.mean(np.square(
                        actual.astype(np.float64) - reference.astype(np.float64))))),
                    "same_input_local_max_abs": float(np.max(np.abs(actual - same_input))),
                    "worst": {"index": list(map(int, worst_index)),
                              "actual": float(actual[worst_index]),
                              "reference": float(reference[worst_index])},
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "scope": "reference codes through right-cropped first decoder upsample; waveform NOT_TESTED",
                }
                print("QWEN3_TTS_CODEC_DECODER_UPSAMPLE " + json.dumps(metrics, sort_keys=True))
                self.assertTrue(np.isfinite(actual).all())
                self.assertLessEqual(metrics["input_max_abs"], 2e-5)
                self.assertLessEqual(metrics["connected_max_abs"], 1e-5)
                self.assertLessEqual(metrics["same_input_local_max_abs"], 1e-5)


if __name__ == "__main__":
    unittest.main()
