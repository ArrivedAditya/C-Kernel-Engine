"""Generated Qwen3-TTS codec through the first causal vocoder upsample."""

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
from version.v8.tts.build_qwen3_tts_codec_first_upsample_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_first_upsample.npz")
WEIGHT = "codec.decoder.upsample.0.0.conv.weight"
BIAS = "codec.decoder.upsample.0.0.conv.bias"


class Qwen3TTSCodecFirstUpsampleTest(unittest.TestCase):
    def test_authored_circuit_and_fixture_integrity(self):
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_first_upsample_bounded.json"
        self.assertEqual(json.loads(path.read_text()), build_circuit())
        graph = build_circuit()
        ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
        self.assertEqual(ops[-2]["id"], "decoder_channel")
        self.assertEqual(ops[-1]["id"], "first_upsample")
        self.assertEqual(ops[-1]["weight_refs"], {"weight": WEIGHT, "bias": BIAS})
        self.assertEqual(ops[-1]["params"]["call_constants"]["dense_deconv_padding"], 0)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        prior = MODEL.with_name("qwen3_tts_codec_pretransformer.npz")
        self.assertEqual(hashlib.sha256(prior.read_bytes()).hexdigest(),
                         manifest["source_pretransformer_sha256"])
        self.assertLessEqual(manifest["source_pretransformer_input_max_abs"], 1e-7)
        self.assertEqual((manifest["frames"], manifest["upsampled_frames"],
                          manifest["latent_width"]), (46, 92, 1024))
        with np.load(CHECKPOINT) as oracle:
            self.assertEqual(set(oracle.files), {"decoder_channel", "first_upsample"})
            self.assertEqual(oracle["decoder_channel"].shape, (1024, 46))
            self.assertEqual(oracle["first_upsample"].shape, (1024, 92))
            self.assertTrue(np.isfinite(oracle["first_upsample"]).all())

    def test_generated_causal_upsample_and_recovery(self):
        rng = np.random.default_rng(20261024)
        arrays = synthetic_weights(rng, vocab=12)
        add_synthetic_weights(arrays, rng, 2)
        arrays[WEIGHT] = (rng.normal(size=(3, 3, 2)) * .1).astype(np.float32)
        arrays[BIAS] = (rng.normal(size=3) * .1).astype(np.float32)
        for frames in (1, 5):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, _ = generated_entry(
                    Path(temp), arrays, frames, acoustic_codebooks=2,
                    first_decoder_upsample=True, transformer_layers=2,
                    qkv_heads=2, qkv_head_dim=2, sliding_window=3)
                codes = np.stack((np.arange(1, frames + 1),
                                  np.arange(2, frames + 2),
                                  np.arange(3, frames + 3)), axis=1).astype(np.int32)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                latent = view(arena, acts, "decoder_output", np.float32,
                              (frames, 3))
                channels = view(arena, acts, "decoder_channel", np.float32,
                                (3, frames))
                np.testing.assert_array_equal(channels, latent.T)
                output = view(arena, acts, "first_upsample", np.float32,
                              (3, frames * 2))
                expected = np.zeros_like(output)
                expected[:] = arrays[BIAS][:, None]
                for frame in range(frames):
                    for tap in range(2):
                        expected[:, frame * 2 + tap] += (
                            channels[:, frame] @ arrays[WEIGHT][:, :, tap])
                np.testing.assert_allclose(output, expected, atol=2e-6, rtol=2e-6)
                baseline = output.copy()
                if frames > 1:
                    ids = view(arena, acts, "acoustic_ids_1", np.int32, (frames,))
                    ids[-1] = (int(ids[-1]) + 1) % 12
                    self.assertEqual(fn(arena, len(arena)), 0)
                    np.testing.assert_array_equal(output[:, :-2], baseline[:, :-2])
                    self.assertGreater(float(np.max(np.abs(
                        output[:, -2:] - baseline[:, -2:]))), 1e-9)
                    ids[-1] = codes[-1, 2]
                entry = next(item for item in calls["memory"]["weights"]["entries"]
                             if item["name"] == WEIGHT)
                bound = np.ndarray(arrays[WEIGHT].shape, dtype=np.float32,
                                   buffer=arena, offset=entry["abs_offset"])
                saved = bound.copy()
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
                         "pinned weights unavailable; direct full-model vocoder parity NOT_TESTED")
    def test_pinned_connected_first_upsample(self):
        from safetensors import safe_open
        import torch

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        arrays = pinned_pretransformer_weights(model_dir)
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            for name in (WEIGHT, BIAS):
                arrays[name] = np.ascontiguousarray(
                    source.get_tensor(name.removeprefix("codec.")).float().numpy())
        with np.load(MODEL) as fixture, np.load(CHECKPOINT) as oracle:
            codes = fixture["codec_tokens"].astype(np.int32)
            with tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, _, library = generated_entry(
                    Path(temp), arrays, len(codes), acoustic_codebooks=15,
                    first_decoder_upsample=True)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                errors, rmses, worst = {}, {}, {}
                for name in ("decoder_channel", "first_upsample"):
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
                actual_input = view(arena, acts, "decoder_channel", np.float32,
                                    oracle["decoder_channel"].shape)
                actual_output = view(arena, acts, "first_upsample", np.float32,
                                     oracle["first_upsample"].shape)
                with torch.inference_mode():
                    same_input_oracle = torch.nn.functional.conv_transpose1d(
                        torch.from_numpy(actual_input.copy()[None]),
                        torch.from_numpy(arrays[WEIGHT]),
                        torch.from_numpy(arrays[BIAS]), stride=2)[0].numpy()
                local_error = float(np.max(np.abs(
                    actual_output - same_input_oracle)))
                print("QWEN3_TTS_CODEC_FIRST_UPSAMPLE " + json.dumps({
                    "max_abs_errors": errors, "rmse": rmses, "worst": worst,
                    "same_input_local_max_abs": local_error,
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "scope": "reference codec codes through first causal factor-2 upsample; ConvNeXt and waveform NOT_TESTED",
                }, sort_keys=True))
                for name, limit in LIMITS.items():
                    self.assertLessEqual(errors[name], limit, name)
                self.assertLessEqual(local_error, 5e-6)


LIMITS = {"decoder_channel": 1e-6, "first_upsample": 6e-6}


if __name__ == "__main__":
    unittest.main()
