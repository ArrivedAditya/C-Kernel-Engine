"""Generated reference-code codec through the causal decoder ingress."""

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
from version.v8.tts.build_qwen3_tts_codec_decoder_ingress_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_decoder_ingress.npz")
CONV_WEIGHT = "codec.decoder.decoder.0.conv.weight"
CONV_BIAS = "codec.decoder.decoder.0.conv.bias"


class Qwen3TTSCodecDecoderIngressTest(unittest.TestCase):
    def test_circuit_and_fixture_integrity(self):
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_decoder_ingress_bounded.json"
        graph = build_circuit()
        self.assertEqual(json.loads(path.read_text()), graph)
        ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
        self.assertEqual([op["id"] for op in ops[-2:]],
                         ["decoder_ingress_left_pad", "decoder_ingress_output"])
        self.assertEqual(ops[-2]["graph_slots"]["inputs"]["input"],
                         "second_convnext_output")
        self.assertEqual(ops[-1]["graph_slots"]["inputs"]["input"],
                         "decoder_ingress_padded")
        self.assertEqual(ops[-1]["weight_refs"],
                         {"weight": CONV_WEIGHT, "bias": CONV_BIAS})
        self.assertEqual(ops[-1]["params"]["call_constants"]["conv_input_frames"], 190)
        self.assertEqual(ops[-1]["params"]["call_constants"]["conv_output_frames"], 184)
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                build_circuit(decoder_width=bad)
            with self.assertRaises(ValueError):
                build_circuit(ingress_kernel=bad)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        prior = MODEL.with_name("qwen3_tts_codec_second_convnext.npz")
        self.assertEqual(hashlib.sha256(prior.read_bytes()).hexdigest(),
                         manifest["source_second_convnext_sha256"])
        prior_manifest = json.loads(prior.with_suffix(".json").read_text())
        for identity in ("model_revision", "reference_revision", "codec_source_sha256"):
            self.assertEqual(manifest[identity], prior_manifest[identity], identity)
        self.assertLessEqual(manifest["source_second_convnext_max_abs"], 1e-7)
        with np.load(CHECKPOINT) as oracle:
            self.assertEqual(set(oracle.files),
                             {"second_convnext_output", "decoder_ingress_output"})
            self.assertEqual(oracle["second_convnext_output"].shape, (1024, 184))
            self.assertEqual(oracle["decoder_ingress_output"].shape, (1536, 184))
            self.assertTrue(np.isfinite(oracle["decoder_ingress_output"]).all())

    def test_small_generated_causal_routing_and_recovery(self):
        rng = np.random.default_rng(20261024)
        arrays = synthetic_weights(rng, vocab=12)
        add_synthetic_weights(arrays, rng, 2)
        add_synthetic_convnext_weights(arrays, rng)
        arrays[SECOND_UPSAMPLE_WEIGHT] = (
            rng.normal(size=(3, 3, 2)) * .1).astype(np.float32)
        arrays[SECOND_UPSAMPLE_BIAS] = (
            rng.normal(size=3) * .1).astype(np.float32)
        add_synthetic_second_weights(arrays)
        arrays[CONV_WEIGHT] = (
            rng.normal(size=(4, 3, 7)) * .1).astype(np.float32)
        arrays[CONV_BIAS] = (
            rng.normal(size=4) * .1).astype(np.float32)
        for frames in (1, 5):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, _ = generated_entry(
                    Path(temp), arrays, frames, acoustic_codebooks=2,
                    decoder_ingress=True, convnext_expansion=2,
                    transformer_layers=2, qkv_heads=2, qkv_head_dim=2,
                    sliding_window=3)
                self.assertEqual([op["function"] for op in calls["operations"][-2:]],
                    ["audio_zero_pad1d_left_channel_major_f32_checked",
                     "audio_conv1d_checked_channel_major_f32"])
                codes = np.stack((np.arange(1, frames + 1),
                                  np.arange(2, frames + 2),
                                  np.arange(3, frames + 3)), axis=1).astype(np.int32)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                length = frames * 4
                source = view(arena, acts, "second_convnext_output", np.float32,
                              (3, length))
                padded = view(arena, acts, "decoder_ingress_padded", np.float32,
                              (3, length + 6))
                output = view(arena, acts, "decoder_ingress_output", np.float32,
                              (4, length))
                np.testing.assert_array_equal(padded[:, :6], 0)
                np.testing.assert_array_equal(padded[:, 6:], source)
                expected = np.zeros_like(output)
                for oc in range(4):
                    for frame in range(length):
                        value = float(arrays[CONV_BIAS][oc])
                        for ic in range(3):
                            for tap in range(7):
                                value += float(padded[ic, frame + tap]) * float(
                                    arrays[CONV_WEIGHT][oc, ic, tap])
                        expected[oc, frame] = value
                np.testing.assert_allclose(output, expected, atol=2e-6, rtol=2e-6)
                baseline = output.copy()
                first = source.copy()
                entry = next(e for e in calls["memory"]["weights"]["entries"]
                             if e["name"] == CONV_WEIGHT)
                weight = np.ndarray(arrays[CONV_WEIGHT].shape, dtype=np.float32,
                                    buffer=arena, offset=entry["abs_offset"])
                saved = weight.copy()
                weight[0, 0, 6] += 2
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(
                    view(arena, acts, "second_convnext_output", np.float32,
                         (3, length)), first)
                self.assertGreater(float(np.max(np.abs(output - baseline))), 1e-4)
                weight[0, 0, 6] = np.nan
                output[:] = -123
                self.assertNotEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, -123)
                weight[:] = saved
                self.assertNotEqual(fn(arena, len(arena) - 1), 0)
                np.testing.assert_array_equal(output, -123)
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned weights unavailable; connected decoder ingress NOT_TESTED")
    def test_pinned_connected_decoder_ingress(self):
        from safetensors import safe_open
        import torch

        torch.set_num_threads(4)
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        arrays = pinned_pretransformer_weights(model_dir)
        names = (UPSAMPLE_WEIGHT, UPSAMPLE_BIAS,
                 DEPTHWISE_WEIGHT, DEPTHWISE_BIAS, *FIRST_WEIGHT_NAMES,
                 SECOND_UPSAMPLE_WEIGHT, SECOND_UPSAMPLE_BIAS,
                 *SECOND_WEIGHT_NAMES, CONV_WEIGHT, CONV_BIAS)
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
                    decoder_ingress=True)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                source_value = view(arena, acts, "second_convnext_output",
                                    np.float32, oracle["second_convnext_output"].shape)
                actual = view(arena, acts, "decoder_ingress_output",
                              np.float32, oracle["decoder_ingress_output"].shape)
                reference = oracle["decoder_ingress_output"]
                input_error = float(np.max(np.abs(
                    source_value - oracle["second_convnext_output"])))
                error = np.abs(actual - reference)
                worst_index = np.unravel_index(int(np.argmax(error)), error.shape)
                with torch.inference_mode():
                    same_input = torch.nn.functional.conv1d(
                        torch.nn.functional.pad(
                            torch.from_numpy(source_value.copy()[None]), (6, 0)),
                        torch.from_numpy(arrays[CONV_WEIGHT]),
                        torch.from_numpy(arrays[CONV_BIAS]))[0].numpy()
                    reference_input = torch.nn.functional.conv1d(
                        torch.nn.functional.pad(
                            torch.from_numpy(oracle["second_convnext_output"].copy()[None]),
                            (6, 0)),
                        torch.from_numpy(arrays[CONV_WEIGHT]),
                        torch.from_numpy(arrays[CONV_BIAS]))[0].numpy()
                metrics = {
                    "input_max_abs": input_error,
                    "connected_max_abs": float(np.max(error)),
                    "connected_rmse": float(np.sqrt(np.mean(np.square(
                        actual.astype(np.float64) - reference.astype(np.float64))))),
                    "same_input_local_max_abs": float(np.max(np.abs(actual - same_input))),
                    "reference_input_local_max_abs": float(np.max(np.abs(
                        reference_input - reference))),
                    "worst": {"index": list(map(int, worst_index)),
                              "actual": float(actual[worst_index]),
                              "reference": float(reference[worst_index])},
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "scope": "reference codec codes through decoder ingress; waveform NOT_TESTED",
                }
                print("QWEN3_TTS_CODEC_DECODER_INGRESS " + json.dumps(
                    metrics, sort_keys=True))
                self.assertTrue(np.isfinite(actual).all())
                self.assertLessEqual(input_error, 6e-5)
                self.assertLessEqual(metrics["connected_max_abs"], 2e-5)
                self.assertLessEqual(metrics["same_input_local_max_abs"], 1e-5)
                self.assertLessEqual(metrics["reference_input_local_max_abs"], 1e-6)


if __name__ == "__main__":
    unittest.main()
