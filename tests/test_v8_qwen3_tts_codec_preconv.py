"""Generated split-RVQ → causal pre-convolution with independent checkpoints."""

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
from version.v8.tts.build_qwen3_tts_codec_preconv_circuit import (
    build_circuit,
)

ROOT = Path(__file__).resolve().parents[1]
CONV_WEIGHT = "codec.decoder.pre_conv.conv.weight"
CONV_BIAS = "codec.decoder.pre_conv.conv.bias"


def fill_ids(arena, acts, codes):
    frames = len(codes)
    view(arena, acts, "semantic_ids", np.int32, (frames,))[:] = codes[:, 0]
    for index in range(codes.shape[1] - 1):
        view(arena, acts, f"acoustic_ids_{index}", np.int32,
             (frames,))[:] = codes[:, index + 1]


class Qwen3TTSCodecPreconvTest(unittest.TestCase):
    def test_authored_circuit_is_current(self):
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_preconv_bounded.json"
        self.assertEqual(json.loads(path.read_text()), build_circuit())
        with self.assertRaises(ValueError):
            build_circuit(latent_width=0)
        with self.assertRaises(ValueError):
            build_circuit(frames=4097)

    def test_generated_nonzero_causal_edge_and_failure(self):
        rng = np.random.default_rng(20261024)
        frames, vocab, width, output_width, latent = 5, 11, 4, 7, 3
        arrays = {
            TABLE: rng.normal(size=(vocab, width)).astype(np.float32),
            PROJECTION: (rng.normal(size=(output_width, width)) * .1).astype(np.float32),
            ACOUSTIC_PROJECTION: (rng.normal(size=(output_width, width)) * .1).astype(np.float32),
            CONV_WEIGHT: (rng.normal(size=(latent, output_width, 3)) * .1).astype(np.float32),
            CONV_BIAS: (rng.normal(size=latent) * .1).astype(np.float32),
        }
        for index in range(2):
            arrays[f"codec.acoustic.{index}.effective_embedding"] = (
                rng.normal(size=(vocab, width)).astype(np.float32))
        with tempfile.TemporaryDirectory() as temp:
            fn, arena, acts, calls, _ = generated_entry(
                Path(temp), arrays, frames, acoustic_codebooks=2, pre_conv=True)
            self.assertEqual([op["function"] for op in calls["operations"][-3:]],
                ["transpose_strided_f32_checked",
                 "audio_zero_pad1d_left_channel_major_f32_checked",
                 "audio_conv1d_checked_channel_major_f32"])
            codes = np.stack([np.arange(1, frames + 1),
                              np.arange(2, frames + 2),
                              np.arange(3, frames + 3)], axis=1).astype(np.int32)
            fill_ids(arena, acts, codes)
            self.assertEqual(fn(arena, len(arena)), 0)
            rvq = view(arena, acts, "rvq_output", np.float32,
                       (frames, output_width))
            padded = view(arena, acts, "preconv_padded", np.float32,
                          (output_width, frames + 2))
            output = view(arena, acts, "preconv_output", np.float32,
                          (latent, frames))
            np.testing.assert_array_equal(padded[:, :2], 0.)
            np.testing.assert_array_equal(padded[:, 2:], rvq.T)
            expected = np.zeros_like(output)
            for oc in range(latent):
                for frame in range(frames):
                    value = float(arrays[CONV_BIAS][oc])
                    for ic in range(output_width):
                        for tap in range(3):
                            value += float(padded[ic, frame + tap]) * float(
                                arrays[CONV_WEIGHT][oc, ic, tap])
                    expected[oc, frame] = value
            np.testing.assert_allclose(output, expected, atol=1e-6, rtol=1e-6)
            baseline = output.copy()
            weight_entry = next(e for e in calls["memory"]["weights"]["entries"]
                                if e["name"] == CONV_WEIGHT)
            weight = np.ndarray(arrays[CONV_WEIGHT].shape, dtype=np.float32,
                                buffer=arena, offset=weight_entry["abs_offset"])
            original = float(weight[0, 0, 0])
            weight[0, 0, 0] = original + 2.
            self.assertEqual(fn(arena, len(arena)), 0)
            self.assertGreater(float(np.max(np.abs(output - baseline))), 1e-3)
            weight[0, 0, 0] = np.nan
            output[:] = -123.
            self.assertNotEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, -123.)
            weight[0, 0, 0] = original
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, baseline)
            # A wrong late codebook ID must stop before the pre-convolution.
            late = view(arena, acts, "acoustic_ids_1", np.int32, (frames,))
            late[-1] = vocab
            output[:] = -123.
            self.assertNotEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, -123.)
            late[-1] = codes[-1, 2]
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned codec weights unavailable; pre-convolution parity NOT_TESTED")
    def test_pinned_official_preconv_checkpoint(self):
        from safetensors import safe_open
        from version.v8.tts.capture_qwen3_tts_customvoice_oracle import verify_model_dir
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        checkpoint = MODEL.with_name("qwen3_tts_codec_preconv.npz")
        manifest = json.loads(checkpoint.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        self.assertEqual(manifest["model_revision"],
                         "85e237c12c027371202489a0ec509ded67b5e4b5")
        self.assertEqual(manifest["reference_revision"],
                         "022e286b98fbec7e1e916cb940cdf532cd9f488e")
        self.assertEqual(manifest["source_rvq_sha256"], hashlib.sha256(
            MODEL.with_name("qwen3_tts_codec_semantic.npz").read_bytes()).hexdigest())
        arrays = {}
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            for group, count in (("rvq_first", 1), ("rvq_rest", 15)):
                root = f"decoder.quantizer.{group}."
                for index in range(count):
                    layer = f"{root}vq.layers.{index}._codebook."
                    usage = source.get_tensor(layer + "cluster_usage").float().numpy()
                    summed = source.get_tensor(layer + "embedding_sum").float().numpy()
                    name = (TABLE if group == "rvq_first" else
                            f"codec.acoustic.{index}.effective_embedding")
                    arrays[name] = np.ascontiguousarray(
                        (summed / np.maximum(usage, 1e-5)[:, None]).astype(np.float32))
                projection = np.ascontiguousarray(source.get_tensor(
                    root + "output_proj.weight").float().numpy().reshape(512, 256))
                arrays[PROJECTION if group == "rvq_first" else
                       ACOUSTIC_PROJECTION] = projection
            for key in ("weight", "bias"):
                arrays[f"codec.decoder.pre_conv.conv.{key}"] = np.ascontiguousarray(
                    source.get_tensor(f"decoder.pre_conv.conv.{key}").float().numpy())
        with np.load(MODEL) as fixture, np.load(checkpoint) as oracle:
            codes = fixture["codec_tokens"].astype(np.int32)
            reference = oracle["preconv_output"]
            for frames in (3, len(codes)):
                with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                    fn, arena, acts, calls, so_path = generated_entry(
                        Path(temp), arrays, frames, acoustic_codebooks=15,
                        pre_conv=True)
                    self.assertEqual(len(calls["operations"]), 38)
                    fill_ids(arena, acts, codes[:frames])
                    self.assertEqual(fn(arena, len(arena)), 0)
                    result = view(arena, acts, "preconv_output", np.float32,
                                  (1024, frames))
                    difference = np.abs(result - reference[:, :frames])
                    error = float(np.max(difference))
                    self.assertTrue(np.isfinite(result).all())
                    self.assertLessEqual(error, 1e-5)
                    print("QWEN3_TTS_CODEC_PRECONV " + json.dumps({
                        "frames": frames, "max_abs_error": error,
                        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                        "library_sha256": hashlib.sha256(so_path.read_bytes()).hexdigest(),
                        "scope": "all codebooks through causal pre-convolution; transformer and waveform NOT_TESTED",
                    }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
