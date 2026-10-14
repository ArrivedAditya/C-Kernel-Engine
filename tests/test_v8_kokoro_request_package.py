"""Offline Kokoro request package bounds and BUMP binding contracts."""

import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/tts"))
import export_kokoro_bump as bump_export
import export_kokoro_request_package as request_export
import build_kokoro_request_runtime as runtime_build


def _buffer(name, offset, size, dtype="fp32"):
    return {"name": name, "abs_offset": offset, "size": size,
            "dtype": dtype}


class KokoroRequestPackageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bump_dir = self.root / "bump"
        values = np.array([1., -2., 3., -4.], np.float32)
        bump_export.write_bundle(self.bump_dir, {"weight.a": values},
            {"weight.a": {"source_name": "synthetic.weight.a",
                           "transform": "identity"}},
            {"n_token": 178, "hidden_dim": 512,
             "plbert": {"intermediate_size": 2048,
                        "max_position_embeddings": 512,
                        "num_attention_heads": 12}},
            {"source": "synthetic package regression"})
        entry = json.loads((self.bump_dir / "weights_manifest.json").read_text())["entries"][0]
        self.layout = {"memory": {
            "arena": {"total_size": 4_000_000},
            "weights": {"entries": [_buffer("weight.a", 64, 16)]},
            "activations": {"buffers": [
                _buffer("word_ids", 128, 32, "i32"),
                _buffer("predictor_style", 256, 512),
                _buffer("decoder_style", 768, 512),
                _buffer("source_gaussian", 2048, 76800 * 9 * 4),
                _buffer("source_stft_window", 2_768_000, 80),
                _buffer("source_stft_cos", 2_768_128, 880),
                _buffer("source_stft_sin", 2_769_024, 880),
                _buffer("generator_waveform", 3_000_000, 80),
            ]},
        }}
        self.catalog = {
            "schema": "cke.kokoro.request_weight_bindings.v1",
            "source_manifest_sha256": request_export.sha256_file(
                self.bump_dir / "weights_manifest.json"),
            "source_bump_sha256": request_export.sha256_file(
                self.bump_dir / "weights.bump"),
            "weights": {"weight.a": {"sha256": entry["sha256"],
                                     "source": ["weight.a"]}},
            "constants_fp32_le_hex": {
                "source_stft_window": bytes(80).hex(),
                "source_stft_cos": bytes(880).hex(),
                "source_stft_sin": bytes(880).hex(),
            },
        }
        self.layout_path = self.root / "layout.json"
        self.catalog_path = self.root / "bindings.json"
        self.output = self.root / "request.cktts"

    def build(self):
        self.layout_path.write_text(json.dumps(self.layout))
        self.catalog_path.write_text(json.dumps(self.catalog))
        return request_export.package_request(self.layout_path, self.bump_dir,
                                              self.output, self.catalog_path)

    def test_sparse_package_contains_only_weights_and_constants(self):
        report = self.build()
        data = self.output.read_bytes()
        self.assertEqual(data[:8], b"CKTTS003")
        self.assertEqual(struct.unpack_from("<9QI", data, 8), (
            4_000_000, 3_000_000, 20, 2048, 76800 * 9,
            128, 8, 256, 768, 4))
        cursor = 8 + struct.calcsize("<9QI")
        segments = []
        for _ in range(4):
            offset, size = struct.unpack_from("<QQ", data, cursor)
            cursor += 16
            segments.append((offset, data[cursor:cursor + size]))
            cursor += size
        self.assertEqual(cursor, len(data))
        self.assertEqual(segments[0], (64, np.array([1., -2., 3., -4.],
            dtype="<f4").tobytes()))
        self.assertEqual([offset for offset, _ in segments[1:]],
                         [2_768_000, 2_768_128, 2_769_024])
        self.assertEqual(report["phoneme_count"], 8)
        first_hash = report["package_sha256"]
        self.assertEqual(self.build()["package_sha256"], first_hash)
        self.assertEqual(hashlib.sha256(data).hexdigest(), first_hash)
        print("CKE_NUMERICAL_CASE " + json.dumps({
            "case_id": "kokoro.request-package.static-bump-v1",
            "name": "Kokoro static request package from verified BUMP",
            "provider": "offline_kokoro_request_package", "dtype": "fp32",
            "direction": "inference", "oracle": "independently decoded synthetic BUMP bytes",
            "status": "pass", "gate": "session", "blocking": True,
            "max_diff": 0, "tolerance": 0,
            "configuration": "8-ID sparse CKTTS003 package, four static segments",
            "reproduction_command": "python3 -m unittest tests.test_v8_kokoro_request_package",
        }, sort_keys=True))

    def test_rejects_wrong_geometry_and_artifact_identity(self):
        self.build()
        original = self.output.read_bytes()
        cases = []
        bad = copy.deepcopy(self.layout)
        bad["memory"]["activations"]["buffers"][-1]["abs_offset"] = 2048
        cases.append((bad, self.catalog, "overlapping output"))
        bad = copy.deepcopy(self.layout)
        bad["memory"]["activations"]["buffers"][0]["size"] = 164
        cases.append((bad, self.catalog, "unsupported tokens"))
        bad = copy.deepcopy(self.layout)
        bad["memory"]["weights"]["entries"][0]["size"] = 12
        cases.append((bad, self.catalog, "weight geometry"))
        bad_catalog = copy.deepcopy(self.catalog)
        bad_catalog["source_bump_sha256"] = "0" * 64
        cases.append((self.layout, bad_catalog, "BUMP identity"))
        bad = copy.deepcopy(self.layout)
        bad["memory"]["activations"]["buffers"][0]["dtype"] = "fp32"
        cases.append((bad, self.catalog, "ID dtype"))
        bad = copy.deepcopy(self.layout)
        bad["memory"]["activations"]["buffers"].append(
            copy.deepcopy(bad["memory"]["activations"]["buffers"][0]))
        cases.append((bad, self.catalog, "duplicate activation"))
        for layout, catalog, label in cases:
            with self.subTest(label=label):
                self.layout, self.catalog = layout, catalog
                with self.assertRaises(ValueError):
                    self.build()
                self.assertEqual(self.output.read_bytes(), original)

    def test_compiler_manifest_uses_pinned_bindings_without_fixture_weights(self):
        source = {"entries": [
            {"name": "source.a", "dtype": "fp32", "shape": [2, 3], "size": 24},
            {"name": "source.fwd", "dtype": "fp32", "shape": [4], "size": 16},
            {"name": "source.rev", "dtype": "fp32", "shape": [4], "size": 16},
        ]}
        catalog = {"schema": "cke.kokoro.request_weight_bindings.v1",
            "source_manifest_sha256": "manifest", "source_bump_sha256": "bump",
            "weights": {
                "graph.a": {"source": ["source.a"], "sha256": "a" * 64},
                "graph.scan": {"source": ["source.fwd", "source.rev"],
                               "transform": "concatenate-forward-reverse",
                               "sha256": "b" * 64}}}
        (self.bump_dir / "weights_manifest.json").write_text("stub")
        self.catalog_path.write_text(json.dumps(catalog))
        with mock.patch.object(runtime_build, "sha256_file",
                               side_effect=["manifest", "bump"]), \
                mock.patch.object(runtime_build, "verify_bundle", return_value=source):
            manifest = runtime_build.compiler_manifest(self.bump_dir, 8,
                                                       self.catalog_path)
        entries = {entry["name"]: entry for entry in manifest["entries"]}
        self.assertEqual((entries["graph.a"]["shape"],
                          entries["graph.scan"]["shape"]), ([2, 3], [2, 4]))
        self.assertEqual((entries["graph.a"]["file_offset"],
                          entries["graph.scan"]["file_offset"]), (64, 128))
        self.assertEqual(manifest["template"]["native_entry"]["function"],
                         "ck_kokoro_waveform_bounded")
        for mutation in ("wrong_artifact", "wrong_shape", "wrong_direction"):
            with self.subTest(mutation=mutation):
                invalid = copy.deepcopy(catalog)
                altered = copy.deepcopy(source)
                hashes = ["manifest", "bump"]
                if mutation == "wrong_artifact":
                    hashes[1] = "different"
                elif mutation == "wrong_shape":
                    altered["entries"][0]["size"] = 20
                else:
                    altered["entries"][2]["shape"] = [3]
                self.catalog_path.write_text(json.dumps(invalid))
                with mock.patch.object(runtime_build, "sha256_file",
                                       side_effect=hashes), \
                        mock.patch.object(runtime_build, "verify_bundle",
                                          return_value=altered):
                    with self.assertRaises(ValueError):
                        runtime_build.compiler_manifest(self.bump_dir, 8,
                                                        self.catalog_path)

    def test_build_refuses_nonempty_output_before_touching_artifacts(self):
        output = self.root / "existing-build"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("existing artifact")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            runtime_build.build_runtime(self.bump_dir, output, 8,
                                        self.catalog_path)
        self.assertEqual(sentinel.read_text(), "existing artifact")

    @unittest.skipUnless(os.environ.get("CKE_TTS_RUN_REQUEST_PACKAGE_FULL"),
                         "asset-backed generated Kokoro request replay not requested")
    def test_pinned_bump_package_runs_complete_generated_waveform(self):
        bump_dir = Path(os.environ["CKE_TTS_NATIVE_BUMP_DIR"])
        voice_bank = Path(os.environ["CKE_TTS_NATIVE_VOICE_BANK"])
        wav_hashes = {}
        for count, fixture_path, expected_samples, expected_hash in (
            (8, ROOT / "tests/fixtures/tts/kokoro_waveform_hello_8_pinned.npz",
             32400, "8f88a076382944cfd26826d964cf12f378ebc8c16f7ce98ea5ed24e98667eda1"),
            (36, ROOT / "tests/fixtures/tts/kokoro_encoder_pinned.npz",
             61800, "187e777636f7071776968bb7a7c4ced52f3d0438bedc281b84416562c844015e"),
        ):
            with self.subTest(phoneme_count=count):
                graph_root = self.root / f"native-{count}"
                report = runtime_build.build_runtime(bump_dir, graph_root, count)
                self.assertEqual((report["phoneme_count"],
                                  report["generated_operations"]), (count, 555))
                with np.load(fixture_path, allow_pickle=False) as fixture:
                    ids_csv = ",".join(str(int(value))
                                       for value in fixture["word_ids"].ravel())
                wav = graph_root / "hello.wav"
                run = subprocess.run([str(graph_root / "native-kokoro-wav"),
                    str(graph_root / "request.cktts"), str(wav), "7",
                    ids_csv, str(voice_bank)], cwd=graph_root,
                    check=True, capture_output=True, text=True)
                self.assertIn(f"wrote {expected_samples} samples at 24000 Hz",
                              run.stdout)
                wav_hashes[count] = hashlib.sha256(wav.read_bytes()).hexdigest()
                self.assertEqual(wav_hashes[count], expected_hash)
        wav_hash = wav_hashes[8]
        long_hash = wav_hashes[36]
        print("CKE_NUMERICAL_CASE " + json.dumps({
            "case_id": "kokoro.request-package.pinned-bump-wav-v1",
            "name": "Pinned BUMP packages to two complete generated Kokoro WAVs",
            "provider": "generated_kokoro_waveform+native_request_host",
            "dtype": "fp32", "direction": "inference",
            "oracle": "seeded CKTTS003 WAV hash from pinned fixture package",
            "status": "pass", "gate": "session", "blocking": True,
            "max_diff": 0, "tolerance": 0,
            "configuration": "8 and 36 IDs; af_heart rows 5 and 33; seed 7; "
                "32400 and 61800 samples",
            "short_wav_sha256": wav_hash,
            "long_wav_sha256": long_hash,
            "reproduction_command": "CKE_TTS_RUN_REQUEST_PACKAGE_FULL=1 "
                "CKE_TTS_NATIVE_BUMP_DIR=<pinned-bump-dir> "
                "CKE_TTS_NATIVE_VOICE_BANK=<pinned-ckvoice> "
                "python3 -m unittest tests.test_v8_kokoro_request_package",
        }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
