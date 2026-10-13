"""Offline Kokoro request package bounds and BUMP binding contracts."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/tts"))
import export_kokoro_bump as bump_export
import export_kokoro_request_package as request_export


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

    @unittest.skipUnless(os.environ.get("CKE_TTS_RUN_REQUEST_PACKAGE_FULL"),
                         "asset-backed generated Kokoro request replay not requested")
    def test_pinned_bump_package_runs_complete_generated_waveform(self):
        from tests.test_v8_kokoro_generated_waveform import KokoroGeneratedWaveformTest
        from tests.v8_checked_graph_test_support import compile_native_graph
        import build_kokoro_waveform_circuit as author

        bump_dir = Path(os.environ["CKE_TTS_NATIVE_BUMP_DIR"])
        voice_bank = Path(os.environ["CKE_TTS_NATIVE_VOICE_BANK"])
        baseline = KokoroGeneratedWaveformTest
        baseline.setUpClass()
        self.addCleanup(baseline.tearDownClass)
        graph_root = self.root / "generated"
        graph_root.mkdir()
        circuit = graph_root / "circuit.json"
        circuit.write_text(json.dumps(author.build_circuit(8)))
        source = copy.deepcopy(baseline.waveform_source)
        source["template"] = author.build_circuit(8)
        source["config"].update(model=source["template"]["name"],
            arch=source["template"]["name"], context_length=8,
            max_seq_len=8, T=8)
        build_root = graph_root / "build"
        build_root.mkdir()
        layout, calls, library, _, _ = compile_native_graph(
            build_root, source, circuit)
        self.assertEqual(calls["errors"], [])
        layout_path = graph_root / "layout.json"
        layout_path.write_text(json.dumps(layout))
        package = graph_root / "request.cktts"
        report = request_export.package_request(layout_path, bump_dir, package)
        self.assertEqual((report["phoneme_count"], report["weights"],
                          report["segments"]), (8, 580, 583))
        for source_path, name in (
                (library, "generated.so"),
                (ROOT / "version/v8/tts/native_kokoro_wav_host.c", "host.c"),
                (ROOT / "src/kernels/audio_noise_counter.c", "audio_noise_counter.c"),
                (ROOT / "src/kernels/checked_float_buffer.h", "checked_float_buffer.h"),
                (ROOT / "include/ckernel_tts.h", "ckernel_tts.h")):
            shutil.copy2(source_path, graph_root / name)
        binary = graph_root / "native-kokoro-wav"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra",
            "-Werror", "-pedantic", "-DCK_TTS_COMPILED_TOKEN_COUNT=8",
            "-I", str(graph_root), str(graph_root / "host.c"),
            str(graph_root / "audio_noise_counter.c"), "-L", str(graph_root),
            "-l:generated.so", "-Wl,-rpath,$ORIGIN", "-lm", "-o", str(binary)],
            check=True, capture_output=True)
        with np.load(ROOT / "tests/fixtures/tts/kokoro_waveform_hello_8_pinned.npz",
                     allow_pickle=False) as fixture:
            ids_csv = ",".join(str(int(value)) for value in fixture["word_ids"])
        wav = graph_root / "hello.wav"
        run = subprocess.run([str(binary), str(package), str(wav), "7",
            ids_csv, str(voice_bank)], cwd=graph_root, check=True,
            capture_output=True, text=True)
        self.assertIn("wrote 32400 samples at 24000 Hz", run.stdout)
        wav_hash = hashlib.sha256(wav.read_bytes()).hexdigest()
        self.assertEqual(wav_hash,
            "8f88a076382944cfd26826d964cf12f378ebc8c16f7ce98ea5ed24e98667eda1")
        long_root = self.root / "generated-36"
        long_root.mkdir()
        long_layout = long_root / "layout.json"
        long_layout.write_text(json.dumps(baseline.layout))
        long_package = long_root / "request.cktts"
        long_report = request_export.package_request(long_layout, bump_dir,
                                                     long_package)
        self.assertEqual((long_report["phoneme_count"], long_report["weights"],
                          long_report["segments"]), (36, 580, 583))
        for source_path, name in (
                (baseline.native_dir / "generated.so", "generated.so"),
                (ROOT / "version/v8/tts/native_kokoro_wav_host.c", "host.c"),
                (ROOT / "src/kernels/audio_noise_counter.c", "audio_noise_counter.c"),
                (ROOT / "src/kernels/checked_float_buffer.h", "checked_float_buffer.h"),
                (ROOT / "include/ckernel_tts.h", "ckernel_tts.h")):
            shutil.copy2(source_path, long_root / name)
        long_binary = long_root / "native-kokoro-wav"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra",
            "-Werror", "-pedantic", "-DCK_TTS_COMPILED_TOKEN_COUNT=36",
            "-I", str(long_root), str(long_root / "host.c"),
            str(long_root / "audio_noise_counter.c"), "-L", str(long_root),
            "-l:generated.so", "-Wl,-rpath,$ORIGIN", "-lm", "-o",
            str(long_binary)], check=True, capture_output=True)
        long_ids = ",".join(str(int(value)) for value in
                            baseline.encoder["word_ids"].ravel())
        long_wav = long_root / "hello.wav"
        long_run = subprocess.run([str(long_binary), str(long_package),
            str(long_wav), "7", long_ids, str(voice_bank)], cwd=long_root,
            check=True, capture_output=True, text=True)
        self.assertIn("wrote 61800 samples at 24000 Hz", long_run.stdout)
        long_hash = hashlib.sha256(long_wav.read_bytes()).hexdigest()
        self.assertEqual(long_hash,
            "187e777636f7071776968bb7a7c4ced52f3d0438bedc281b84416562c844015e")
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
