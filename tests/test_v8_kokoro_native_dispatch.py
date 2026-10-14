"""Cheap native exact-length selector and artifact-identity controls."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "version/v8/tts/native_kokoro_dispatch.c"
sys.path.insert(0, str(ROOT / "version/v8/tts"))
import build_kokoro_dispatch_bundle as bundle_builder


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class KokoroNativeDispatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        voice = self.root / "af_heart.ckvoice"
        voice.write_bytes(b"synthetic voice bank")
        self.voice_hash = _hash(voice)
        records = []
        for count in (8, 36):
            directory = self.root / str(count)
            directory.mkdir()
            host = directory / "native-kokoro-wav"
            host.write_text(
                "#!/bin/sh\n"
                f"printf 'bucket={count}:%s:%s\\n' \"$3\" \"$4\" > \"$2\"\n"
            )
            host.chmod(0o755)
            library = directory / "generated.so"
            library.write_bytes(f"synthetic library {count}".encode())
            package = directory / "request.cktts"
            package.write_bytes(f"synthetic package {count}".encode())
            records.append((count, _hash(host), _hash(library), _hash(package)))
        self.assertNotEqual(records[0][1], records[1][1])
        header = [
            "#ifndef CKE_KOKORO_BUNDLE_IDENTITY_H",
            "#define CKE_KOKORO_BUNDLE_IDENTITY_H",
            "typedef struct { unsigned count; const char *host_sha256; "
            "const char *library_sha256; const char *package_sha256; } ck_tts_bucket;",
            f'static const char ck_tts_voice_sha256[] = "{self.voice_hash}";',
            "static const ck_tts_bucket ck_tts_buckets[] = {",
        ]
        header.extend(f'{{{count}, "{host}", "{library}", "{package}"}},'
                      for count, host, library, package in records)
        header.extend(["};", "#endif"])
        (self.root / "bundle_identity.h").write_text("\n".join(header))
        self.binary = self.root / "dispatch"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra",
            "-Werror", "-pedantic", "-I", str(self.root), str(SOURCE),
            "-lcrypto", "-o", str(self.binary)], check=True,
            capture_output=True, text=True)

    def invoke(self, ids, name="out.wav"):
        output = self.root / name
        result = subprocess.run([str(self.binary), str(self.root), str(output),
            "7", ids], capture_output=True, text=True)
        return result, output

    def test_bundle_builder_rejects_bad_voice_and_occupied_output(self):
        output = self.root / "new-bundle"
        with self.assertRaisesRegex(ValueError, "voice bank differs"):
            bundle_builder.build_bundle(self.root / "missing-bump",
                                        self.root / "af_heart.ckvoice", output)
        self.assertFalse(output.exists())
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            bundle_builder.build_bundle(self.root / "missing-bump",
                                        self.root / "af_heart.ckvoice", output)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_failed_build_does_not_publish_partial_bundle(self):
        voice = self.root / "valid.ckvoice"
        voice.write_bytes(b"CKVOICE1" + (34).to_bytes(4, "little") +
                          (256).to_bytes(4, "little") +
                          b"\0" * (34 * 256 * 4))
        bindings = self.root / "bindings.json"
        bindings.write_text(json.dumps({
            "native_voice_sha256": _hash(voice),
            "source_bump_sha256": "synthetic",
        }))
        output = self.root / "failed-build"
        output.mkdir()

        def fail_after_first_bucket(_bump, directory, count):
            directory.mkdir()
            (directory / "partial").write_text("not deployable")
            if count == 36:
                raise RuntimeError("second bucket build failed")
            return {key: "synthetic" for key in (
                "library_sha256", "host_sha256", "package_sha256",
                "compiler_manifest_sha256", "generated_c_sha256",
                "generated_operations")}

        with mock.patch.object(bundle_builder, "DEFAULT_BINDINGS", bindings), \
             mock.patch.object(bundle_builder, "build_runtime",
                               side_effect=fail_after_first_bucket):
            with self.assertRaisesRegex(RuntimeError,
                                        "second bucket build failed"):
                bundle_builder.build_bundle(self.root / "synthetic-bump",
                                            voice, output)
        self.assertEqual(list(output.iterdir()), [])
        self.assertEqual(list(self.root.glob(".failed-build.staging-*")), [])

    def test_selects_exact_lengths_and_rejects_others(self):
        for count in (8, 36):
            ids = ",".join(map(str, [0] + [4] * (count - 2) + [0]))
            result, output = self.invoke(ids, f"{count}.wav")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text().strip(), f"bucket={count}:7:{ids}")
        for count in (7, 9, 35, 37, 40):
            ids = ",".join(map(str, [0] + [4] * (count - 2) + [0]))
            result, output = self.invoke(ids, f"unsupported-{count}.wav")
            self.assertEqual(result.returncode, 2)
            self.assertIn("unsupported phoneme count", result.stderr)
            self.assertFalse(output.exists())
        for ids in ("", "0,4,,0", "0,178,0", "-1,4,0", "0," * 41):
            result, output = self.invoke(ids, "invalid.wav")
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())

    def test_wrong_host_route_is_rejected_before_execution(self):
        original_call = 'joined(host, sizeof(host), argv[1], count, "native-kokoro-wav")'
        wrong_call = 'joined(host, sizeof(host), argv[1], 8u, "native-kokoro-wav")'
        source = SOURCE.read_text()
        self.assertEqual(source.count(original_call), 1)
        mutant = self.root / "wrong-route.c"
        mutant.write_text(source.replace(original_call, wrong_call, 1))
        wrong_dispatch = self.root / "wrong-route-dispatch"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra",
            "-Werror", "-pedantic", "-I", str(self.root), str(mutant),
            "-lcrypto", "-o", str(wrong_dispatch)], check=True,
            capture_output=True, text=True)
        ids = ",".join(map(str, [0] + [4] * 34 + [0]))
        output = self.root / "wrong-route.wav"
        result = subprocess.run([str(wrong_dispatch), str(self.root), str(output),
            "7", ids], capture_output=True, text=True)
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertIn("artifact identity mismatch", result.stderr)
        self.assertFalse(output.exists())

    def test_existing_output_and_asset_aliases_are_rejected(self):
        ids = ",".join(map(str, [0] + [4] * 6 + [0]))
        package = self.root / "8/request.cktts"
        original = package.read_bytes()
        for label, target in (
                ("package", package),
                ("voice", self.root / "af_heart.ckvoice"),
                ("host", self.root / "8/native-kokoro-wav")):
            with self.subTest(target=label):
                result, _ = self.invoke(ids, str(target))
                self.assertEqual(result.returncode, 6, result.stderr)
                self.assertIn("output path already exists", result.stderr)
        for kind in ("symlink", "hardlink"):
            alias = self.root / f"{kind}-to-package.wav"
            if kind == "symlink":
                alias.symlink_to(package)
            else:
                os.link(package, alias)
            with self.subTest(alias=kind):
                result, _ = self.invoke(ids, str(alias))
                self.assertEqual(result.returncode, 6, result.stderr)
                self.assertEqual(package.read_bytes(), original)
                self.assertEqual(alias.read_bytes(), original)
        dangling = self.root / "dangling.wav"
        dangling.symlink_to(self.root / "missing.wav")
        result, _ = self.invoke(ids, str(dangling))
        self.assertEqual(result.returncode, 6, result.stderr)
        self.assertTrue(dangling.is_symlink())
        result, output = self.invoke(ids, "fresh-after-alias.wav")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.read_text().strip(), f"bucket=8:7:{ids}")

    def test_tampered_package_library_host_or_voice_fails_before_execution(self):
        ids = ",".join(map(str, [0] + [4] * 6 + [0]))
        for target in (self.root / "8/request.cktts",
                       self.root / "8/generated.so",
                       self.root / "8/native-kokoro-wav",
                       self.root / "af_heart.ckvoice"):
            with self.subTest(target=target.name):
                original = target.read_bytes()
                target.write_bytes(original + b"x")
                result, output = self.invoke(ids, f"tampered-{target.name}.wav")
                self.assertEqual(result.returncode, 4)
                self.assertIn("artifact identity mismatch", result.stderr)
                self.assertFalse(output.exists())
                target.write_bytes(original)
        print("CKE_NUMERICAL_CASE " + json.dumps({
            "case_id": "kokoro.native-dispatch.exact-length-identity-v1",
            "name": "Kokoro native 8/36 selector and artifact identity",
            "provider": "native_kokoro_dispatch", "dtype": "fp32",
            "direction": "inference", "oracle": "synthetic distinguishable hosts",
            "status": "pass", "gate": "session", "blocking": True,
            "max_diff": 0, "tolerance": 0,
            "configuration": "8/36 accepted; unsupported counts and four tampered artifacts rejected",
            "reproduction_command": "python3 -m unittest tests.test_v8_kokoro_native_dispatch",
        }, sort_keys=True))


class KokoroNativeDispatchFullTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("CKE_TTS_RUN_DISPATCH_FULL"),
                         "asset-backed dispatch bundle replay not requested")
    def test_pinned_bundle_routes_both_native_waveforms(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("CKE_TTS_BUILD_TMP_DIR")) as tmp:
            bundle = Path(tmp) / "bundle"
            report = bundle_builder.build_bundle(
                Path(os.environ["CKE_TTS_NATIVE_BUMP_DIR"]),
                Path(os.environ["CKE_TTS_NATIVE_VOICE_BANK"]), bundle)
            self.assertEqual(report["supported_phoneme_counts"], [8, 36])
            wav_hashes = {}
            for count, fixture, expected in (
                (8, "kokoro_waveform_hello_8_pinned.npz",
                 "8f88a076382944cfd26826d964cf12f378ebc8c16f7ce98ea5ed24e98667eda1"),
                (36, "kokoro_encoder_pinned.npz",
                 "187e777636f7071776968bb7a7c4ced52f3d0438bedc281b84416562c844015e"),
            ):
                with np.load(ROOT / "tests/fixtures/tts" / fixture,
                             allow_pickle=False) as values:
                    ids = ",".join(str(int(value)) for value in
                                   values["word_ids"].ravel())
                wav = bundle / f"{count}.wav"
                run = subprocess.run([str(bundle / "native-kokoro-dispatch"),
                    str(bundle), str(wav), "7", ids], capture_output=True,
                    text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("resolved generated symbol in", run.stderr)
                wav_hashes[count] = hashlib.sha256(wav.read_bytes()).hexdigest()
                self.assertEqual(wav_hashes[count], expected)
            unsupported = bundle / "unsupported.wav"
            run = subprocess.run([str(bundle / "native-kokoro-dispatch"),
                str(bundle), str(unsupported), "7", "0,4,4,4,4,4,0"],
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertFalse(unsupported.exists())
            print("CKE_NUMERICAL_CASE " + json.dumps({
                "case_id": "kokoro.native-dispatch.pinned-bump-two-wavs-v1",
                "name": "Pinned BUMP native exact-length dispatch to two WAVs",
                "provider": "native_kokoro_dispatch+generated_kokoro_waveform",
                "dtype": "fp32", "direction": "inference",
                "oracle": "retained seeded native WAV SHA-256 baselines",
                "status": "pass", "gate": "session", "blocking": True,
                "max_diff": 0, "tolerance": 0,
                "wav_sha256": wav_hashes,
                "configuration": "8 and 36 IDs, af_heart, seed 7; unsupported 7 IDs rejected",
                "reproduction_command": "CKE_TTS_RUN_DISPATCH_FULL=1 "
                    "CKE_TTS_NATIVE_BUMP_DIR=<pinned-bump> "
                    "CKE_TTS_NATIVE_VOICE_BANK=<pinned-voice> "
                    "python3 -m unittest tests.test_v8_kokoro_native_dispatch",
            }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
