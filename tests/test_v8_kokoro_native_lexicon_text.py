"""Native dictionary-only Kokoro text preparation and dispatch contracts."""

import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
import unittest
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/tts"))
import build_kokoro_lexicon_frontend as frontend


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class KokoroNativeLexiconTextTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        short = json.loads((ROOT / "tests/fixtures/tts" /
                            "kokoro_waveform_hello_8_pinned.json").read_text())
        long = json.loads((ROOT / "version/v8/tts/reference" /
                           "fixture_manifest.json").read_text())
        medium_path = ROOT / "tests/fixtures/tts" / \
            "kokoro_runtime_waveform_hello_there_13_pinned.npz"
        medium_meta = json.loads(medium_path.with_suffix(".json").read_text())
        self.assertEqual(_hash(medium_path), medium_meta["fixture_sha256"])
        self.assertEqual(medium_meta["model_pin"], short["model_pin"]["revision"])
        self.assertEqual(medium_meta["code_pin"], short["code_pin"]["kokoro"])
        self.assertEqual(medium_meta["asset_sha256"], short["asset_sha256"])
        self.assertEqual(medium_meta["asset_sha256"], long["assets"])
        with np.load(medium_path) as capture:
            medium_ids = capture["word_ids"].tolist()
        self.cases = (("Hello.", short["phonemes"], short["input_ids"]),
                      ("Hello there.", None, medium_ids),
                      ("Read this assistant response aloud.",
                       long["phonemes"], long["input_ids"]))
        vocab = {" ": 16, ".": 4, ",": 3, "!": 5, "?": 6}
        for _, phones, ids in self.cases:
            if phones is None:
                continue
            self.assertEqual(len(phones) + 2, len(ids))
            for phone, token in zip(phones, ids[1:-1]):
                self.assertEqual(vocab.setdefault(phone, token), token)
        for token in medium_ids[7:-2]:
            if token not in vocab.values():
                vocab[chr(0xE000 + token)] = token
        config = self.root / "config.json"
        config.write_text(json.dumps({"vocab": vocab}))
        phones = {"hello": self.cases[0][1][:-1]}
        by_id = {token: phone for phone, token in vocab.items()}
        phones["there"] = "".join(by_id[token] for token in medium_ids[7:-2])
        for word, pronunciation in zip(
                ("read", "this", "assistant", "response", "aloud"),
            self.cases[2][1][:-1].split(" ")):
            phones[word] = pronunciation
        source = self.root / "us_gold.json"
        source.write_text(json.dumps(phones))
        self.export = self.root / "frontend"
        self.report = frontend.build_frontend(config, source, self.export,
            config_sha256=_hash(config), lexicon_sha256=_hash(source))
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        dispatch = self.bundle / "native-kokoro-dispatch"
        dispatch.write_text("#!/bin/sh\nprintf '%s\\n' \"$4\" > \"$2\"\n")
        dispatch.chmod(0o755)

    def invoke(self, text, name):
        output = self.root / name
        run = subprocess.run([str(self.export / "native-kokoro-text"),
            str(self.bundle), str(self.export / "us-gold.cklex"),
            str(output), "7", text], capture_output=True, text=True)
        return run, output

    def test_pinned_text_fixtures_produce_reference_ids(self):
        for index, (text, _, expected) in enumerate(self.cases):
            with self.subTest(text=text):
                run, output = self.invoke(text, f"case-{index}.wav")
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual([int(value) for value in
                                  output.read_text().strip().split(",")], expected)
        self.assertEqual(self.report["word_count"], 7)
        print("CKE_NUMERICAL_CASE " + json.dumps({
            "case_id": "kokoro.native-lexicon-text.three-pinned-phrases-v1",
            "name": "Native dictionary frontend IDs for three pinned Kokoro phrases",
            "provider": "native_kokoro_lexicon_text", "dtype": "i32",
            "direction": "inference", "oracle": "pinned Misaki/Kokoro capture",
            "status": "pass", "gate": "session", "blocking": True,
            "max_diff": 0, "tolerance": 0,
            "configuration": "synthetic seven-word lexicon; 8/13/36-ID phrases",
            "reproduction_command": "python3 -m unittest "
                "tests.test_v8_kokoro_native_lexicon_text",
        }, sort_keys=True))

    def test_unsupported_text_and_tampered_lexicon_reject(self):
        for index, text in enumerate(("", "A missing word.", "Hello 2.",
                                      "Hello;", "Hello..", "hello " * 8,
                                      "Hello" * 120)):
            with self.subTest(text=text[:30]):
                run, output = self.invoke(text, f"bad-{index}.wav")
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertFalse(output.exists())
        lexicon = self.export / "us-gold.cklex"
        lexicon.write_bytes(lexicon.read_bytes() + b"tamper")
        run, output = self.invoke("Hello.", "tampered.wav")
        self.assertEqual(run.returncode, 3, run.stderr)
        self.assertFalse(output.exists())
        protected = self.root / "protected.wav"
        protected.write_bytes(b"existing output")
        run, output = self.invoke("Hello.", protected.name)
        self.assertEqual(run.returncode, 3, run.stderr)
        self.assertEqual(output.read_bytes(), b"existing output")

    def test_export_rejects_wrong_source_before_publication(self):
        target = self.root / "rejected-export"
        with self.assertRaisesRegex(ValueError, "pinned hashes"):
            frontend.build_frontend(self.root / "config.json",
                self.root / "us_gold.json", target,
                config_sha256="0" * 64,
                lexicon_sha256=_hash(self.root / "us_gold.json"))
        self.assertFalse(target.exists())

    def test_runtime_asset_coverage_is_explicit(self):
        if not os.environ.get("CKE_TTS_RUN_RUNTIME_TEXT_FULL"):
            for case_id, name in (
                ("kokoro.native-runtime-text.three-length-wavs-v1",
                 "Native dictionary text through one variable-length generated graph"),
                ("kokoro.native-runtime-text.session-recovery-v1",
                 "Generated Kokoro text session stop error and recovery"),
            ):
                print("CKE_NUMERICAL_CASE " + json.dumps({
                    "case_id": case_id, "name": name,
                    "provider": "generated_kokoro_runtime_waveform",
                    "dtype": "pcm16", "direction": "inference",
                    "oracle": "asset-backed pinned BUMP and voice bank",
                    "status": "not_tested", "gate": "execution", "blocking": False,
                    "reason": "pinned runtime bundle not supplied; set "
                              "CKE_TTS_RUN_RUNTIME_TEXT_FULL and "
                              "CKE_TTS_RUNTIME_TEXT_BUNDLE_DIR",
                    "configuration": "8/13/36-ID native text and session replay",
                    "reproduction_command": "CKE_TTS_RUN_RUNTIME_TEXT_FULL=1 "
                        "CKE_TTS_RUNTIME_TEXT_BUNDLE_DIR=<pinned-bundle> "
                        "python3 -m unittest tests.test_v8_kokoro_native_lexicon_text",
                }, sort_keys=True))
        if not os.environ.get("CKE_TTS_RUN_RUNTIME_PINNED_TEXT_FULL"):
            print("CKE_NUMERICAL_CASE " + json.dumps({
                "case_id": "kokoro.native-runtime-text.pinned-lexicon-v1",
                "name": "Pinned full-lexicon routing through runtime Kokoro bundle",
                "provider": "native_kokoro_lexicon_text+generated_kokoro_runtime_waveform",
                "dtype": "pcm16", "direction": "inference",
                "oracle": "pinned frontend artifact and seeded CKE WAV baselines",
                "status": "not_tested", "gate": "execution", "blocking": False,
                "reason": "pinned full lexicon and runtime bundle not supplied; set "
                          "CKE_TTS_RUN_RUNTIME_PINNED_TEXT_FULL, "
                          "CKE_TTS_PINNED_FRONTEND_DIR and "
                          "CKE_TTS_RUNTIME_TEXT_BUNDLE_DIR",
                "configuration": "8/12/13/36-ID pinned dictionary behavior",
                "reproduction_command": "CKE_TTS_RUN_RUNTIME_PINNED_TEXT_FULL=1 "
                    "CKE_TTS_PINNED_FRONTEND_DIR=<pinned-frontend> "
                    "CKE_TTS_RUNTIME_TEXT_BUNDLE_DIR=<pinned-bundle> "
                    "python3 -m unittest tests.test_v8_kokoro_native_lexicon_text."
                    "KokoroNativeLexiconFullTest."
                    "test_pinned_dictionary_runtime_length_routes",
            }, sort_keys=True))

    @unittest.skipUnless(os.environ.get("CKE_TTS_RUN_RUNTIME_TEXT_FULL"),
                         "pinned variable-length generated bundle not requested")
    def test_one_runtime_bundle_speaks_three_text_lengths(self):
        bundle = Path(os.environ["CKE_TTS_RUNTIME_TEXT_BUNDLE_DIR"])
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        self.assertTrue(manifest["runtime_lengths"])
        self.assertEqual(manifest["supported_phoneme_counts"], [8, 13, 36])
        self.assertEqual(len({item["library_sha256"] for item in
                              manifest["buckets"].values()}), 1)
        expected = (
            (0, 32400, "8f88a076382944cfd26826d964cf12f378ebc8c16f7ce98ea5ed24e98667eda1"),
            (1, 36600, "6a60f4d72ced2e3dac9e6200835592d5c330adcdd161bbf015f410f9fee8c33a"),
            (2, 61800, "187e777636f7071776968bb7a7c4ced52f3d0438bedc281b84416562c844015e"),
            (0, 32400, "8f88a076382944cfd26826d964cf12f378ebc8c16f7ce98ea5ed24e98667eda1"),
        )
        for request, (index, samples, digest) in enumerate(expected):
            text = self.cases[index][0]
            wav = self.root / f"runtime-text-{request}.wav"
            run = subprocess.run([str(self.export / "native-kokoro-text"),
                str(bundle), str(self.export / "us-gold.cklex"),
                str(wav), "7", text], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("artifact SHA-256 verified", run.stderr)
            with wave.open(str(wav), "rb") as audio:
                self.assertEqual((audio.getnchannels(), audio.getsampwidth(),
                    audio.getframerate(), audio.getnframes()),
                    (1, 2, 24000, samples))
            self.assertEqual(_hash(wav), digest)
        print("CKE_NUMERICAL_CASE " + json.dumps({
            "case_id": "kokoro.native-runtime-text.three-length-wavs-v1",
            "name": "Native dictionary text through one variable-length generated graph",
            "provider": "generated_kokoro_runtime_waveform", "dtype": "pcm16",
            "direction": "inference", "oracle": "retained CKE seeded WAV baselines",
            "status": "pass", "gate": "execution", "blocking": True,
            "max_diff": 0, "tolerance": 0,
            "configuration": "8/13/36/8 IDs; af_heart; native seed 7; one library artifact",
            "library_sha256": manifest["buckets"]["8"]["library_sha256"],
            "reproduction_command": "CKE_TTS_RUN_RUNTIME_TEXT_FULL=1 "
                "CKE_TTS_RUNTIME_TEXT_BUNDLE_DIR=<pinned-bundle> "
                "python3 -m unittest tests.test_v8_kokoro_native_lexicon_text."
                "KokoroNativeLexiconTextTest."
                "test_one_runtime_bundle_speaks_three_text_lengths",
        }, sort_keys=True))

    @unittest.skipUnless(os.environ.get("CKE_TTS_RUN_RUNTIME_TEXT_FULL"),
                         "pinned variable-length generated bundle not requested")
    def test_runtime_text_session_stop_error_and_recovery(self):
        bundle = Path(os.environ["CKE_TTS_RUNTIME_TEXT_BUNDLE_DIR"])
        process = subprocess.Popen([
            str(self.export / "native-kokoro-session"),
            str(self.export / "native-kokoro-text"), str(bundle),
            str(self.export / "us-gold.cklex"), str(self.root), "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        pending = b""

        def line(timeout=90):
            nonlocal pending
            deadline = time.monotonic() + timeout
            while b"\n" not in pending:
                remaining = deadline - time.monotonic()
                self.assertGreater(remaining, 0, "native session event timed out")
                readable, _, _ = select.select([process.stdout], [], [], remaining)
                self.assertTrue(readable, "native session event timed out")
                chunk = os.read(process.stdout.fileno(), 4096)
                self.assertTrue(chunk, "native session closed unexpectedly")
                pending += chunk
            result, pending = pending.split(b"\n", 1)
            return result.decode()

        def send(command):
            process.stdin.write(command.encode() + b"\n")
            process.stdin.flush()

        try:
            ready = line()
            self.assertTrue(ready.startswith("READY\t"), ready)
            directory = Path(ready.split("\t", 1)[1])
            send("SPEAK\t7\tHello.")
            self.assertEqual(line(), "EVENT\t1\tSTARTED")
            self.assertEqual(line(), f"EVENT\t1\tWAV_READY\t{directory / '1.wav'}")
            self.assertEqual(line(), "EVENT\t1\tCOMPLETED")
            short_hash = _hash(directory / "1.wav")
            send("SPEAK\t7\tRead this assistant response aloud.")
            self.assertEqual(line(), "EVENT\t2\tSTARTED")
            send("STOP")
            self.assertEqual(line(), "EVENT\t2\tCANCELLED")
            self.assertFalse((directory / "2.wav").exists())
            send("SPEAK\t7\tUnsupported word.")
            self.assertEqual(line(), "EVENT\t3\tSTARTED")
            self.assertEqual(line(), "EVENT\t3\tERROR\tnative child failed")
            send("SPEAK\t7\tHello there.")
            self.assertEqual(line(), "EVENT\t4\tSTARTED")
            self.assertEqual(line(), f"EVENT\t4\tWAV_READY\t{directory / '4.wav'}")
            self.assertEqual(line(), "EVENT\t4\tCOMPLETED")
            with wave.open(str(directory / "4.wav"), "rb") as audio:
                self.assertEqual(audio.getnframes(), 36600)
            send("SPEAK\t7\tHello.")
            self.assertEqual(line(), "EVENT\t5\tSTARTED")
            self.assertEqual(line(), f"EVENT\t5\tWAV_READY\t{directory / '5.wav'}")
            self.assertEqual(line(), "EVENT\t5\tCOMPLETED")
            self.assertEqual(_hash(directory / "5.wav"), short_hash)
            send("QUIT")
            self.assertEqual(process.wait(timeout=5), 0)
            print("CKE_NUMERICAL_CASE " + json.dumps({
                "case_id": "kokoro.native-runtime-text.session-recovery-v1",
                "name": "Generated Kokoro text session stop error and recovery",
                "provider": "native_kokoro_speech_session+generated_kokoro_runtime_waveform",
                "dtype": "pcm16", "direction": "inference",
                "oracle": "session event order and seeded short-WAV replay",
                "status": "pass", "gate": "session", "blocking": True,
                "max_diff": 0, "tolerance": 0,
                "configuration": "8-ID success; 36-ID cancellation; unsupported word; "
                                 "13-ID success; 8-ID exact recovery; no player",
                "reproduction_command": "CKE_TTS_RUN_RUNTIME_TEXT_FULL=1 "
                    "CKE_TTS_RUNTIME_TEXT_BUNDLE_DIR=<pinned-bundle> "
                    "python3 -m unittest tests.test_v8_kokoro_native_lexicon_text."
                    "KokoroNativeLexiconTextTest."
                    "test_runtime_text_session_stop_error_and_recovery",
            }, sort_keys=True))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()


class KokoroNativeLexiconFullTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("CKE_TTS_RUN_RUNTIME_PINNED_TEXT_FULL"),
                         "pinned full lexicon and runtime bundle not requested")
    def test_pinned_dictionary_runtime_length_routes(self):
        frontend_dir = Path(os.environ["CKE_TTS_PINNED_FRONTEND_DIR"])
        bundle = Path(os.environ["CKE_TTS_RUNTIME_TEXT_BUNDLE_DIR"])
        identity = json.loads((frontend_dir / "frontend_manifest.json").read_text())
        self.assertEqual(identity["schema"], "cke.kokoro.native_lexicon_text.v1")
        self.assertEqual(identity["source_misaki_commit"], frontend.MISAKI_COMMIT)
        self.assertEqual(identity["source_kokoro_asset_commit"],
                         frontend.KOKORO_ASSET_COMMIT)
        self.assertEqual(identity["exported_lexicon_sha256"],
                         _hash(frontend_dir / "us-gold.cklex"))
        self.assertEqual(identity["frontend_binary_sha256"],
                         _hash(frontend_dir / "native-kokoro-text"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, phrase, samples, digest in (
                ("hello", "Hello.", 32400,
                 "8f88a076382944cfd26826d964cf12f378ebc8c16f7ce98ea5ed24e98667eda1"),
                ("hello-comma-there", "Hello, there.", 36600,
                 "3ede801cb0063d67ebea3bbf6a2ed16309f3debe3a7fd2edff445dd86029ca72"),
                ("read-reply", "Read this assistant response aloud.", 61800,
                 "187e777636f7071776968bb7a7c4ced52f3d0438bedc281b84416562c844015e"),
            ):
                output = root / f"{label}.wav"
                run = subprocess.run([str(frontend_dir / "native-kokoro-text"),
                    str(bundle), str(frontend_dir / "us-gold.cklex"),
                    str(output), "7", phrase], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                with wave.open(str(output), "rb") as audio:
                    self.assertEqual((audio.getframerate(), audio.getnframes()),
                                     (24000, samples))
                self.assertEqual(_hash(output), digest)
            rejected = root / "unsupported-twelve.wav"
            run = subprocess.run([str(frontend_dir / "native-kokoro-text"),
                str(bundle), str(frontend_dir / "us-gold.cklex"),
                str(rejected), "7", "Hello there."],
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn("unsupported phoneme count 12", run.stderr)
            self.assertFalse(rejected.exists())
        print("CKE_NUMERICAL_CASE " + json.dumps({
            "case_id": "kokoro.native-runtime-text.pinned-lexicon-v1",
            "name": "Pinned full-lexicon routing through runtime Kokoro bundle",
            "provider": "native_kokoro_lexicon_text+generated_kokoro_runtime_waveform",
            "dtype": "pcm16", "direction": "inference",
            "oracle": "pinned frontend artifact and seeded CKE WAV baselines",
            "status": "pass", "gate": "execution", "blocking": True,
            "max_diff": 0, "tolerance": 0,
            "configuration": "Hello. 8 IDs; Hello there. 12 IDs rejected; "
                             "Hello, there. 13 IDs; pinned reply 36 IDs",
            "reproduction_command": "CKE_TTS_RUN_RUNTIME_PINNED_TEXT_FULL=1 "
                "CKE_TTS_PINNED_FRONTEND_DIR=<pinned-frontend> "
                "CKE_TTS_RUNTIME_TEXT_BUNDLE_DIR=<pinned-bundle> "
                "python3 -m unittest tests.test_v8_kokoro_native_lexicon_text."
                "KokoroNativeLexiconFullTest."
                "test_pinned_dictionary_runtime_length_routes",
        }, sort_keys=True))

    @unittest.skipUnless(os.environ.get("CKE_TTS_RUN_TEXT_FULL"),
                         "pinned lexicon, config and generated bundle not requested")
    def test_real_native_text_to_two_generated_wavs(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("CKE_TTS_BUILD_TMP_DIR")) as tmp:
            root = Path(tmp)
            export = root / "frontend"
            frontend.build_frontend(
                Path(os.environ["CKE_TTS_KOKORO_CONFIG"]),
                Path(os.environ["CKE_TTS_MISAKI_US_GOLD"]), export)
            bundle = Path(os.environ["CKE_TTS_TEXT_BUNDLE_DIR"])
            for name, text, expected in (
                ("hello", "Hello.",
                 "8f88a076382944cfd26826d964cf12f378ebc8c16f7ce98ea5ed24e98667eda1"),
                ("read", "Read this assistant response aloud.",
                 "187e777636f7071776968bb7a7c4ced52f3d0438bedc281b84416562c844015e"),
            ):
                wav = root / f"{name}.wav"
                run = subprocess.run([str(export / "native-kokoro-text"),
                    str(bundle), str(export / "us-gold.cklex"), str(wav),
                    "7", text], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(_hash(wav), expected)
            print("CKE_NUMERICAL_CASE " + json.dumps({
                "case_id": "kokoro.native-lexicon-text.two-generated-wavs-v1",
                "name": "Native dictionary text to two generated Kokoro WAVs",
                "provider": "native_kokoro_lexicon_text", "dtype": "pcm16",
                "direction": "inference", "oracle": "retained seeded Kokoro WAV hashes",
                "status": "pass", "gate": "session", "blocking": True,
                "configuration": "pinned full Misaki US gold, Kokoro config, 8/36-ID bundle, seed 7",
                "output_hashes": [
                    "8f88a076382944cfd26826d964cf12f378ebc8c16f7ce98ea5ed24e98667eda1",
                    "187e777636f7071776968bb7a7c4ced52f3d0438bedc281b84416562c844015e"],
                "reproduction_command": "CKE_TTS_RUN_TEXT_FULL=1 python3 -m unittest "
                    "tests.test_v8_kokoro_native_lexicon_text",
            }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
