"""Native dictionary-only Kokoro text preparation and dispatch contracts."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


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
        self.cases = (("Hello.", short["phonemes"], short["input_ids"]),
                      ("Read this assistant response aloud.",
                       long["phonemes"], long["input_ids"]))
        vocab = {" ": 16, ".": 4, ",": 3, "!": 5, "?": 6}
        for _, phones, ids in self.cases:
            self.assertEqual(len(phones) + 2, len(ids))
            for phone, token in zip(phones, ids[1:-1]):
                self.assertEqual(vocab.setdefault(phone, token), token)
        config = self.root / "config.json"
        config.write_text(json.dumps({"vocab": vocab}))
        phones = {"hello": self.cases[0][1][:-1]}
        for word, pronunciation in zip(
                ("read", "this", "assistant", "response", "aloud"),
                self.cases[1][1][:-1].split(" ")):
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
        self.assertEqual(self.report["word_count"], 6)
        print("CKE_NUMERICAL_CASE " + json.dumps({
            "case_id": "kokoro.native-lexicon-text.two-pinned-phrases-v1",
            "name": "Native dictionary frontend IDs for pinned Kokoro phrases",
            "provider": "native_kokoro_lexicon_text", "dtype": "i32",
            "direction": "inference", "oracle": "pinned Misaki/Kokoro capture",
            "status": "pass", "gate": "session", "blocking": True,
            "max_diff": 0, "tolerance": 0,
            "configuration": "synthetic six-word lexicon; Hello. and pinned 36-ID sentence",
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


class KokoroNativeLexiconFullTest(unittest.TestCase):
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
