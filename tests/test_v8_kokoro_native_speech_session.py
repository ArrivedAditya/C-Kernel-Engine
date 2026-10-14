"""Bounded native Kokoro speech stop, replacement and recovery tests."""

import json
import hashlib
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "version/v8/tts/native_kokoro_speech_session.c"


class NativeSpeechSessionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "native-kokoro-session"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra",
                        "-Werror", "-pedantic", str(SOURCE), "-o",
                        str(self.binary)], check=True, capture_output=True, text=True)
        self.frontend = self.root / "fake-native-frontend"
        self.frontend.write_text("""#!/usr/bin/env python3
import sys, time, wave
if sys.argv[5] == 'slow':
    time.sleep(10)
if sys.argv[5] == 'fail':
    sys.exit(7)
if sys.argv[5] == 'empty':
    open(sys.argv[3], 'wb').close()
    sys.exit(0)
print('child diagnostics must stay off the session event channel')
with wave.open(sys.argv[3], 'wb') as audio:
    audio.setnchannels(1)
    audio.setsampwidth(2)
    audio.setframerate(16000 if sys.argv[5] == 'wrong-rate' else 24000)
    audio.writeframes(b'\\0\\0' * 4)
if sys.argv[5] == 'bad-header':
    with open(sys.argv[3], 'r+b') as output:
        output.write(b'NOTW')
if sys.argv[5] == 'trailing-bytes':
    with open(sys.argv[3], 'ab') as output:
        output.write(b'xx')
""")
        self.frontend.chmod(0o755)
        self.player = self.root / "fake-player"
        self.player.write_text("#!/bin/sh\nsleep 10\n")
        self.player.chmod(0o755)
        self.processes = []
        self.addCleanup(self._cleanup_processes)

    def _cleanup_processes(self):
        for process in self.processes:
            if process.poll() is None:
                try:
                    process.stdin.write(b"QUIT\n")
                    process.stdin.flush()
                    process.wait(timeout=2)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait(timeout=2)
            process.stdout.close()
            process.stdin.close()
            process.stderr.close()

    def start(self, player=False, *, frontend=None, bundle=None, lexicon=None):
        args = [str(self.binary), str(frontend or self.frontend),
                str(bundle or self.root),
                str(lexicon or self.root / "lexicon.cklex"), str(self.root)]
        if player:
            args.append(str(self.player))
        process = subprocess.Popen(args, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.processes.append(process)
        process._cke_pending = b""
        ready = self.line(process)
        self.assertTrue(ready.startswith("READY\t"), ready)
        return process, Path(ready.split("\t", 1)[1])

    def line(self, process, timeout=2):
        deadline = time.monotonic() + timeout
        while b"\n" not in process._cke_pending:
            remaining = deadline - time.monotonic()
            self.assertGreater(remaining, 0, "timed out waiting for session event")
            readable, _, _ = select.select([process.stdout], [], [], remaining)
            self.assertTrue(readable, "timed out waiting for session output")
            chunk = os.read(process.stdout.fileno(), 4096)
            self.assertTrue(chunk, "session closed before expected event")
            process._cke_pending += chunk
        line, process._cke_pending = process._cke_pending.split(b"\n", 1)
        return line.decode()

    def send(self, process, command):
        process.stdin.write(command.encode() + b"\n")
        process.stdin.flush()

    def test_stop_replace_and_failure_recovery(self):
        process, directory = self.start()
        self.send(process, "SPEAK\t7\tslow")
        self.assertEqual(self.line(process), "EVENT\t1\tSTARTED")
        started = time.monotonic()
        self.send(process, "STOP")
        self.assertEqual(self.line(process), "EVENT\t1\tCANCELLED")
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertFalse((directory / "1.wav").exists())
        self.send(process, "SPEAK\t7\tslow")
        self.assertEqual(self.line(process), "EVENT\t2\tSTARTED")
        self.send(process, "SPEAK\t8\tfast")
        self.assertEqual(self.line(process), "EVENT\t2\tCANCELLED")
        self.assertEqual(self.line(process), "EVENT\t3\tSTARTED")
        self.assertEqual(self.line(process), f"EVENT\t3\tWAV_READY\t{directory / '3.wav'}")
        self.assertEqual(self.line(process), "EVENT\t3\tCOMPLETED")
        self.assertFalse((directory / "2.wav").exists())
        self.assertGreater((directory / "3.wav").stat().st_size, 44)
        self.send(process, "SPEAK\t9\tfail")
        self.assertEqual(self.line(process), "EVENT\t4\tSTARTED")
        self.assertEqual(self.line(process), "EVENT\t4\tERROR\tnative child failed")
        self.assertFalse((directory / "4.wav").exists())
        self.send(process, "SPEAK\t10\tempty")
        self.assertEqual(self.line(process), "EVENT\t5\tSTARTED")
        self.assertEqual(self.line(process), "EVENT\t5\tERROR\tnative WAV missing or invalid")
        self.assertFalse((directory / "5.wav").exists())
        self.send(process, "SPEAK\t11\tfast")
        self.assertEqual(self.line(process), "EVENT\t6\tSTARTED")
        self.assertIn("EVENT\t6\tWAV_READY", self.line(process))
        self.assertEqual(self.line(process), "EVENT\t6\tCOMPLETED")

    def test_invalid_wav_rejected_before_playback_then_recovers(self):
        process, directory = self.start(player=True)
        for request_id, kind in enumerate(("bad-header", "wrong-rate",
                                           "trailing-bytes"), 1):
            self.send(process, f"SPEAK\t7\t{kind}")
            self.assertEqual(self.line(process), f"EVENT\t{request_id}\tSTARTED")
            self.assertEqual(self.line(process),
                             f"EVENT\t{request_id}\tERROR\tnative WAV missing or invalid")
            self.assertFalse((directory / f"{request_id}.wav").exists())
        self.send(process, "SPEAK\t7\tfast")
        self.assertEqual(self.line(process), "EVENT\t4\tSTARTED")
        self.assertEqual(self.line(process),
                         f"EVENT\t4\tWAV_READY\t{directory / '4.wav'}")
        self.assertEqual(self.line(process), "EVENT\t4\tPLAYING")
        self.send(process, "STOP")
        self.assertEqual(self.line(process), "EVENT\t4\tCANCELLED")

    def test_playback_stop_and_invalid_command_recovery(self):
        process, directory = self.start(player=True)
        for bad in ("INVALID", "SPEAK\tbad\tfast", "SPEAK\t7\t"):
            self.send(process, bad)
            self.assertTrue(self.line(process).startswith("EVENT\t0\tERROR"))
        self.send(process, "SPEAK\t7\tfast")
        self.assertEqual(self.line(process), "EVENT\t1\tSTARTED")
        self.assertIn("EVENT\t1\tWAV_READY", self.line(process))
        self.assertEqual(self.line(process), "EVENT\t1\tPLAYING")
        self.send(process, "STOP")
        self.assertEqual(self.line(process), "EVENT\t1\tCANCELLED")
        self.assertFalse((directory / "1.wav").exists())
        self.send(process, "SPEAK\t8\tfast")
        self.assertEqual(self.line(process), "EVENT\t2\tSTARTED")
        self.assertIn("EVENT\t2\tWAV_READY", self.line(process))
        self.assertEqual(self.line(process), "EVENT\t2\tPLAYING")
        self.send(process, "QUIT")
        self.assertEqual(self.line(process), "EVENT\t2\tCANCELLED")
        self.assertEqual(process.wait(timeout=2), 0)
        self.assertFalse((directory / "2.wav").exists())

    def test_oversized_command_rejects_then_recovers(self):
        process, directory = self.start()
        self.send(process, "SPEAK\t7\t" + "a" * 1100)
        self.assertEqual(self.line(process), "EVENT\t0\tERROR\tinvalid or oversized command")
        process.stdin.write(b"SPEAK\t7\tfast\0hidden\n")
        process.stdin.flush()
        self.assertEqual(self.line(process), "EVENT\t0\tERROR\tinvalid or oversized command")
        self.send(process, "SPEAK\t7\tfast")
        self.assertEqual(self.line(process), "EVENT\t1\tSTARTED")
        self.assertIn("EVENT\t1\tWAV_READY", self.line(process))
        self.assertEqual(self.line(process), "EVENT\t1\tCOMPLETED")
        self.assertTrue((directory / "1.wav").exists())
        print("CKE_NUMERICAL_CASE " + json.dumps({
            "case_id": "kokoro.native-speech-session.stop-replace-v1",
            "name": "Native Kokoro speech stop, replacement and recovery",
            "provider": "native_kokoro_speech_session", "dtype": "pcm16",
            "direction": "inference", "oracle": "synthetic native child and player",
            "status": "pass", "gate": "session", "blocking": True,
            "max_diff": 0, "tolerance": 0,
            "metric": "observed event and output-state mismatches",
            "configuration": "one active request; process-isolated synthesis; no audio after cancellation",
            "reproduction_command": "python3 -m unittest "
                "tests.test_v8_kokoro_native_speech_session",
        }, sort_keys=True))

    @unittest.skipUnless(os.environ.get("CKE_TTS_RUN_SESSION_FULL"),
                         "pinned native text bundle not requested")
    def test_real_generated_cancel_then_hello_waveform(self):
        args = [str(self.binary), os.environ["CKE_TTS_TEXT_FRONTEND"],
                os.environ["CKE_TTS_TEXT_BUNDLE_DIR"],
                os.environ["CKE_TTS_TEXT_LEXICON"], str(self.root)]
        process = subprocess.Popen(args, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.processes.append(process)
        process._cke_pending = b""
        ready = self.line(process)
        self.assertTrue(ready.startswith("READY\t"), ready)
        directory = Path(ready.split("\t", 1)[1])
        self.send(process, "SPEAK\t7\tRead this assistant response aloud.")
        self.assertEqual(self.line(process), "EVENT\t1\tSTARTED")
        time.sleep(0.1)
        started = time.monotonic()
        self.send(process, "STOP")
        self.assertEqual(self.line(process), "EVENT\t1\tCANCELLED")
        cancellation_seconds = time.monotonic() - started
        self.assertLess(cancellation_seconds, 2.0)
        self.assertFalse((directory / "1.wav").exists())
        self.send(process, "SPEAK\t7\tHello.")
        self.assertEqual(self.line(process), "EVENT\t2\tSTARTED")
        self.assertEqual(self.line(process, timeout=180),
                         f"EVENT\t2\tWAV_READY\t{directory / '2.wav'}")
        self.assertEqual(self.line(process), "EVENT\t2\tCOMPLETED")
        actual = hashlib.sha256((directory / "2.wav").read_bytes()).hexdigest()
        self.assertEqual(actual,
            "8f88a076382944cfd26826d964cf12f378ebc8c16f7ce98ea5ed24e98667eda1")
        print("CKE_NUMERICAL_CASE " + json.dumps({
            "case_id": "kokoro.native-speech-session.real-cancel-recover-v1",
            "name": "Cancel real generated Kokoro speech then synthesize Hello",
            "provider": "native_kokoro_speech_session+generated_kokoro_waveform",
            "dtype": "pcm16", "direction": "inference",
            "oracle": "retained seeded 8-ID native WAV hash",
            "status": "pass", "gate": "session", "blocking": True,
            "max_diff": 0, "tolerance": 0,
            "metric": "retained WAV hash mismatches",
            "cancellation_seconds": cancellation_seconds, "wav_sha256": actual,
            "configuration": "one 36-ID request cancelled; one 8-ID request completed, seed 7",
            "reproduction_command": "CKE_TTS_RUN_SESSION_FULL=1 "
                "python3 -m unittest tests.test_v8_kokoro_native_speech_session",
        }, sort_keys=True))

    @unittest.skipUnless(os.environ.get("CKE_TTS_RUN_THIRD_LENGTH_FULL"),
                         "three-length generated bundle not requested")
    def test_real_generated_short_third_short_session(self):
        bundle = Path(os.environ["CKE_TTS_THIRD_LENGTH_BUNDLE_DIR"])
        native_frontend = Path(os.environ["CKE_TTS_TEXT_FRONTEND"])
        lexicon = Path(os.environ["CKE_TTS_TEXT_LEXICON"])
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        self.assertEqual(manifest["supported_phoneme_counts"], [8, 12, 36])
        self.assertEqual(manifest["experimental_phoneme_counts"], [12])
        probe = self.root / "phoneme-probe"
        probe.mkdir()
        dispatch = probe / "native-kokoro-dispatch"
        dispatch.write_text("#!/bin/sh\nprintf '%s\\n' \"$4\" > \"$2\"\n")
        dispatch.chmod(0o755)
        ids_path = self.root / "hello-there-ids.txt"
        encoded = subprocess.run([str(native_frontend), str(probe),
            str(lexicon), str(ids_path), "7", "Hello there."],
            capture_output=True, text=True)
        self.assertEqual(encoded.returncode, 0, encoded.stderr)
        self.assertEqual(ids_path.read_text().strip(),
            "0,50,83,54,156,31,16,81,86,123,4,0")
        process, directory = self.start(
            frontend=native_frontend, bundle=bundle, lexicon=lexicon)
        expected = (
            (1, "Hello.", 32400,
             "8f88a076382944cfd26826d964cf12f378ebc8c16f7ce98ea5ed24e98667eda1"),
            (2, "Hello there.", 36000,
             "53dc2fa85f9cf190bc59e51689e45206ca213daad87a71f9024c1bc8b5cafa13"),
            (4, "Hello.", 32400,
             "8f88a076382944cfd26826d964cf12f378ebc8c16f7ce98ea5ed24e98667eda1"),
        )
        for request_id, phrase, frames, digest in expected:
            self.send(process, f"SPEAK\t7\t{phrase}")
            self.assertEqual(self.line(process), f"EVENT\t{request_id}\tSTARTED")
            wav = directory / f"{request_id}.wav"
            self.assertEqual(self.line(process, timeout=180),
                             f"EVENT\t{request_id}\tWAV_READY\t{wav}")
            self.assertEqual(self.line(process), f"EVENT\t{request_id}\tCOMPLETED")
            self.assertEqual(hashlib.sha256(wav.read_bytes()).hexdigest(), digest)
            with wave.open(str(wav), "rb") as audio:
                self.assertEqual((audio.getnchannels(), audio.getsampwidth(),
                                  audio.getframerate(), audio.getnframes()),
                                 (1, 2, 24000, frames))
                samples = audio.readframes(frames)
                self.assertNotEqual(samples, b"\0" * len(samples))
            if request_id == 2:
                self.send(process, "SPEAK\t7\tHello there there.")
                self.assertEqual(self.line(process), "EVENT\t3\tSTARTED")
                self.assertEqual(self.line(process),
                                 "EVENT\t3\tERROR\tnative child failed")
                self.assertFalse((directory / "3.wav").exists())
        print("CKE_NUMERICAL_CASE " + json.dumps({
            "case_id": "kokoro.native-speech-session.three-length-routing-v1",
            "name": "Native session 8-ID to experimental 12-ID to 8-ID recovery",
            "provider": "native_kokoro_speech_session+generated_kokoro_waveform",
            "dtype": "pcm16", "direction": "inference",
            "oracle": "retained CKE seeded WAV hashes, not independent PyTorch",
            "status": "pass", "gate": "session", "blocking": True,
            "max_diff": 0, "tolerance": 0, "metric": "replay WAV hash mismatches",
            "configuration": "8/12/8 IDs; unsupported 16-ID request rejected; seed 7; one session",
            "reproduction_command": "CKE_TTS_RUN_THIRD_LENGTH_FULL=1 "
                "python3 -m unittest tests.test_v8_kokoro_native_speech_session",
        }, sort_keys=True))
        print("CKE_NUMERICAL_CASE " + json.dumps({
            "case_id": "kokoro.native-speech-session.third-length-oracle-parity-v1",
            "name": "Experimental 12-ID connected waveform versus pinned Kokoro",
            "provider": "generated_kokoro_waveform", "dtype": "fp32",
            "direction": "inference", "oracle": "pinned Kokoro full model",
            "status": "not_tested", "gate": "diagnostic", "blocking": False,
            "configuration": "Hello there.; no direct 12-ID full-model capture yet",
            "reason": "pinned-reference waveform capture not yet available",
        }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
