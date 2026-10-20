"""Optional generated Kokoro WAV -> generated Whisper diagnostic round trip.

The transcript gate checks known text. It does not certify Kokoro's separate
waveform comparison against the pinned PyTorch model.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]
WHISPER = ROOT / "version/v8/scripts/run_whisper_v8.py"
CASES = (("hello", "Hello.", 8, 32400),
         ("hello_there", "Hello there.", 12, 36000))
REQUIRED_ENV = ("CKE_TTS_ROUNDTRIP_BUNDLE_DIR",
                "CKE_TTS_ROUNDTRIP_FRONTEND_DIR",
                "CKE_TTS_ROUNDTRIP_WHISPER_ENCODER_DIR",
                "CKE_TTS_ROUNDTRIP_WHISPER_DECODER_DIR")
REPRODUCE = ("CKE_TTS_RUN_WHISPER_ROUNDTRIP=1 " +
             " ".join(f"{key}=<path>" for key in REQUIRED_ENV) +
             " python3 -m unittest tests.test_v8_kokoro_whisper_roundtrip")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report(case: str, stage: str, status: str, reason: str,
            **details: object) -> None:
    data = {
        "case_id": f"kokoro.whisper-roundtrip.{case}.{stage}-v1",
        "name": f"Generated Kokoro to Whisper {case} {stage}",
        "provider": ("generated_kokoro_runtime_waveform" if stage == "tts"
                     else "generated_whisper_encoder_decoder" if stage == "asr"
                     else "native_kokoro_text+generated_whisper"),
        "dtype": "pcm16" if stage != "fidelity" else "utf8",
        "direction": "inference",
        "oracle": "known source text and generated-runtime artifact identities",
        "gate": "transcript" if stage == "fidelity" else "execution",
        "status": status,
        "blocking": status != "not_tested",
        "configuration": "pinned af_heart; seed 7; 24 kHz mono PCM16; English",
        "reproduction_command": REPRODUCE,
    }
    if reason:
        data["reason"] = reason
    if status != "not_tested":
        data.update(max_diff=0 if status == "pass" else 1, tolerance=0)
    data.update(details)
    print("CKE_NUMERICAL_CASE " + json.dumps(data, sort_keys=True))


class KokoroWhisperRoundTripTest(unittest.TestCase):
    def test_asset_coverage_is_explicit(self):
        missing = [key for key in REQUIRED_ENV if not os.environ.get(key)]
        if os.environ.get("CKE_TTS_RUN_WHISPER_ROUNDTRIP") and not missing:
            return
        reason = ("asset-backed run not requested" if not
                  os.environ.get("CKE_TTS_RUN_WHISPER_ROUNDTRIP") else
                  "missing required assets: " + ", ".join(missing))
        for case, _, _, _ in CASES:
            for stage in ("tts", "asr", "fidelity"):
                _report(case, stage, "not_tested", reason)
        self.skipTest(reason)

    @unittest.skipUnless(os.environ.get("CKE_TTS_RUN_WHISPER_ROUNDTRIP"),
                         "generated Kokoro and Whisper assets not requested")
    def test_generated_speech_round_trip(self):
        missing = [key for key in REQUIRED_ENV if not os.environ.get(key)]
        self.assertFalse(missing, "missing assets: " + ", ".join(missing))
        bundle, frontend, encoder, decoder = (Path(os.environ[key])
                                               for key in REQUIRED_ENV)
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        self.assertTrue(manifest["runtime_lengths"])
        self.assertIn(12, manifest["execution_only_phoneme_counts"])
        self.assertEqual(len({item["library_sha256"] for item in
                              manifest["buckets"].values()}), 1)
        for _, _, count, _ in CASES:
            self.assertIn(count, manifest["supported_phoneme_counts"])
            self.assertEqual(
                _hash(bundle / "64" / "generated.so"),
                manifest["buckets"][str(count)]["library_sha256"],
            )
        lexicon = frontend / "us-gold.cklex"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            failures = []
            for case, text, count, expected_samples in CASES:
                wav = directory / f"{case}.wav"
                report_path = directory / f"{case}-whisper.json"
                tts = subprocess.run([
                    str(frontend / "native-kokoro-text"), str(bundle),
                    str(lexicon), str(wav), "7", text,
                ], capture_output=True, text=True, timeout=120)
                tts_reason = ""
                wav_hash = ""
                if tts.returncode:
                    tts_reason = f"native text-to-WAV exited {tts.returncode}: {tts.stderr[-300:]}"
                else:
                    try:
                        with wave.open(str(wav), "rb") as audio:
                            geometry = (audio.getnchannels(), audio.getsampwidth(),
                                        audio.getframerate(), audio.getnframes())
                        if geometry != (1, 2, 24000, expected_samples):
                            tts_reason = f"unexpected WAV geometry: {geometry}"
                        else:
                            wav_hash = _hash(wav)
                    except (OSError, wave.Error) as error:
                        tts_reason = f"invalid native WAV: {error}"
                _report(case, "tts", "fail" if tts_reason else "pass",
                        tts_reason, phoneme_count=count,
                        generated_library_sha256=manifest["buckets"][str(count)][
                            "library_sha256"], wav_sha256=wav_hash)
                if tts_reason:
                    failures.append(f"{case} TTS: {tts_reason}")
                    _report(case, "asr", "not_tested", "TTS did not produce valid WAV")
                    _report(case, "fidelity", "not_tested", "ASR could not run")
                    continue
                asr = subprocess.run([
                    sys.executable, str(WHISPER), "run",
                    "--encoder-run-dir", str(encoder),
                    "--decoder-run-dir", str(decoder),
                    "--wav", str(wav), "--language", "en",
                    "--task", "transcribe", "--max-tokens", "48",
                    "--output", str(report_path),
                ], capture_output=True, text=True, timeout=120)
                asr_reason = ""
                result = {}
                if asr.returncode:
                    asr_reason = f"generated Whisper exited {asr.returncode}: {asr.stderr[-300:]}"
                else:
                    try:
                        result = json.loads(report_path.read_text())
                        if (result.get("status") != "ok" or
                                result.get("wav_sha256") != wav_hash or
                                result.get("encoder", {}).get("audio", {}).get(
                                    "source_sample_rate") != 24000 or
                                not result.get("encoder_runtime_sha256") or
                                not result.get("decoder_runtime_sha256")):
                            asr_reason = "generated ASR report has wrong input or identity"
                    except (OSError, ValueError) as error:
                        asr_reason = f"invalid generated ASR report: {error}"
                _report(case, "asr", "fail" if asr_reason else "pass",
                        asr_reason, wav_sha256=wav_hash,
                        encoder_runtime_sha256=result.get("encoder_runtime_sha256", ""),
                        decoder_runtime_sha256=result.get("decoder_runtime_sha256", ""))
                if asr_reason:
                    failures.append(f"{case} ASR: {asr_reason}")
                    _report(case, "fidelity", "not_tested", "ASR did not return valid transcript")
                    continue
                transcript = result.get("decoder", {}).get("transcript_text")
                matched = isinstance(transcript, str) and transcript.strip().casefold() == text.casefold()
                _report(case, "fidelity", "pass" if matched else "fail",
                        "" if matched else f"expected {text!r}; got {transcript!r}",
                        source_text=text, transcript=transcript,
                        wav_sha256=wav_hash)
                if not matched:
                    failures.append(f"{case} transcript: {transcript!r}")
            retain = os.environ.get("CKE_TTS_ROUNDTRIP_RETAIN_DIR")
            if retain:
                target = Path(retain)
                target.mkdir(parents=True, exist_ok=True)
                saved = Path(tempfile.mkdtemp(prefix="kokoro-whisper-", dir=target))
                for item in directory.iterdir():
                    shutil.copy2(item, saved / item.name)
            self.assertFalse(failures, "; ".join(failures))


if __name__ == "__main__":
    unittest.main()
