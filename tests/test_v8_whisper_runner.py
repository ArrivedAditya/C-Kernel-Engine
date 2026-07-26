from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "run_whisper_v8.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_whisper_v8", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_whisper_filter_bank_and_forced_prefix_are_stable() -> None:
    runner = _module()
    filters = runner.whisper_mel_filters()
    assert filters.shape == (80, 201)
    assert int((filters != 0).sum()) == 391
    assert hashlib.sha256(filters.tobytes()).hexdigest() == (
        "2150c30cbbeb6029f52002ffa666c1c72d83dbf53f463cb8462052055806e891"
    )
    generation = {
        "decoder_start_token_id": 50258,
        "lang_to_id": {"<|en|>": 50259},
        "task_to_id": {"transcribe": 50359},
        "no_timestamps_token_id": 50363,
    }
    assert runner.forced_decoder_prefix(generation, "en", "transcribe") == [
        50258,
        50259,
        50359,
        50363,
    ]


def test_whisper_tiny_jfk_exact_transcript_when_artifacts_are_configured(
    tmp_path: Path,
) -> None:
    encoder = os.environ.get("CK_WHISPER_ENCODER_RUN_DIR")
    decoder = os.environ.get("CK_WHISPER_DECODER_RUN_DIR")
    wav = os.environ.get("CK_WHISPER_WAV")
    if not all((encoder, decoder, wav)):
        pytest.skip(
            "set CK_WHISPER_ENCODER_RUN_DIR, CK_WHISPER_DECODER_RUN_DIR, "
            "and CK_WHISPER_WAV"
        )
    report = tmp_path / "whisper.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "run",
            "--encoder-run-dir",
            encoder,
            "--decoder-run-dir",
            decoder,
            "--wav",
            wav,
            "--language",
            "en",
            "--task",
            "transcribe",
            "--max-tokens",
            "64",
            "--output",
            str(report),
        ],
        check=True,
        cwd=ROOT,
    )
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["decoder"]["generated_tokens"] == [
        400,
        370,
        452,
        7177,
        6280,
        1029,
        406,
        437,
        428,
        1941,
        393,
        360,
        337,
        291,
        1029,
        437,
        291,
        393,
        360,
        337,
        428,
        1941,
        13,
    ]
    assert result["decoder"]["stop"] == "eos"
    assert result["decoder"]["text"] == (
        " And so my fellow Americans ask not what your country can do for you "
        "ask what you can do for your country."
    )
