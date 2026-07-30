#!/usr/bin/env python3
"""Compare generated CKE Whisper transcription with Hugging Face PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_whisper_v8.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(checkpoint: Path) -> dict[str, dict[str, object]]:
    files = sorted(
        path
        for path in checkpoint.iterdir()
        if path.is_file()
        and path.suffix in {".json", ".safetensors", ".txt", ".model"}
    )
    return {
        str(path.relative_to(checkpoint)): {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    }


def validate_runtime_checkpoint(
    run_dir: Path,
    expected: dict[str, dict[str, object]],
    role: str,
) -> None:
    stamp_path = run_dir / ".ck-whisper-runtime.json"
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"missing or invalid Whisper {role} provenance stamp: {stamp_path}"
        ) from exc
    inputs = stamp.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("role") != role:
        raise RuntimeError(
            f"Whisper {role} provenance has the wrong runtime role: {stamp_path}"
        )
    if inputs.get("checkpoint") != expected:
        raise RuntimeError(
            f"Whisper {role} runtime was not built from the requested "
            f"checkpoint: {run_dir}"
        )


def _read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError("PyTorch Whisper oracle requires PCM16 WAV")
        rate = source.getframerate()
        channels = source.getnchannels()
        raw = source.readframes(source.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    samples = samples.reshape(-1, channels).mean(axis=1)
    return samples / 32768.0, rate


def first_token_difference(
    subject: list[int], oracle: list[int]
) -> dict[str, int] | None:
    common = min(len(subject), len(oracle))
    for index in range(common):
        if subject[index] != oracle[index]:
            return {
                "index": index,
                "subject": subject[index],
                "oracle": oracle[index],
            }
    if len(subject) != len(oracle):
        return {
            "index": common,
            "subject": subject[common] if common < len(subject) else -1,
            "oracle": oracle[common] if common < len(oracle) else -1,
        }
    return None


def _pytorch_reference(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    samples, sample_rate = _read_pcm16(args.wav)
    processor = WhisperProcessor.from_pretrained(
        args.checkpoint, local_files_only=True
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.checkpoint, local_files_only=True
    ).eval()
    inputs = processor(
        samples, sampling_rate=sample_rate, return_tensors="pt"
    )
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            inputs.input_features,
            language=args.language,
            task=args.task,
            return_timestamps=args.timestamps,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - started
    tokens = [int(value) for value in generated[0].tolist()]
    text = processor.decode(
        generated[0],
        skip_special_tokens=True,
        decode_with_timestamps=args.timestamps,
        clean_up_tokenization_spaces=False,
    )
    return {
        "tokens": tokens,
        "text": text,
        "seconds": elapsed,
        "pytorch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-run-dir", type=Path, required=True)
    parser.add_argument("--decoder-run-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--task", choices=("transcribe", "translate"), default="transcribe"
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    args.checkpoint = args.checkpoint.resolve()
    args.encoder_run_dir = args.encoder_run_dir.resolve()
    args.decoder_run_dir = args.decoder_run_dir.resolve()
    args.wav = args.wav.resolve()
    checkpoint = checkpoint_identity(args.checkpoint)
    validate_runtime_checkpoint(args.encoder_run_dir, checkpoint, "encoder")
    validate_runtime_checkpoint(args.decoder_run_dir, checkpoint, "decoder")
    os.environ["CK_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    with tempfile.TemporaryDirectory(prefix="cke-whisper-pytorch-") as text:
        cke_path = Path(text) / "cke.json"
        subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "run",
                "--encoder-run-dir",
                str(args.encoder_run_dir),
                "--decoder-run-dir",
                str(args.decoder_run_dir),
                "--wav",
                str(args.wav),
                "--language",
                args.language,
                "--task",
                args.task,
                "--max-tokens",
                str(args.max_tokens),
                *(["--timestamps"] if args.timestamps else []),
                "--output",
                str(cke_path),
            ],
            check=True,
        )
        cke = json.loads(cke_path.read_text(encoding="utf-8"))

    oracle = _pytorch_reference(args)
    subject_tokens = [
        int(value) for value in cke["decoder"]["generated_tokens"]
    ]
    oracle_tokens = [int(value) for value in oracle["tokens"]]
    difference = first_token_difference(subject_tokens, oracle_tokens)
    report = {
        "schema": "cke.whisper_e2e_pytorch_parity",
        "schema_version": 1,
        "status": "pass" if difference is None else "fail",
        "language": args.language,
        "task": args.task,
        "timestamps": bool(args.timestamps),
        "max_tokens": args.max_tokens,
        "threads": args.threads,
        "wav": str(args.wav),
        "wav_sha256": _sha256(args.wav),
        "checkpoint": str(args.checkpoint),
        "checkpoint_files": checkpoint,
        "cke": cke,
        "pytorch": oracle,
        "matched_tokens": (
            len(subject_tokens)
            if difference is None
            else int(difference["index"])
        ),
        "subject_token_count": len(subject_tokens),
        "oracle_token_count": len(oracle_tokens),
        "first_difference": difference,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"status={report['status']} "
        f"matched={report['matched_tokens']}/"
        f"{max(len(subject_tokens), len(oracle_tokens))} "
        f"report={args.output}"
    )
    return 0 if difference is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
