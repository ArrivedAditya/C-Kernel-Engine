#!/usr/bin/env python3
"""Compare CKE, PyTorch, and whisper.cpp Whisper execution on one host.

The report keeps process wall time separate from backend compute time. Each
repetition rotates backend order so sustained CPU state does not always favor
the same implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any
import wave


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
RUNNER = SCRIPTS / "run_whisper_v8.py"
sys.path.insert(0, str(SCRIPTS))
import compare_whisper_e2e_pytorch_v8 as pytorch_oracle  # noqa: E402


TIMING_RE = re.compile(
    r"whisper_print_timings:\s+(?P<name>load|mel|sample|encode|decode|batchd|"
    r"prompt|total) time\s*=\s*(?P<milliseconds>[0-9.]+) ms"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wav_metadata(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as source:
        frames = source.getnframes()
        sample_rate = source.getframerate()
        return {
            "path": str(path),
            "sha256": sha256(path),
            "frames": frames,
            "sample_rate": sample_rate,
            "channels": source.getnchannels(),
            "sample_width_bytes": source.getsampwidth(),
            "duration_seconds": frames / sample_rate,
        }


def cpu_metadata() -> dict[str, Any]:
    model = ""
    flags: list[str] = []
    try:
        for line in Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("model name") and not model:
                model = line.split(":", 1)[1].strip()
            elif line.startswith("flags") and not flags:
                flags = line.split(":", 1)[1].strip().split()
    except OSError:
        pass
    return {
        "model": model or platform.processor(),
        "logical_cpus": os.cpu_count(),
        "machine": platform.machine(),
        "flags": flags,
    }


def parse_whisper_cpp_timings(stderr: str) -> dict[str, float]:
    result = {
        match.group("name"): float(match.group("milliseconds")) / 1000.0
        for match in TIMING_RE.finditer(stderr)
    }
    if "total" not in result:
        raise ValueError("whisper.cpp output did not contain total timing")
    return result


def parse_whisper_cpp_result(payload: dict[str, Any]) -> dict[str, Any]:
    transcription = payload.get("transcription")
    if not isinstance(transcription, list):
        raise ValueError("whisper.cpp JSON has no transcription array")
    text_parts: list[str] = []
    tokens: list[int] = []
    for segment in transcription:
        if not isinstance(segment, dict):
            continue
        text_parts.append(str(segment.get("text", "")))
        for token in segment.get("tokens", []):
            if not isinstance(token, dict):
                continue
            token_text = str(token.get("text", ""))
            if token_text.startswith("[_") and token_text.endswith("_]"):
                continue
            tokens.append(int(token["id"]))
    return {"text": "".join(text_parts), "tokens": tokens}


def backend_order(index: int, backends: list[str]) -> list[str]:
    if not backends:
        return []
    offset = index % len(backends)
    return backends[offset:] + backends[:offset]


def run_process(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path = ROOT,
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed, time.perf_counter() - started


def run_cke(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "cke.json"
    command = [
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
        "--output",
        str(report_path),
    ]
    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(args.threads)
    env["OMP_NUM_THREADS"] = str(args.threads)
    completed, wall = run_process(command, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"CKE failed:\n{completed.stderr}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    encoder = payload["encoder"]
    decoder = payload["decoder"]
    compute = (
        float(encoder["audio_encoder_seconds"])
        + float(decoder["prefill_seconds"])
        + float(decoder["decode_seconds"])
    )
    return {
        "backend": "cke",
        "command": command,
        "wall_seconds": wall,
        "compute_seconds": compute,
        "stages": {
            "audio_encoder_seconds": encoder["audio_encoder_seconds"],
            "prefill_seconds": decoder["prefill_seconds"],
            "decode_seconds": decoder["decode_seconds"],
        },
        "tokens": [int(value) for value in decoder["generated_tokens"]],
        "text": str(decoder["transcript_text"]),
        "runtime": {
            "encoder_path": str(args.encoder_run_dir / "libmodel.so"),
            "encoder_sha256": payload["encoder_runtime_sha256"],
            "decoder_path": str(args.decoder_run_dir / "libmodel.so"),
            "decoder_sha256": payload["decoder_runtime_sha256"],
        },
    }


def run_pytorch(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "pytorch.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--pytorch-worker",
        "--checkpoint",
        str(args.checkpoint),
        "--wav",
        str(args.wav),
        "--language",
        args.language,
        "--task",
        args.task,
        "--max-tokens",
        str(args.max_tokens),
        "--threads",
        str(args.threads),
        "--worker-output",
        str(report_path),
    ]
    completed, wall = run_process(command, env=os.environ.copy())
    if completed.returncode != 0:
        raise RuntimeError(f"PyTorch failed:\n{completed.stderr}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "backend": "pytorch",
        "command": command,
        "wall_seconds": wall,
        "compute_seconds": (
            float(payload["frontend_seconds"])
            + float(payload["generation_seconds"])
        ),
        "stages": {
            "load_seconds": payload["load_seconds"],
            "frontend_seconds": payload["frontend_seconds"],
            "generation_seconds": payload["generation_seconds"],
        },
        "tokens": payload["tokens"],
        "text": payload["transcript_text"],
        "runtime": {
            "pytorch": payload["pytorch"],
            "transformers": payload["transformers"],
        },
    }


def run_whisper_cpp(
    args: argparse.Namespace, run_dir: Path
) -> dict[str, Any]:
    output_prefix = run_dir / "whisper-cpp"
    output_path = output_prefix.with_suffix(".json")
    command = [
        str(args.whisper_cpp_cli),
        "-m",
        str(args.whisper_cpp_model),
        "-f",
        str(args.wav),
        "-t",
        str(args.threads),
        "-l",
        args.language,
        "-nt",
        "-oj",
        "-ojf",
        "-of",
        str(output_prefix),
        "--beam-size",
        "1",
        "--best-of",
        "1",
        "--no-fallback",
        "--temperature",
        "0",
        "--no-gpu",
    ]
    if args.task == "translate":
        command.append("--translate")
    completed, wall = run_process(command, env=os.environ.copy())
    if completed.returncode != 0:
        raise RuntimeError(f"whisper.cpp failed:\n{completed.stderr}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    decoded = parse_whisper_cpp_result(payload)
    timings = parse_whisper_cpp_timings(completed.stderr)
    return {
        "backend": "whisper_cpp",
        "command": command,
        "wall_seconds": wall,
        "compute_seconds": timings["total"],
        "stages": {
            f"{name}_seconds": seconds for name, seconds in timings.items()
        },
        "tokens": decoded["tokens"],
        "text": decoded["text"],
        "runtime": {
            "cli_path": str(args.whisper_cpp_cli),
            "cli_sha256": sha256(args.whisper_cpp_cli),
            "model_path": str(args.whisper_cpp_model),
            "model_sha256": sha256(args.whisper_cpp_model),
            "system_info": payload.get("systeminfo"),
        },
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_backend: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_backend.setdefault(str(run["backend"]), []).append(run)
    medians = {
        backend: {
            "wall_seconds": statistics.median(
                float(run["wall_seconds"]) for run in backend_runs
            ),
            "compute_seconds": statistics.median(
                float(run["compute_seconds"]) for run in backend_runs
            ),
        }
        for backend, backend_runs in by_backend.items()
    }
    reference = (
        "whisper_cpp" if "whisper_cpp" in medians else "pytorch"
    )
    reference_seconds = medians[reference]["compute_seconds"]
    for backend, values in medians.items():
        values["compute_ratio_vs_reference"] = (
            values["compute_seconds"] / reference_seconds
        )
    return {"reference_backend": reference, "backends": medians}


def validate_results(runs: list[dict[str, Any]]) -> dict[str, Any]:
    first = runs[0]
    expected_tokens = first["tokens"]
    expected_text = first["text"]
    mismatches = []
    for run in runs[1:]:
        if run["tokens"] != expected_tokens or run["text"] != expected_text:
            mismatches.append(
                {
                    "backend": run["backend"],
                    "repetition": run["repetition"],
                    "token_match": run["tokens"] == expected_tokens,
                    "text_match": run["text"] == expected_text,
                }
            )
    return {
        "status": "pass" if not mismatches else "fail",
        "token_count": len(expected_tokens),
        "mismatches": mismatches,
    }


def pytorch_worker(args: argparse.Namespace) -> int:
    oracle_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        wav=args.wav,
        language=args.language,
        task=args.task,
        timestamps=False,
        max_tokens=args.max_tokens,
        threads=args.threads,
    )
    payload = pytorch_oracle._pytorch_reference(oracle_args)
    args.worker_output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--encoder-run-dir", type=Path)
    result.add_argument("--decoder-run-dir", type=Path)
    result.add_argument("--wav", type=Path, required=True)
    result.add_argument("--whisper-cpp-cli", type=Path)
    result.add_argument("--whisper-cpp-model", type=Path)
    result.add_argument("--language", default="en")
    result.add_argument(
        "--task", choices=("transcribe", "translate"), default="transcribe"
    )
    result.add_argument("--max-tokens", type=int, default=128)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--repetitions", type=int, default=3)
    result.add_argument("--output", type=Path)
    result.add_argument("--pytorch-worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for name in (
        "checkpoint",
        "encoder_run_dir",
        "decoder_run_dir",
        "wav",
        "whisper_cpp_cli",
        "whisper_cpp_model",
        "output",
        "worker_output",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.pytorch_worker:
        if args.worker_output is None:
            raise SystemExit("--pytorch-worker requires --worker-output")
        return pytorch_worker(args)
    if args.encoder_run_dir is None or args.decoder_run_dir is None:
        raise SystemExit("CKE benchmarking requires both runtime directories")
    if bool(args.whisper_cpp_cli) != bool(args.whisper_cpp_model):
        raise SystemExit("provide both whisper.cpp executable and model")
    if args.output is None:
        raise SystemExit("--output is required")
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")

    checkpoint_files = pytorch_oracle.checkpoint_identity(args.checkpoint)
    pytorch_oracle.validate_runtime_checkpoint(
        args.encoder_run_dir, checkpoint_files, "encoder"
    )
    pytorch_oracle.validate_runtime_checkpoint(
        args.decoder_run_dir, checkpoint_files, "decoder"
    )

    backends = ["cke", "pytorch"]
    if args.whisper_cpp_cli:
        backends.append("whisper_cpp")
    runners = {
        "cke": run_cke,
        "pytorch": run_pytorch,
        "whisper_cpp": run_whisper_cpp,
    }
    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cke-whisper-benchmark-") as text:
        temp = Path(text)
        for repetition in range(args.repetitions):
            order = backend_order(repetition, backends)
            print(
                f"repetition={repetition + 1}/{args.repetitions} "
                f"order={','.join(order)}",
                flush=True,
            )
            for backend in order:
                run_dir = temp / f"{repetition:02d}-{backend}"
                run_dir.mkdir()
                result = runners[backend](args, run_dir)
                result["repetition"] = repetition
                result["order"] = order
                runs.append(result)
                print(
                    f"  {backend}: wall={result['wall_seconds']:.3f}s "
                    f"compute={result['compute_seconds']:.3f}s",
                    flush=True,
                )

    parity = validate_results(runs)
    report = {
        "schema": "cke.whisper_backend_benchmark",
        "schema_version": 1,
        "status": parity["status"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": cpu_metadata(),
        "threads": args.threads,
        "repetitions": args.repetitions,
        "wav": wav_metadata(args.wav),
        "checkpoint": {
            "path": str(args.checkpoint),
            "files": checkpoint_files,
        },
        "parity": parity,
        "summary": summarize(runs),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"status={report['status']} report={args.output}", flush=True
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
