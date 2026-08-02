#!/usr/bin/env python3
"""Compare CKE OCR scheduling across hybrid-CPU affinity/thread policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OCR_BENCH = ROOT / "benchmarks" / "bench_v8_qwen3vl_ocr.py"
CONFIGS = {
    "auto20": {"threads": 20, "cpus": None},
    "physical20": {"threads": 20, "cpus": "0,2,4,6,8,10,12,14,16-27"},
    "logical28": {"threads": 28, "cpus": "0-27"},
    "p_core8": {"threads": 8, "cpus": "0,2,4,6,8,10,12,14"},
    "p_smt16": {"threads": 16, "cpus": "0-15"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="Extract visible form fields as compact JSON.")
    parser.add_argument("--config", dest="configs", action="append", choices=sorted(CONFIGS))
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--image-max-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.model = args.model.resolve()
    args.mmproj = args.mmproj.resolve()
    args.image = args.image.resolve()
    args.output = args.output.resolve()
    payload: dict[str, Any] = {
        "schema": "cke.v8.topology_sweep",
        "version": 1,
        "provenance": {
            "model_sha256": sha256(args.model),
            "mmproj_sha256": sha256(args.mmproj),
            "image_sha256": sha256(args.image),
            "engine_sha256": sha256(ROOT / "build" / "libckernel_engine.so"),
        },
        "rows": [],
    }
    save(args.output, payload)

    names = args.configs or list(CONFIGS)
    for repetition in range(args.repetitions):
        order = names if repetition % 2 == 0 else list(reversed(names))
        for name in order:
            config = CONFIGS[name]
            case_path = args.output.parent / "topology-cases" / f"{name}-r{repetition}.json"
            cmd = []
            if config["cpus"]:
                cmd.extend(["taskset", "-c", str(config["cpus"])])
            cmd.extend([
                sys.executable, str(OCR_BENCH),
                "--model", str(args.model), "--mmproj", str(args.mmproj),
                "--images", str(args.image), "--prompt", args.prompt,
                "--threads", str(config["threads"]), "--max-tokens", "1",
                "--context-len", str(args.context),
                "--image-max-tokens", str(args.image_max_tokens),
                "--gemm-schedule", "dynamic",
                "--json-out", str(case_path),
            ])
            env = os.environ.copy()
            env.update({
                "CK_NUM_THREADS": str(config["threads"]),
                "OMP_NUM_THREADS": "1",
                "OMP_DYNAMIC": "FALSE",
            })
            started = time.perf_counter()
            proc = subprocess.run(
                cmd, cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800, check=False,
            )
            wall = time.perf_counter() - started
            detail = None
            if case_path.is_file():
                rows = json.loads(case_path.read_text(encoding="utf-8")).get("results", [])
                detail = rows[0] if rows else None
            row = {
                "config": name,
                "repetition": repetition,
                "threads": config["threads"],
                "cpus": config["cpus"],
                "status": "pass" if proc.returncode == 0 and detail and detail.get("status") == "ok" else "fail",
                "wall_seconds": wall,
                "steady_state_ms": None if not detail else detail.get("steady_state_ms"),
                "encoder_execute_ms": None if not detail else detail.get("encoder_execute_ms"),
                "decoder_forward_mixed_ms": None if not detail else detail.get("decoder_forward_mixed_ms"),
                "generated_text": "" if not detail else detail.get("generated_text", ""),
                "stdout_tail": proc.stdout[-2000:],
            }
            payload["rows"].append(row)
            save(args.output, payload)
            print(
                f"{name}:r{repetition} {row['status']} wall={wall:.2f}s "
                f"steady={row['steady_state_ms']}", flush=True,
            )

    return 0 if all(row["status"] == "pass" for row in payload["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
