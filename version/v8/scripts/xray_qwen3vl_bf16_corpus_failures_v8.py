#!/usr/bin/env python3
"""Attribute every failed image from a Qwen3-VL BF16 corpus certification."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
XRAY_SCRIPT = SCRIPT_DIR / "xray_qwen3vl_bf16_v8.py"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _runtime_map(values: list[str]) -> dict[tuple[int, int, int], Path]:
    result: dict[tuple[int, int, int], Path] = {}
    for value in values:
        try:
            geometry, raw_path = value.split("=", 1)
            grid = tuple(int(part) for part in geometry.lower().split("x"))
        except ValueError as error:
            raise ValueError(f"invalid --runtime {value!r}; expected T×H×W=/path") from error
        if len(grid) != 3:
            raise ValueError(f"invalid runtime geometry {geometry!r}; expected T×H×W")
        path = Path(raw_path).resolve()
        if grid in result:
            raise ValueError(f"duplicate runtime geometry: {grid}")
        if not (path / "call.json").is_file():
            raise ValueError(f"runtime has no call.json: {path}")
        result[grid] = path
    return result


def _sample_image(manifest_path: Path, index: int) -> Path:
    manifest = _read_json(manifest_path)
    samples = manifest.get("samples") or []
    sample = samples[index]
    inputs = sample.get("inputs") or []
    if not inputs or not inputs[0].get("path"):
        raise ValueError(f"sample {index} has no image input")
    return (manifest_path.parent / str(inputs[0]["path"])).resolve()


def _group_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(row.get("first_divergent_checkpoint") or row.get("status") or "unknown")
        for row in rows
    )
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certification-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--runtime", action="append", required=True,
        help="Geometry/runtime mapping, for example 1x56x72=/path/to/runtime",
    )
    parser.add_argument("--weights-bump", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument(
        "--attn-implementation", choices=("auto", "eager", "sdpa"), default="sdpa",
        help="PyTorch attention backend; must match the certification lane",
    )
    parser.add_argument("--max-failures", type=int, default=0)
    parser.add_argument(
        "--discard-captures", action="store_true",
        help="Delete large raw tensor captures after metrics and manifests are sealed",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    certification = _read_json(args.certification_summary.resolve())
    runtimes = _runtime_map(args.runtime)
    failed = [row for row in certification.get("results", []) if not row.get("exact_pre_eos")]
    if args.max_failures > 0:
        failed = failed[: args.max_failures]
    output_dir = args.output_dir.resolve()
    rows: list[dict[str, Any]] = []

    for ordinal, failure in enumerate(failed, start=1):
        index = int(failure["index"])
        sample_dir = output_dir / f"{index:03d}"
        report_path = sample_dir / "xray_summary.json"
        grid = tuple(int(value) for value in failure.get("grid_thw") or ())
        runtime = runtimes.get(grid)
        row: dict[str, Any] = {
            "index": index,
            "id": str(failure.get("id", index)),
            "token_divergence": failure.get("first_divergence"),
            "grid_thw": list(grid),
            "status": "error",
        }
        try:
            if runtime is None:
                raise RuntimeError(f"no runtime registered for grid {grid}")
            image = _sample_image(args.manifest.resolve(), index)
            if not image.is_file():
                raise RuntimeError(f"image is unavailable: {image}")
            if args.force or not report_path.is_file():
                command = [
                    sys.executable, str(XRAY_SCRIPT),
                    "--checkpoint", str(args.checkpoint.resolve()),
                    "--runtime-dir", str(runtime),
                    "--weights-bump", str(args.weights_bump.resolve()),
                    "--call-ir", str(runtime / "call.json"),
                    "--image", str(image),
                    "--output-dir", str(sample_dir),
                    "--threads", str(args.threads),
                    "--max-rounds", str(args.max_rounds),
                    "--attn-implementation", args.attn_implementation,
                ]
                if args.profile:
                    command.extend(["--profile", str(args.profile.resolve())])
                completed = subprocess.run(command, check=False)
                if not report_path.is_file():
                    raise RuntimeError(f"X-ray exited {completed.returncode} without {report_path}")
            report = _read_json(report_path)
            final = report.get("final_report") or {}
            divergence = final.get("first_divergence") or {}
            non_exact = final.get("first_non_exact_checkpoint") or {}
            row.update({
                "status": str(report.get("status", "error")),
                "first_divergent_checkpoint": divergence.get("checkpoint_id"),
                "first_material_checkpoint": divergence.get("checkpoint_id"),
                "classification": divergence.get("classification"),
                "producer": divergence.get("producer"),
                "kernel_id": divergence.get("kernel_id"),
                "metrics": divergence.get("metrics"),
                "first_non_exact_checkpoint": non_exact.get("checkpoint_id"),
                "first_non_exact_metrics": non_exact.get("metrics"),
                "last_passing_checkpoint": final.get("last_passing_checkpoint"),
                "report": str(report_path),
            })
            if args.discard_captures:
                for capture_dir in sample_dir.glob("round_*/capture"):
                    shutil.rmtree(capture_dir)
        except Exception as error:  # A corpus run must preserve later samples.
            row["error"] = str(error)
        rows.append(row)
        summary = {
            "schema": "cke.qwen3vl_bf16_corpus_xray",
            "schema_version": 1,
            "certification_summary": str(args.certification_summary.resolve()),
            "selected_failures": len(failed),
            "completed": len(rows),
            "attributed": sum(bool(item.get("first_divergent_checkpoint")) for item in rows),
            "errors": sum(item.get("status") == "error" for item in rows),
            "checkpoint_groups": _group_counts(rows),
            "results": rows,
        }
        _write_json(output_dir / "summary.json", summary)
        print(
            f"[{ordinal}/{len(failed)}] {row['id']}: {row['status']} "
            f"token={row['token_divergence']} edge={row.get('first_divergent_checkpoint')}",
            flush=True,
        )

    return 0 if rows and all(row.get("status") != "error" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
