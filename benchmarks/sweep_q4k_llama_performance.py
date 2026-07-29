#!/usr/bin/env python3
"""Sweep exact Q4_K x Q8_K providers and compare production with llama.cpp.

This runner is deliberately report-only. It never changes kernel maps or
runtime dispatch. The focused C benchmark compares every CKE candidate
bit-for-bit with the certified CKE 4M reference; the llama.cpp production
oracle independently certifies the public dispatcher and supplies the external
performance reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
QWEN36_HOT_SHAPES = (
    "qwen36_prompt33_mlp_gate_up=33x34816x5120",
    "qwen36_prompt33_recurrent_gate=33x6144x5120",
    "qwen36_context1034_mlp_gate_up=1034x34816x5120",
    "qwen36_context1034_recurrent_gate=1034x6144x5120",
)
PROVIDER_LAYOUTS = {
    "mreuse": "q4_k_packed_meta_x8",
    "baseline": "q4_k_packed_meta_x8",
    "4m": "q4_k_packed_meta_x8",
    "8m": "q4_k_packed_meta_x8",
    "4m-vnni-x8": "q4_k_packed_vnni_x8",
}
PROVIDER_LINE = re.compile(
    r"provider=(?P<provider>\S+)\s+"
    r"reference_provider=(?P<reference_provider>\S+)\s+"
    r"exact=(?P<exact>true|false)\s+"
    r"M=(?P<M>\d+)\s+N=(?P<N>\d+)\s+K=(?P<K>\d+)\s+"
    r"threads=(?P<threads>\d+)\s+tile_m=(?P<tile_m>\d+)\s+"
    r"time_ms=(?P<time_ms>[0-9.]+)\s+gflops=(?P<gflops>[0-9.]+)\s+"
    r"checksum=(?P<checksum>\S+)"
)
ORACLE_LINE = re.compile(
    r"Q4_K performance:\s+M=(?P<M>\d+)\s+N=(?P<N>\d+)\s+K=(?P<K>\d+)\s+"
    r"repeats=(?P<repeats>\d+)\s+ck_ms=(?P<ck_ms>[0-9.]+)\s+"
    r"llama_ms=(?P<llama_ms>[0-9.]+)\s+"
    r"ck_over_llama=(?P<ratio>[0-9.]+)"
)


@dataclass(frozen=True)
class Shape:
    name: str
    M: int
    N: int
    K: int


def parse_int_csv(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"expected positive comma-separated integers: {text!r}")
    return values


def parse_shape(text: str) -> Shape:
    if "=" not in text:
        raise ValueError("shape must be NAME=MxNxK")
    name, dims = text.split("=", 1)
    parts = re.split(r"[xX,]", dims)
    if not name.strip() or len(parts) != 3:
        raise ValueError("shape must be NAME=MxNxK")
    M, N, K = (int(part.strip()) for part in parts)
    if min(M, N, K) <= 0 or K % 256 != 0:
        raise ValueError("shape dimensions must be positive and K must be divisible by 256")
    return Shape(name=name.strip(), M=M, N=N, K=K)


def parse_provider_output(text: str) -> dict[str, Any]:
    match = PROVIDER_LINE.search(text)
    if match is None:
        raise ValueError(f"provider result line not found in output:\n{text[-2000:]}")
    row: dict[str, Any] = match.groupdict()
    for key in ("M", "N", "K", "threads", "tile_m"):
        row[key] = int(row[key])
    for key in ("time_ms", "gflops", "checksum"):
        row[key] = float(row[key])
    row["exact"] = row["exact"] == "true"
    row["kind"] = "cke_provider"
    return row


def parse_oracle_output(text: str) -> dict[str, Any]:
    match = ORACLE_LINE.search(text)
    if match is None:
        raise ValueError(f"llama oracle result line not found in output:\n{text[-2000:]}")
    row: dict[str, Any] = match.groupdict()
    for key in ("M", "N", "K", "repeats"):
        row[key] = int(row[key])
    for key in ("ck_ms", "llama_ms", "ratio"):
        row[key] = float(row[key])
    row["bit_exact"] = "bit_exact" in text and "[PASS]" in text
    row["kind"] = "llama_oracle"
    return row


def build_provider_jobs(
    providers: Iterable[str], tiles: Iterable[int]
) -> list[tuple[str, int]]:
    jobs: list[tuple[str, int]] = []
    for provider in providers:
        if provider in {"mreuse", "baseline"}:
            jobs.extend((provider, tile) for tile in tiles)
        else:
            jobs.append((provider, 0))
    return jobs


def _first_cpuinfo_value(key: str) -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith(key.lower() + "\t") or line.lower().startswith(key.lower() + " "):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def collect_hardware() -> dict[str, Any]:
    flags = set(_first_cpuinfo_value("flags").split())
    relevant_isa = [
        name
        for name in ("avx2", "avx_vnni", "avx512f", "avx512_vnni", "amx_int8")
        if name in flags
    ]
    physical_cores: int | None = None
    try:
        physical_cores = int(
            subprocess.run(
                ["lscpu", "-p=core"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.count("\n")
        )
        core_ids = {
            line.strip()
            for line in subprocess.run(
                ["lscpu", "-p=core"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.splitlines()
            if line and not line.startswith("#")
        }
        physical_cores = len(core_ids)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {
        "architecture": platform.machine(),
        "cpu": _first_cpuinfo_value("model name") or platform.processor(),
        "vendor": _first_cpuinfo_value("vendor_id"),
        "family": _first_cpuinfo_value("cpu family"),
        "model": _first_cpuinfo_value("model"),
        "stepping": _first_cpuinfo_value("stepping"),
        "physical_cores_visible": physical_cores,
        "logical_cpus_visible": os.cpu_count(),
        "isa": relevant_isa,
    }


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        errors="replace",
        capture_output=True,
        check=False,
    )


def _base_env(threads: int, library_dirs: list[Path]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CK_NUM_THREADS": str(threads),
            "CK_THREADPOOL_CAPACITY": str(threads),
            "CK_Q4K_LLAMA_THREADS": str(threads),
            "OMP_NUM_THREADS": "1",
        }
    )
    if library_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        prefix = ":".join(str(path.resolve()) for path in library_dirs)
        env["LD_LIBRARY_PATH"] = prefix + (":" + existing if existing else "")
    return env


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    shapes = [parse_shape(text) for text in args.shape]
    threads = parse_int_csv(args.threads)
    tiles = parse_int_csv(args.tiles)
    providers = [part.strip() for part in args.providers.split(",") if part.strip()]
    provider_jobs = build_provider_jobs(providers, tiles)
    library_dirs = [path.resolve() for path in args.library_dir]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for shape in shapes:
        provider_m = shape.M - (shape.M % 4)
        for thread_count in threads:
            env = _base_env(thread_count, library_dirs)
            for provider, tile_m in (provider_jobs if provider_m else []):
                effective_tile = tile_m or args.fixed_tile
                command = [
                    str(args.provider_bin.resolve()),
                    "--provider",
                    provider,
                    "--m",
                    str(provider_m),
                    "--n",
                    str(shape.N),
                    "--k",
                    str(shape.K),
                    "--threads",
                    str(thread_count),
                    "--tile-m",
                    str(effective_tile),
                    "--warmup",
                    str(args.warmup),
                    "--iterations",
                    str(args.iterations),
                ]
                completed = _run(command, env)
                combined = completed.stdout + completed.stderr
                if completed.returncode == 4 and "provider unavailable" in combined:
                    results.append(
                        {
                            "kind": "cke_provider",
                            "shape_name": shape.name,
                            "requested_M": shape.M,
                            "provider": provider,
                            "provider_layout": PROVIDER_LAYOUTS.get(provider, "unknown"),
                            "phase": "prefill",
                            "tile_m": effective_tile,
                            "threads": thread_count,
                            "status": "unsupported",
                        }
                    )
                    continue
                if completed.returncode != 0:
                    failure = {
                        "shape_name": shape.name,
                        "command": command,
                        "returncode": completed.returncode,
                        "output_tail": combined[-2000:],
                    }
                    failures.append(failure)
                    if not args.keep_going:
                        raise RuntimeError(json.dumps(failure, indent=2))
                    continue
                row = parse_provider_output(combined)
                row["shape_name"] = shape.name
                row["requested_M"] = shape.M
                row["provider_layout"] = PROVIDER_LAYOUTS.get(provider, "unknown")
                row["phase"] = "prefill"
                row["status"] = "pass" if row["exact"] else "fail"
                results.append(row)

            if args.oracle_bin is not None:
                env.update(
                    {
                        "CK_Q4K_PERF_M": str(shape.M),
                        "CK_Q4K_PERF_N": str(shape.N),
                        "CK_Q4K_PERF_K": str(shape.K),
                        "CK_Q4K_PERF_REPEATS": str(args.oracle_repeats),
                        "CK_Q4K_LLAMA_MAX_RATIO": "0",
                    }
                )
                command = [str(args.oracle_bin.resolve()), "--perf"]
                completed = _run(command, env)
                combined = completed.stdout + completed.stderr
                if completed.returncode != 0:
                    failure = {
                        "shape_name": shape.name,
                        "command": command,
                        "returncode": completed.returncode,
                        "output_tail": combined[-2000:],
                    }
                    failures.append(failure)
                    if not args.keep_going:
                        raise RuntimeError(json.dumps(failure, indent=2))
                else:
                    row = parse_oracle_output(combined)
                    row["shape_name"] = shape.name
                    row["threads"] = thread_count
                    row["provider"] = "production"
                    row["provider_layout"] = "llama_q4_K_8x8"
                    row["phase"] = "prefill"
                    row["status"] = "pass" if row["bit_exact"] else "fail"
                    results.append(row)

    return {
        "schema_version": 1,
        "suite": "q4_k_x_q8_k_provider_sweep",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": collect_hardware(),
        "software_provenance": {
            "engine_commit": _git_commit(),
            "python": platform.python_version(),
            "isa_label": args.isa_label,
            "compiler": args.compiler_label,
        },
        "configuration": {
            "shapes": [asdict(shape) for shape in shapes],
            "threads": threads,
            "tiles_m": tiles,
            "providers": providers,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "oracle_repeats": args.oracle_repeats,
        },
        "results": results,
        "failures": failures,
    }


def print_summary(report: dict[str, Any]) -> None:
    print(
        "kind\tshape\tprovider\tthreads\ttile_m\tcke_ms\tllama_ms\tratio\texact"
    )
    for row in report["results"]:
        if row["kind"] == "cke_provider":
            print(
                f"provider\t{row.get('shape_name', '')}\t{row.get('provider', '')}\t"
                f"{row.get('threads', '')}\t{row.get('tile_m', '')}\t"
                f"{row.get('time_ms', '')}\t\t\t{row.get('exact', '')}"
            )
        else:
            print(
                f"oracle\t{row.get('shape_name', '')}\tproduction\t"
                f"{row.get('threads', '')}\t\t{row.get('ck_ms', '')}\t"
                f"{row.get('llama_ms', '')}\t{row.get('ratio', '')}\t"
                f"{row.get('bit_exact', '')}"
            )
    provider_times = [
        row["time_ms"]
        for row in report["results"]
        if row["kind"] == "cke_provider" and "time_ms" in row
    ]
    if provider_times:
        print(f"provider_time_median_ms={statistics.median(provider_times):.3f}")


def write_csv_table(report: dict[str, Any], path: Path) -> None:
    """Write a flat table suitable for inspection and future page generation."""
    columns = [
        "kind",
        "cpu",
        "isa_label",
        "compiler",
        "engine_commit",
        "phase",
        "shape_name",
        "M",
        "requested_M",
        "N",
        "K",
        "provider",
        "provider_layout",
        "threads",
        "tile_m",
        "time_ms",
        "gflops",
        "ck_ms",
        "llama_ms",
        "ratio",
        "exact",
        "bit_exact",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in report["results"]:
            flattened = dict(row)
            flattened.update(
                {
                    "cpu": report["hardware"]["cpu"],
                    "isa_label": report["software_provenance"]["isa_label"],
                    "compiler": report["software_provenance"]["compiler"],
                    "engine_commit": report["software_provenance"]["engine_commit"],
                }
            )
            writer.writerow(flattened)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-bin",
        type=Path,
        default=ROOT / "build" / "bench_q4k_exact_prefill",
    )
    parser.add_argument(
        "--oracle-bin",
        type=Path,
        default=ROOT / "build" / "test_q4k_q8k_llama_packed",
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="NAME=MxNxK; repeat for multiple shapes",
    )
    parser.add_argument("--threads", default="1,4,8,12,16,24")
    parser.add_argument("--tiles", default="4,6,8,12,16,32")
    parser.add_argument("--providers", default="mreuse,4m,8m,4m-vnni-x8")
    parser.add_argument("--fixed-tile", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--oracle-repeats", type=int, default=3)
    parser.add_argument("--library-dir", action="append", type=Path, default=[])
    parser.add_argument("--isa-label", default="native")
    parser.add_argument(
        "--compiler-label",
        default="unknown",
        help="Compiler provenance label, for example icx-2025.3",
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Flat sweep table; defaults to --json-out with a .csv suffix",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.shape:
        args.shape = list(QWEN36_HOT_SHAPES)
    if not args.provider_bin.exists():
        parser.error(f"provider benchmark not found: {args.provider_bin}")
    if args.oracle_bin is not None and not args.oracle_bin.exists():
        parser.error(f"llama oracle benchmark not found: {args.oracle_bin}")
    report = run_sweep(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_out = args.csv_out or args.json_out.with_suffix(".csv")
    write_csv_table(report, csv_out)
    print_summary(report)
    print(f"wrote {args.json_out}")
    print(f"wrote {csv_out}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
