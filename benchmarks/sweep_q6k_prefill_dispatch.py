#!/usr/bin/env python3
"""Sweep Q6_K x Q8_K prefill dispatch choices for local hardware.

This is a performance characterization runner, not a correctness gate. It uses
the portable synthetic Q6_K/Q8_K benchmark to compare row-split dispatch against
raw 2D tile dispatch for representative model-family shapes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "bench_q6k_prefill_tile.py"

SHAPES: dict[str, tuple[int, int]] = {
    "gemma_mlp_down": (640, 2048),
    "qwen2_mlp_down": (896, 4864),
    "qwen35_mlp_down": (1024, 3584),
    "nanbeige_mlp_down": (2560, 10496),
    "large_q6": (2560, 10240),
}
CHECKSUM_ABS_TOL = 1e-3

METRIC_RE = re.compile(
    r"best_ms=([0-9.]+)\s+avg_ms=([0-9.]+).*?checksum=([-0-9.]+)",
    re.S,
)


def _parse_csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def _arg_expr(args: list[dict[str, Any]], name: str) -> int | None:
    for arg in args:
        if arg.get("name") != name:
            continue
        expr = str(arg.get("expr") or "").strip()
        if expr.isdigit():
            return int(expr)
    return None


def _shapes_from_lowered(path: Path, *, max_n: int, max_shapes: int) -> dict[str, tuple[int, int]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    operations = doc.get("operations") or []
    found: dict[str, tuple[int, int]] = {}
    for op in operations:
        if op.get("function") != "gemm_nt_q6_k_q8_k":
            continue
        args = op.get("args") or []
        if not isinstance(args, list):
            continue
        n = _arg_expr(args, "N")
        k = _arg_expr(args, "K")
        if n is None or k is None:
            continue
        if n > max_n:
            continue
        op_name = str(op.get("op") or "q6").replace(" ", "_")
        name = f"{op_name}_N{n}_K{k}"
        found.setdefault(name, (n, k))
        if len(found) >= max_shapes:
            break
    return found


def _run_one(
    *,
    shape_name: str,
    m: int,
    n: int,
    k: int,
    mode: str,
    threads: int,
    warmup: int,
    iters: int,
    force_2d: bool,
    engine_lib: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("CK_Q6K_Q8K_SIMD", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env["CK_NUM_THREADS"] = str(threads)
    if mode == "2d":
        env["CK_ENABLE_Q6K_Q8K_2D_PREFILL"] = "1"
        if force_2d:
            env["CK_FORCE_Q6K_Q8K_2D_PREFILL"] = "1"
    else:
        env.pop("CK_ENABLE_Q6K_Q8K_2D_PREFILL", None)
        env.pop("CK_FORCE_Q6K_Q8K_2D_PREFILL", None)

    cmd = [
        sys.executable,
        "-B",
        str(BENCH),
        "--mode",
        mode,
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--threads",
        str(threads),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--engine-lib",
        str(engine_lib),
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    match = METRIC_RE.search(proc.stdout)
    row: dict[str, Any] = {
        "shape": shape_name,
        "mode": mode,
        "M": m,
        "N": n,
        "K": k,
        "threads": threads,
        "engine_lib": str(engine_lib),
        "returncode": proc.returncode,
        "command": cmd,
    }
    if match:
        row.update(
            {
                "best_ms": float(match.group(1)),
                "avg_ms": float(match.group(2)),
                "checksum": float(match.group(3)),
                "status": "pass" if proc.returncode == 0 else "fail",
            }
        )
    else:
        row.update(
            {
                "status": "fail",
                "error_tail": "\n".join(proc.stdout.splitlines()[-20:]),
            }
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", choices=sorted(SHAPES), help="Shape id to sweep; repeatable")
    parser.add_argument("--from-lowered", type=Path, default=None, help="Extract Q6 prefill shapes from lowered_prefill_call.json")
    parser.add_argument("--m-values", default="16,32,64,128,256,512", help="Comma-separated token counts")
    parser.add_argument("--max-n", type=int, default=32768, help="Skip extracted shapes with very large N, such as full-vocab logits")
    parser.add_argument("--max-shapes", type=int, default=8, help="Maximum unique lowered shapes to sweep")
    parser.add_argument("--threads", type=int, default=int(os.getenv("CK_NUM_THREADS", "12")))
    parser.add_argument("--engine-lib", type=Path, default=ROOT / "build" / "libckernel_engine.so")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="Use a small sweep for CI/dev smoke")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.from_lowered is not None:
        shapes_map = _shapes_from_lowered(args.from_lowered, max_n=args.max_n, max_shapes=args.max_shapes)
        if not shapes_map:
            print(f"No Q6_K x Q8_K prefill shapes found in {args.from_lowered}")
    else:
        selected = args.shape or ["qwen2_mlp_down", "nanbeige_mlp_down"]
        shapes_map = {name: SHAPES[name] for name in selected}
    m_values = [16, 128, 512] if args.quick else _parse_csv_ints(args.m_values)

    results: list[dict[str, Any]] = []
    print(f"Q6_K x Q8_K prefill dispatch sweep threads={args.threads}")
    for shape_name, (n, k) in shapes_map.items():
        print(f"\nshape={shape_name} N={n} K={k}")
        for m in m_values:
            row = _run_one(
                shape_name=shape_name,
                m=m,
                n=n,
                k=k,
                mode="row",
                threads=args.threads,
                warmup=args.warmup,
                iters=args.iters,
                force_2d=False,
                engine_lib=args.engine_lib,
            )
            tiled = _run_one(
                shape_name=shape_name,
                m=m,
                n=n,
                k=k,
                mode="2d",
                threads=args.threads,
                warmup=args.warmup,
                iters=args.iters,
                force_2d=True,
                engine_lib=args.engine_lib,
            )
            results.extend([row, tiled])
            if row.get("status") == "pass" and tiled.get("status") == "pass":
                speedup = float(row["best_ms"]) / float(tiled["best_ms"])
                delta = (float(row["best_ms"]) - float(tiled["best_ms"])) / float(row["best_ms"]) * 100.0
                checksum_abs_diff = abs(float(row["checksum"]) - float(tiled["checksum"]))
                checksum_ok = checksum_abs_diff <= CHECKSUM_ABS_TOL
                row["paired_with"] = "2d"
                tiled["paired_with"] = "row"
                row["checksum_abs_diff"] = checksum_abs_diff
                tiled["checksum_abs_diff"] = checksum_abs_diff
                row["checksum_match"] = checksum_ok
                tiled["checksum_match"] = checksum_ok
                print(
                    f"M={m:4d} row={row['best_ms']:8.2f}ms "
                    f"2d={tiled['best_ms']:8.2f}ms speed={speedup:5.3f} "
                    f"delta={delta:+6.2f}% checksum={'ok' if checksum_ok else 'DIFF'} "
                    f"diff={checksum_abs_diff:.3e}"
                )
            else:
                print(f"M={m:4d} FAIL row={row.get('status')} 2d={tiled.get('status')}")

    report = {
        "kind": "q6k_q8k_prefill_dispatch_sweep",
        "threads": args.threads,
        "engine_lib": str(args.engine_lib),
        "warmup": args.warmup,
        "iters": args.iters,
        "source_lowered": None if args.from_lowered is None else str(args.from_lowered),
        "shapes": {name: {"N": n, "K": k} for name, (n, k) in shapes_map.items()},
        "results": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    return 0 if all(row.get("status") == "pass" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
