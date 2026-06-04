#!/usr/bin/env python3
from __future__ import annotations

"""Compare two raw float32 hidden/vector dumps."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare_vectors(a_path: Path, b_path: Path) -> dict:
    a = np.fromfile(a_path, dtype=np.float32)
    b = np.fromfile(b_path, dtype=np.float32)
    n = min(int(a.size), int(b.size))
    if n <= 0:
        raise ValueError("empty vector input")
    af = a[:n].astype(np.float64)
    bf = b[:n].astype(np.float64)
    diff = af - bf
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    cosine = float(np.dot(af, bf) / denom) if denom else float("nan")
    max_idx = int(np.argmax(np.abs(diff)))
    return {
        "a": str(a_path),
        "b": str(b_path),
        "a_size": int(a.size),
        "b_size": int(b.size),
        "compared": n,
        "cosine": cosine,
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "max_idx": max_idx,
        "a_at_max": float(af[max_idx]),
        "b_at_max": float(bf[max_idx]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare raw float32 hidden/vector dumps")
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args()

    result = compare_vectors(args.a, args.b)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"sizes={result['a_size']}/{result['b_size']} "
            f"cosine={result['cosine']:.9f} "
            f"rmse={result['rmse']:.9f} "
            f"mean_abs={result['mean_abs']:.9f} "
            f"max_abs={result['max_abs']:.9f} "
            f"max_idx={result['max_idx']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
