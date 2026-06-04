#!/usr/bin/env python3
from __future__ import annotations

"""Compare CK hidden dumps against llama.cpp tensor dumps for a token."""

import argparse
import csv
from pathlib import Path

import numpy as np


def _compare(a_path: Path, b_path: Path, token: int | None = None) -> dict:
    a = np.fromfile(a_path, dtype=np.float32)
    b = np.fromfile(b_path, dtype=np.float32)
    if token is not None and a.size > 0 and b.size > a.size and b.size % a.size == 0:
        rows = int(b.size // a.size)
        if 0 <= int(token) < rows:
            start = int(token) * int(a.size)
            b = b[start : start + int(a.size)]
    n = min(int(a.size), int(b.size))
    if n <= 0:
        raise ValueError(f"empty vector input: {a_path} {b_path}")
    af = a[:n].astype(np.float64)
    bf = b[:n].astype(np.float64)
    diff = af - bf
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    cosine = float(np.dot(af, bf) / denom) if denom else float("nan")
    return {
        "a_size": int(a.size),
        "b_size": int(b.size),
        "compared": n,
        "cosine": cosine,
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "max_idx": int(np.argmax(np.abs(diff))),
    }


def _ck_name(token: int, layer: int, name: str) -> str:
    if layer < 0:
        return f"tok_{token:04d}_layer_-01_{name}.f32"
    return f"tok_{token:04d}_layer_{layer:03d}_{name}.f32"


def _llama_name(token: int, layer: int, name: str) -> str:
    if layer < 0:
        return f"{name}-token-{token:06d}-occ-000.bin"
    return f"{name}-{layer}-token-{token:06d}-occ-000.bin"


def _default_pairs(layer: int) -> list[tuple[str, str]]:
    return [
        ("after_attn", "attn_residual"),
        ("post_attn_norm", "attn_post_norm"),
        ("mlp_gate", "ffn_gate"),
        ("mlp_up", "ffn_up"),
        ("mlp_swiglu", "ffn_swiglu"),
        ("mlp_down", "ffn_out"),
        ("layer_out", "l_out"),
    ]


def _recurrent_pairs(layer: int) -> list[tuple[str, str]]:
    del layer
    return [
        ("linear_attn_qkv_mixed", "linear_attn_qkv_mixed"),
        ("q_conv", "q_conv"),
        ("k_conv", "k_conv"),
        ("q_conv_predelta", "q_conv_predelta"),
        ("k_conv_predelta", "k_conv_predelta"),
        ("v_conv_predelta", "v_conv_predelta"),
        ("attn_output", "attn_output"),
        ("final_output", "final_output"),
        ("linear_attn_out", "linear_attn_out"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ck-dir", required=True, type=Path)
    ap.add_argument("--llama-dir", required=True, type=Path)
    ap.add_argument("--token", required=True, type=int)
    ap.add_argument("--layers", default="0-23", help="comma/range list, e.g. 0-23 or 19,23")
    ap.add_argument(
        "--include-recurrent",
        action="store_true",
        help="Also compare recurrent/Qwen3.5 linear-attention intermediate labels when present.",
    )
    ap.add_argument("--csv-out", type=Path)
    args = ap.parse_args()

    layers: list[int] = []
    for part in args.layers.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            layers.extend(range(int(lo_s), int(hi_s) + 1))
        else:
            layers.append(int(part))

    rows: list[dict] = []
    for layer in layers:
        pairs = _default_pairs(layer)
        if args.include_recurrent:
            pairs = _recurrent_pairs(layer) + pairs
        for ck_label, ll_label in pairs:
            ck_path = args.ck_dir / _ck_name(args.token, layer, ck_label)
            ll_path = args.llama_dir / _llama_name(args.token, layer, ll_label)
            if not ck_path.exists() or not ll_path.exists():
                continue
            result = _compare(ck_path, ll_path, token=args.token)
            rows.append({"layer": layer, "ck": ck_label, "llama": ll_label, **result})

    final_ck = args.ck_dir / _ck_name(args.token, -1, "final_hidden")
    final_ll = args.llama_dir / _llama_name(args.token, -1, "result_norm")
    if final_ck.exists() and final_ll.exists():
        rows.append({"layer": -1, "ck": "final_hidden", "llama": "result_norm", **_compare(final_ck, final_ll, token=args.token)})

    fieldnames = [
        "layer",
        "ck",
        "llama",
        "a_size",
        "b_size",
        "compared",
        "cosine",
        "rmse",
        "mean_abs",
        "max_abs",
        "max_idx",
    ]
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row[k]) for k in fieldnames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
