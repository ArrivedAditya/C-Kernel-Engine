#!/usr/bin/env python3
from __future__ import annotations

"""
Export llama.cpp final hidden embedding for an explicit token prefix.

This uses the same token-replay helper as the logits parity runner and writes
the final output embedding from llama.cpp embeddings mode. It does not rely on
tokenizer text reconstruction.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from compare_first_token_logits_v8 import (  # type: ignore
    ROOT,
    discover_gguf,
    ensure_llama_helper,
    parse_tokens_csv,
)


def export_llama_hidden(
    *,
    gguf_path: Path,
    tokens: list[int],
    tokens_before: list[int] | None = None,
    tokens_after: list[int] | None = None,
    ctx_len: int,
    threads: int,
    decode_mode: str,
    prefix_decode_mode: str,
    out_path: Path,
) -> dict:
    helper = ensure_llama_helper()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llama_hidden_probe_") as td:
        logits_path = Path(td) / "llama_logits.f32"
        cmd = [
            str(helper),
            "--model",
            str(gguf_path),
            "--ctx",
            str(int(ctx_len)),
            "--top-k",
            "16",
            "--logits-out",
            str(logits_path),
            "--embeddings-out",
            str(out_path),
            "--decode-mode",
            decode_mode,
        ]
        before = list(tokens_before or [])
        after = list(tokens_after or [])
        if before or after:
            cmd.extend(["--prefix-decode-mode", prefix_decode_mode])
            if before:
                cmd.extend(["--tokens-before", ",".join(str(t) for t in before)])
            if after:
                cmd.extend(["--tokens-after", ",".join(str(t) for t in after)])
        else:
            cmd.extend(["--tokens", ",".join(str(t) for t in tokens)])
        if threads > 0:
            cmd.extend(["--threads", str(int(threads))])
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                "llama hidden export failed\n"
                f"cmd: {' '.join(cmd)}\n"
                f"rc: {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )
        meta = json.loads(proc.stdout.strip())
        if not isinstance(meta, dict) or not meta.get("ok"):
            raise RuntimeError(f"invalid llama helper output: {proc.stdout.strip()}")
        return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Export llama.cpp final hidden embedding for explicit token IDs")
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--tokens", required=True, help="comma-separated token IDs")
    ap.add_argument("--tokens-before", default=None, help="comma-separated prompt/prefix token IDs")
    ap.add_argument("--tokens-after", default=None, help="comma-separated continuation token IDs")
    ap.add_argument("--ctx-len", type=int, default=1034)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--decode-mode", choices=("batched", "sequential"), default="batched")
    ap.add_argument("--prefix-decode-mode", choices=("batched", "sequential"), default="batched")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    gguf_path = discover_gguf(args.gguf, args.gguf.parent)
    tokens_before = parse_tokens_csv(args.tokens_before) if args.tokens_before else None
    tokens_after = parse_tokens_csv(args.tokens_after) if args.tokens_after else None
    meta = export_llama_hidden(
        gguf_path=gguf_path,
        tokens=parse_tokens_csv(args.tokens),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        ctx_len=int(args.ctx_len),
        threads=int(args.threads),
        decode_mode=args.decode_mode,
        prefix_decode_mode=args.prefix_decode_mode,
        out_path=args.out,
    )
    print(f"exported={args.out} n_embd_out={meta.get('n_embd_out')} token_count={meta.get('token_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
