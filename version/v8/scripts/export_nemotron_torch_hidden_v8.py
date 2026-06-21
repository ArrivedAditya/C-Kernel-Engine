#!/usr/bin/env python3
from __future__ import annotations

"""Export PyTorch hidden/logit baselines for Nemotron-H safetensors.

The NVIDIA Nemotron-H remote model requires ``mamba-ssm`` at import time even
when running the pure PyTorch CPU Mamba2 path.  For CK parity bring-up we need a
CPU reference that can run on the same Xeon notebook without Triton/CUDA.  This
script installs a narrow CPU ``rmsnorm_fn`` shim, runs one forward pass, and
exports hidden/logit summaries or tensors for later CK comparison.
"""

import argparse
import contextlib
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _install_nemotron_cpu_shims() -> None:
    def rmsnorm_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        **_: Any,
    ) -> torch.Tensor:
        y = x.float()
        if z is not None and not norm_before_gate:
            y = y * torch.nn.functional.silu(z.float())

        if group_size is None or group_size <= 0 or group_size == y.shape[-1]:
            y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + eps)
        else:
            chunks = []
            for chunk in torch.split(y, int(group_size), dim=-1):
                chunks.append(chunk * torch.rsqrt(chunk.square().mean(dim=-1, keepdim=True) + eps))
            y = torch.cat(chunks, dim=-1)

        y = y * weight.float()
        if bias is not None:
            y = y + bias.float()
        if z is not None and norm_before_gate:
            y = y * torch.nn.functional.silu(z.float())
        return y.to(x.dtype)

    for name in (
        "mamba_ssm",
        "mamba_ssm.ops",
        "mamba_ssm.ops.triton",
        "mamba_ssm.ops.triton.layernorm_gated",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["mamba_ssm.ops.triton.layernorm_gated"].rmsnorm_fn = rmsnorm_fn

    class _NoStream(contextlib.AbstractContextManager):
        def __enter__(self):
            return None

        def __exit__(self, *exc: object) -> bool:
            return False

    # NemotronHBlock wraps all block work in torch.cuda.stream(...).  On CPU this
    # should be a no-op for parity export.
    torch.cuda.stream = lambda _stream=None: _NoStream()  # type: ignore[assignment]
    torch.cuda.default_stream = lambda _device=None: None  # type: ignore[assignment]


def _parse_tokens(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_hidden_indices(text: str, count: int) -> list[int]:
    if text == "all":
        return list(range(count))
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            idx += count
        if idx < 0 or idx >= count:
            raise SystemExit(f"hidden index {item} out of range for {count} hidden states")
        out.append(idx)
    return out


def _tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    x = t.float()
    return {
        "shape": [int(v) for v in t.shape],
        "mean": float(x.mean()),
        "rms": float(torch.sqrt(torch.mean(x * x))),
        "maxabs": float(x.abs().max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path, help="Nemotron-H safetensors directory")
    ap.add_argument("--prompt", default="Hello world.", help="Prompt text for tokenizer input")
    ap.add_argument("--tokens", help="Comma-separated token ids; overrides --prompt")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hidden-indices", default="0,1,2,5,10,20,30,40,-1")
    ap.add_argument("--dump-dir", type=Path, help="Optional directory for .npy hidden/logit dumps")
    ap.add_argument("--json-out", type=Path, help="Optional JSON output path")
    args = ap.parse_args()

    _install_nemotron_cpu_shims()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    load_t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.time() - load_t0

    token_ids = _parse_tokens(args.tokens)
    if token_ids is None:
        encoded = tok(args.prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        prompt_text = args.prompt
    else:
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        prompt_text = tok.decode(token_ids)

    fwd_t0 = time.time()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
    forward_seconds = time.time() - fwd_t0

    logits_last = out.logits[0, -1].float().contiguous()
    top = torch.topk(logits_last, int(args.top_k))
    hidden_count = len(out.hidden_states)
    hidden_indices = _parse_hidden_indices(args.hidden_indices, hidden_count)

    report: dict[str, Any] = {
        "model": str(args.model),
        "dtype": args.dtype,
        "prompt": prompt_text,
        "input_ids": [int(x) for x in input_ids[0].tolist()],
        "load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "logits_shape": [int(v) for v in out.logits.shape],
        "hidden_count": hidden_count,
        "top_logits": {
            "tokens": [int(x) for x in top.indices.tolist()],
            "values": [float(x) for x in top.values.tolist()],
            "texts": [tok.decode([int(x)]) for x in top.indices.tolist()],
        },
        "hidden": {},
    }

    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_dir / "input_ids.npy", input_ids.cpu().numpy().astype(np.int64, copy=False))
        np.save(args.dump_dir / "logits_last.npy", logits_last.cpu().numpy().astype(np.float32, copy=False))

    for idx in hidden_indices:
        h_last = out.hidden_states[idx][0, -1].float().contiguous()
        report["hidden"][str(idx)] = _tensor_stats(h_last)
        if args.dump_dir:
            np.save(args.dump_dir / f"hidden_{idx:03d}_last.npy", h_last.cpu().numpy().astype(np.float32, copy=False))

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
