#!/usr/bin/env python3
from __future__ import annotations

"""
Compare Qwen3.5 safetensors math against CK hidden dumps.

This is a diagnostic runner, not a full PyTorch model wrapper.  It reconstructs
the Qwen3.5 text recurrent path directly from safetensors and compares each
token/layer intermediate against CK_DEBUG_EXPORT_HIDDEN files.  The goal is to
find the first stateful divergence after single-token parity already passes.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


def _load_config(checkpoint: Path) -> dict[str, Any]:
    cfg = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    return cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg


def _load_tensors(checkpoint: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for path in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key).to(torch.float32)
    if not tensors:
        raise FileNotFoundError(f"no safetensors files found under {checkpoint}")
    return tensors


def _prefix(tensors: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("model.language_model.") for k in tensors):
        return "model.language_model."
    if any(k.startswith("model.") for k in tensors):
        return "model."
    return ""


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mv(weight, x)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)


def _l2_norm_heads(x: torch.Tensor, head_dim: int, eps: float) -> torch.Tensor:
    y = x.reshape(-1, head_dim)
    y = y * torch.rsqrt(torch.sum(y * y, dim=1, keepdim=True) + eps)
    return y.reshape_as(x)


def _recurrent_norm_gate(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, num_heads: int, head_dim: int, eps: float) -> torch.Tensor:
    xh = x.reshape(num_heads, head_dim)
    gh = gate.reshape(num_heads, head_dim)
    ms = torch.mean(xh * xh, dim=1, keepdim=True)
    return (xh * torch.rsqrt(ms + eps) * weight.reshape(1, head_dim) * _silu(gh)).reshape(-1)



def _mrope_yarn_corr_dim(n_dims: int, n_ctx_orig: int, n_rot: float, base: float) -> float:
    return n_dims * math.log(float(n_ctx_orig) / (n_rot * 2.0 * math.pi)) / (2.0 * math.log(base))


def _mrope_yarn_corr_dims(n_dims: int, n_ctx_orig: int, freq_base: float, beta_fast: float, beta_slow: float) -> tuple[float, float]:
    start = math.floor(_mrope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base))
    end = math.ceil(_mrope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base))
    return max(0.0, float(start)), min(float(n_dims - 1), float(end))


def _mrope_yarn(theta_extrap: float, freq_scale: float, corr_dims: tuple[float, float], chan: int, ext_factor: float, attn_factor: float) -> tuple[float, float]:
    theta_interp = freq_scale * theta_extrap
    theta = theta_interp
    mscale = attn_factor
    if ext_factor != 0.0:
        y = (float(chan) - corr_dims[0]) / max(0.001, corr_dims[1] - corr_dims[0])
        ramp_mix = (1.0 - min(1.0, max(0.0, y))) * ext_factor
        theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix
        mscale *= 1.0 + 0.1 * math.log(1.0 / max(freq_scale, 1e-6))
    return math.cos(theta) * mscale, math.sin(theta) * mscale


def _mrope_text_one(x: torch.Tensor, position: int, head_dim: int, n_dims: int, sections: list[int], freq_base: float) -> torch.Tensor:
    y = x.clone().reshape(-1, head_dim)
    if n_dims * 2 > head_dim:
        n_dims = head_dim // 2
    sec_w = sections[0] + sections[1]
    sec_e = sec_w + sections[2]
    sect_dims = sum(sections)
    theta_scale = freq_base ** (-2.0 / float(n_dims))
    corr_dims = _mrope_yarn_corr_dims(n_dims, 32768, freq_base, 32.0, 1.0)
    for row in range(y.shape[0]):
        theta_t = float(position)
        theta_h = float(position)
        theta_w = float(position)
        theta_e = 0.0
        for chan in range(n_dims):
            sector = chan % sect_dims if sect_dims > 0 else chan
            theta = theta_t
            if sector >= sections[0] and sector < sec_w:
                theta = theta_h
            elif sector >= sec_w and sector < sec_e:
                theta = theta_w
            elif sector >= sec_e:
                theta = theta_e
            c, si = _mrope_yarn(theta, 1.0, corr_dims, chan, 0.0, 1.0)
            x0 = float(y[row, chan])
            x1 = float(y[row, chan + n_dims])
            y[row, chan] = x0 * c - x1 * si
            y[row, chan + n_dims] = x0 * si + x1 * c
            theta_t *= theta_scale
            theta_h *= theta_scale
            theta_w *= theta_scale
            theta_e *= theta_scale
    return y.reshape(-1)

def _stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.reshape(-1).to(torch.float32)
    b = b.reshape(-1).to(torch.float32)
    n = min(a.numel(), b.numel())
    if n == 0:
        return {"max": math.inf, "mean": math.inf, "cos": 0.0}
    a = a[:n]
    b = b[:n]
    d = torch.abs(a - b)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    cos = float(torch.dot(a, b) / denom) if float(denom) > 0.0 else 0.0
    return {"max": float(torch.max(d)), "mean": float(torch.mean(d)), "cos": cos}


def _read_ck(hidden_dir: Path, tok: int, layer: int, label: str) -> torch.Tensor | None:
    path = hidden_dir / f"tok_{tok:04d}_layer_{layer:03d}_{label}.f32"
    if not path.exists():
        return None
    arr = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(arr.copy())


def _compare(
    hidden_dir: Path,
    tok: int,
    layer: int,
    label: str,
    value: torch.Tensor,
    rows: list[dict[str, Any]],
    fail_max: float,
    fail_cos: float,
) -> bool:
    ck = _read_ck(hidden_dir, tok, layer, label)
    if ck is None:
        return False
    st = _stats(value.reshape(-1), ck.reshape(-1))
    row = {
        "token_index": tok,
        "layer": layer,
        "label": label,
        "shape": list(value.shape),
        "ck_values": int(ck.numel()),
        **st,
    }
    rows.append(row)
    return st["max"] > fail_max or st["cos"] < fail_cos


class Qwen35RecurrentComparator:
    def __init__(self, checkpoint: Path, hidden_dir: Path, tokens: list[int], fail_max: float, fail_cos: float) -> None:
        self.checkpoint = checkpoint
        self.hidden_dir = hidden_dir
        self.tokens = tokens
        self.fail_max = fail_max
        self.fail_cos = fail_cos
        self.cfg = _load_config(checkpoint)
        self.tensors = _load_tensors(checkpoint)
        self.pfx = _prefix(self.tensors)
        self.eps = float(self.cfg.get("rms_norm_eps", 1e-6))
        self.hidden_size = int(self.cfg["hidden_size"])
        self.layer_types = [str(x) for x in self.cfg.get("layer_types", [])]
        self.num_layers = int(self.cfg["num_hidden_layers"])
        self.conv_kernel = int(self.cfg.get("linear_conv_kernel_dim", 4))
        self.head_dim = int(self.cfg.get("linear_key_head_dim", self.cfg.get("head_dim", 128)))
        self.num_heads = int(self.cfg.get("linear_num_key_heads", 1))
        self.inner = int(self.cfg.get("linear_num_value_heads", self.num_heads)) * int(self.cfg.get("linear_value_head_dim", self.head_dim))
        self.conv_channels = self.num_heads * self.head_dim * 2 + self.inner
        self.conv_state = {
            layer: torch.zeros(self.conv_channels, self.conv_kernel - 1, dtype=torch.float32)
            for layer, kind in enumerate(self.layer_types)
            if kind != "full_attention"
        }
        self.ssm_state = {
            layer: torch.zeros(self.num_heads, self.head_dim, self.head_dim, dtype=torch.float32)
            for layer, kind in enumerate(self.layer_types)
            if kind != "full_attention"
        }
        self.attn_k_cache: dict[int, list[torch.Tensor]] = {layer: [] for layer, kind in enumerate(self.layer_types) if kind == "full_attention"}
        self.attn_v_cache: dict[int, list[torch.Tensor]] = {layer: [] for layer, kind in enumerate(self.layer_types) if kind == "full_attention"}

    def w(self, name: str) -> torch.Tensor:
        key = self.pfx + name
        if key not in self.tensors:
            raise KeyError(key)
        value = self.tensors[key]
        # Match llama.cpp/Qwen3.5 conversion: HF stores most norm weights
        # offset by -1. The recurrent linear_attn.norm.weight is not shifted.
        if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            return value + 1.0
        return value

    def _layer_prefix(self, layer: int) -> str:
        return f"layers.{layer}."

    def _recurrent_layer(self, x: torch.Tensor, token_index: int, layer: int, rows: list[dict[str, Any]]) -> tuple[torch.Tensor, bool]:
        lp = self._layer_prefix(layer)
        norm = _rmsnorm(x, self.w(lp + "input_layernorm.weight"), self.eps)

        qkv = _linear(norm, self.w(lp + "linear_attn.in_proj_qkv.weight"))
        if _compare(self.hidden_dir, token_index, layer, "linear_attn_qkv_mixed", qkv, rows, self.fail_max, self.fail_cos):
            return x, True

        z = _linear(norm, self.w(lp + "linear_attn.in_proj_z.weight"))
        if _compare(self.hidden_dir, token_index, layer, "z", z, rows, self.fail_max, self.fail_cos):
            return x, True

        alpha = _linear(norm, self.w(lp + "linear_attn.in_proj_a.weight"))
        if _compare(self.hidden_dir, token_index, layer, "alpha", alpha, rows, self.fail_max, self.fail_cos):
            return x, True

        beta = _linear(norm, self.w(lp + "linear_attn.in_proj_b.weight"))
        if _compare(self.hidden_dir, token_index, layer, "beta", beta, rows, self.fail_max, self.fail_cos):
            return x, True

        conv_x = torch.cat([self.conv_state[layer], qkv.reshape(self.conv_channels, 1)], dim=1)
        if _compare(self.hidden_dir, token_index, layer, "conv_input", conv_x, rows, self.fail_max, self.fail_cos):
            return x, True

        kernel = self.w(lp + "linear_attn.conv1d.weight").reshape(self.conv_channels, self.conv_kernel)
        conv_raw = torch.sum(conv_x * kernel, dim=1)
        if _compare(self.hidden_dir, token_index, layer, "conv_output_raw", conv_raw, rows, self.fail_max, self.fail_cos):
            return x, True

        conv = _silu(conv_raw)
        if _compare(self.hidden_dir, token_index, layer, "conv_output_silu", conv, rows, self.fail_max, self.fail_cos):
            return x, True

        q = conv[: self.num_heads * self.head_dim].clone()
        k0 = self.num_heads * self.head_dim
        k = conv[k0 : k0 + self.num_heads * self.head_dim].clone()
        v = conv[k0 + self.num_heads * self.head_dim :].clone()
        if _compare(self.hidden_dir, token_index, layer, "q_conv", q, rows, self.fail_max, self.fail_cos):
            return x, True
        if _compare(self.hidden_dir, token_index, layer, "k_conv", k, rows, self.fail_max, self.fail_cos):
            return x, True
        if _compare(self.hidden_dir, token_index, layer, "v_conv_predelta", v, rows, self.fail_max, self.fail_cos):
            return x, True

        q = _l2_norm_heads(q, self.head_dim, self.eps)
        k = _l2_norm_heads(k, self.head_dim, self.eps)
        if _compare(self.hidden_dir, token_index, layer, "q_conv_predelta", q, rows, self.fail_max, self.fail_cos):
            return x, True
        if _compare(self.hidden_dir, token_index, layer, "k_conv_predelta", k, rows, self.fail_max, self.fail_cos):
            return x, True

        ssm_a = -torch.exp(self.w(lp + "linear_attn.A_log")).reshape(self.num_heads)
        g = _softplus(alpha + self.w(lp + "linear_attn.dt_bias")) * ssm_a
        decay = torch.exp(g)
        beta_s = torch.sigmoid(beta)
        qh = q.reshape(self.num_heads, self.head_dim) / math.sqrt(float(self.head_dim))
        kh = k.reshape(self.num_heads, self.head_dim)
        vh = v.reshape(self.num_heads, self.head_dim)
        state_prev = self.ssm_state[layer]
        state_cur = torch.empty_like(state_prev)
        attn = torch.empty_like(vh)
        for h in range(self.num_heads):
            s = state_prev[h] * decay[h]
            kv_mem = torch.mv(s.T, kh[h])
            delta = (vh[h] - kv_mem) * beta_s[h]
            s = s + torch.outer(kh[h], delta)
            state_cur[h] = s
            attn[h] = torch.mv(s.T, qh[h])
        attn_flat = attn.reshape(-1)
        if _compare(self.hidden_dir, token_index, layer, "attn_output", attn_flat, rows, self.fail_max, self.fail_cos):
            return x, True
        if _compare(self.hidden_dir, token_index, layer, "new_state", state_cur, rows, self.fail_max, self.fail_cos):
            return x, True

        final = _recurrent_norm_gate(attn_flat, z, self.w(lp + "linear_attn.norm.weight"), self.num_heads, self.head_dim, self.eps)
        if _compare(self.hidden_dir, token_index, layer, "final_output", final, rows, self.fail_max, self.fail_cos):
            return x, True

        attn_out = _linear(final, self.w(lp + "linear_attn.out_proj.weight"))
        if _compare(self.hidden_dir, token_index, layer, "linear_attn_out", attn_out, rows, self.fail_max, self.fail_cos):
            return x, True
        x = x + attn_out
        if _compare(self.hidden_dir, token_index, layer, "after_attn", x, rows, self.fail_max, self.fail_cos):
            return x, True

        post = _rmsnorm(x, self.w(lp + "post_attention_layernorm.weight"), self.eps)
        if _compare(self.hidden_dir, token_index, layer, "post_attn_norm", post, rows, self.fail_max, self.fail_cos):
            return x, True
        gate = _linear(post, self.w(lp + "mlp.gate_proj.weight"))
        up = _linear(post, self.w(lp + "mlp.up_proj.weight"))
        swiglu = _silu(gate) * up
        if _compare(self.hidden_dir, token_index, layer, "mlp_swiglu", swiglu, rows, self.fail_max, self.fail_cos):
            return x, True
        down = _linear(swiglu, self.w(lp + "mlp.down_proj.weight"))
        if _compare(self.hidden_dir, token_index, layer, "mlp_down", down, rows, self.fail_max, self.fail_cos):
            return x, True
        x = x + down
        if _compare(self.hidden_dir, token_index, layer, "layer_out", x, rows, self.fail_max, self.fail_cos):
            return x, True

        self.conv_state[layer] = conv_x[:, 1:].clone()
        self.ssm_state[layer] = state_cur.clone()
        return x, False


    def _full_attention_prefix(self, x: torch.Tensor, token_index: int, layer: int, rows: list[dict[str, Any]]) -> tuple[torch.Tensor, bool]:
        lp = self._layer_prefix(layer)
        num_heads = int(self.cfg.get("num_attention_heads", 1))
        num_kv_heads = int(self.cfg.get("num_key_value_heads", 1))
        head_dim = int(self.cfg.get("head_dim", self.hidden_size // max(1, num_heads)))
        norm = _rmsnorm(x, self.w(lp + "input_layernorm.weight"), self.eps)

        q_gate = _linear(norm, self.w(lp + "self_attn.q_proj.weight"))
        if _compare(self.hidden_dir, token_index, layer, "q_proj", q_gate, rows, self.fail_max, self.fail_cos):
            return x, True
        qg = q_gate.reshape(num_heads, 2, head_dim)
        q = qg[:, 0, :].reshape(-1).clone()

        k = _linear(norm, self.w(lp + "self_attn.k_proj.weight"))
        if _compare(self.hidden_dir, token_index, layer, "k_proj", k, rows, self.fail_max, self.fail_cos):
            return x, True
        v = _linear(norm, self.w(lp + "self_attn.v_proj.weight"))
        if _compare(self.hidden_dir, token_index, layer, "v_proj", v, rows, self.fail_max, self.fail_cos):
            return x, True

        q = _rmsnorm(q.reshape(num_heads, head_dim), self.w(lp + "self_attn.q_norm.weight"), self.eps).reshape(-1)
        k = _rmsnorm(k.reshape(num_kv_heads, head_dim), self.w(lp + "self_attn.k_norm.weight"), self.eps).reshape(-1)
        if _compare(self.hidden_dir, token_index, layer, "qk_norm_q", q, rows, self.fail_max, self.fail_cos):
            return x, True
        if _compare(self.hidden_dir, token_index, layer, "qk_norm_k", k, rows, self.fail_max, self.fail_cos):
            return x, True

        rope_params = self.cfg.get("rope_parameters") if isinstance(self.cfg.get("rope_parameters"), dict) else {}
        sections = [int(v) for v in (rope_params.get("mrope_section") or [11, 11, 10])]
        if len(sections) == 3:
            sections.append(0)
        n_dims = int(sum(sections))
        freq_base = float(rope_params.get("rope_theta", self.cfg.get("rope_theta", 10000000.0)))
        q_rope = _mrope_text_one(q, token_index, head_dim, n_dims, sections, freq_base)
        k_rope = _mrope_text_one(k, token_index, head_dim, n_dims, sections, freq_base)
        if _compare(self.hidden_dir, token_index, layer, "rope_q", q_rope, rows, self.fail_max, self.fail_cos):
            return x, True
        if _compare(self.hidden_dir, token_index, layer, "rope_k", k_rope, rows, self.fail_max, self.fail_cos):
            return x, True

        qh = q_rope.reshape(num_heads, head_dim)
        kh = k_rope.reshape(num_kv_heads, head_dim)
        vh = v.reshape(num_kv_heads, head_dim)
        self.attn_k_cache[layer].append(kh.clone())
        self.attn_v_cache[layer].append(vh.clone())
        k_cache = torch.stack(self.attn_k_cache[layer], dim=0)
        v_cache = torch.stack(self.attn_v_cache[layer], dim=0)
        group = max(1, num_heads // max(1, num_kv_heads))
        attn = torch.empty((num_heads, head_dim), dtype=torch.float32)
        scale = 1.0 / math.sqrt(float(head_dim))
        for h in range(num_heads):
            kv_h = min(num_kv_heads - 1, h // group)
            scores = torch.mv(k_cache[:, kv_h, :], qh[h]) * scale
            probs = torch.softmax(scores, dim=0)
            attn[h] = torch.sum(probs.reshape(-1, 1) * v_cache[:, kv_h, :], dim=0)
        attn_flat = (attn.reshape(-1) * torch.sigmoid(qg[:, 1, :].reshape(-1))).reshape(-1)
        if _compare(self.hidden_dir, token_index, layer, "attn_out", attn_flat, rows, self.fail_max, self.fail_cos):
            return x, True

        out = _linear(attn_flat, self.w(lp + "self_attn.o_proj.weight"))
        if _compare(self.hidden_dir, token_index, layer, "out_proj", out, rows, self.fail_max, self.fail_cos):
            return x, True
        x = x + out
        if _compare(self.hidden_dir, token_index, layer, "after_attn", x, rows, self.fail_max, self.fail_cos):
            return x, True

        post = _rmsnorm(x, self.w(lp + "post_attention_layernorm.weight"), self.eps)
        if _compare(self.hidden_dir, token_index, layer, "post_attn_norm", post, rows, self.fail_max, self.fail_cos):
            return x, True
        gate_proj = _linear(post, self.w(lp + "mlp.gate_proj.weight"))
        up_proj = _linear(post, self.w(lp + "mlp.up_proj.weight"))
        swiglu = _silu(gate_proj) * up_proj
        if _compare(self.hidden_dir, token_index, layer, "mlp_swiglu", swiglu, rows, self.fail_max, self.fail_cos):
            return x, True
        down = _linear(swiglu, self.w(lp + "mlp.down_proj.weight"))
        if _compare(self.hidden_dir, token_index, layer, "mlp_down", down, rows, self.fail_max, self.fail_cos):
            return x, True
        x = x + down
        if _compare(self.hidden_dir, token_index, layer, "layer_out", x, rows, self.fail_max, self.fail_cos):
            return x, True
        return x, False

    def run(self, stop_at_full_attention: bool, max_layer: int | None = None) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        token_emb = self.w("embed_tokens.weight")
        first_failure: dict[str, Any] | None = None
        for tok_i, tok in enumerate(self.tokens):
            x = token_emb[int(tok)].clone()
            layer_limit = self.num_layers if max_layer is None else min(self.num_layers, int(max_layer))
            for layer in range(layer_limit):
                kind = self.layer_types[layer] if layer < len(self.layer_types) else "linear_attention"
                if kind == "full_attention":
                    x, failed = self._full_attention_prefix(x, tok_i, layer, rows)
                    if failed:
                        first_failure = rows[-1]
                        return {"first_failure": first_failure, "comparisons": rows}
                    continue
                x, failed = self._recurrent_layer(x, tok_i, layer, rows)
                if failed:
                    first_failure = rows[-1]
                    return {"first_failure": first_failure, "comparisons": rows}
        return {"first_failure": first_failure, "comparisons": rows}


def _parse_tokens_csv(value: str) -> list[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("token list is empty")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Qwen3.5 safetensors recurrent hidden stream against CK dumps")
    ap.add_argument("--checkpoint", required=True, type=Path, help="HF safetensors checkpoint directory")
    ap.add_argument("--ck-hidden-dir", required=True, type=Path, help="CK_DEBUG_EXPORT_HIDDEN output directory")
    ap.add_argument("--tokens", required=True, type=_parse_tokens_csv, help="comma-separated token IDs matching the CK dump")
    ap.add_argument("--fail-max", type=float, default=5e-3)
    ap.add_argument("--fail-cos", type=float, default=0.999)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--stop-at-full-attention", action="store_true", default=True)
    ap.add_argument("--max-layer", type=int, help="compare only layers [0, max_layer) so recurrent state can be isolated before full attention")
    args = ap.parse_args()

    comp = Qwen35RecurrentComparator(
        checkpoint=args.checkpoint,
        hidden_dir=args.ck_hidden_dir,
        tokens=args.tokens,
        fail_max=args.fail_max,
        fail_cos=args.fail_cos,
    )
    result = comp.run(stop_at_full_attention=args.stop_at_full_attention, max_layer=args.max_layer)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(result["first_failure"], indent=2, sort_keys=True))
    print(f"comparisons={len(result['comparisons'])}")
    return 1 if result["first_failure"] and result["first_failure"].get("label") != "full_attention_not_implemented" else 0


if __name__ == "__main__":
    raise SystemExit(main())
