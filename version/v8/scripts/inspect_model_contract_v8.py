#!/usr/bin/env python3
from __future__ import annotations

"""Inspect a HF-style model config and summarize the CK v8 bring-up contract.

This is intentionally config-first.  It does not download weights and it does
not pretend unsupported architectures are compatible.  The output is a small
JSON report that says which layer families and kernels CK can already lower,
and which kernels/templates must be added before safetensors->BUMP conversion
should be attempted.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SUPPORTED_DENSE_ARCHES = {"llama", "qwen2", "qwen3", "gemma3"}
SUPPORTED_HYBRID_ARCHES = {"qwen35", "gemma4"}


def _load_config(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "config.json"
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").replace("Infinity", "1e100")
    return json.loads(text)


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config") if isinstance(config.get("text_config"), dict) else config


def _infer_arch(config: dict[str, Any]) -> str:
    text = _text_config(config)
    model_type = str(text.get("model_type") or config.get("model_type") or "").lower()
    architectures = " ".join(str(x).lower() for x in config.get("architectures", []))
    if "qwen3_5" in model_type or "qwen3.5" in model_type or "qwen35" in model_type:
        return "qwen35"
    if "nemotron_h" in model_type or "nemotronh" in architectures:
        return "nemotron_h"
    if "cohere" in model_type or "cohere" in architectures or "command" in model_type:
        return "cohere"
    if "gemma" in model_type and "4" in model_type:
        return "gemma4"
    if "gemma" in model_type and "3" in model_type:
        return "gemma3"
    if "qwen3" in model_type:
        return "qwen3"
    if "qwen2" in model_type or model_type == "qwen":
        return "qwen2"
    if "llama" in model_type:
        return "llama"
    return model_type or "unknown"


def _parse_nemotron_h_pattern(pattern: str, num_layers: int) -> list[str]:
    """Parse Nemotron-H per-layer pattern characters into CK layer labels.

    NVIDIA's config code defines the pattern as one character per layer:
    M = Mamba2, * = attention, - = dense MLP, E = MoE.  The hyphen is a layer
    kind, not a separator.
    """
    mapping = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}
    chars = [ch for ch in str(pattern).strip() if not ch.isspace()]
    kinds = [mapping.get(ch, "unknown") for ch in chars]
    if num_layers > 0 and len(kinds) != num_layers:
        kinds.extend(["unspecified"] * max(0, num_layers - len(kinds)))
        kinds = kinds[:num_layers]
    return kinds


def _generic_dense_layers(config: dict[str, Any]) -> list[str]:
    text = _text_config(config)
    n = int(text.get("num_hidden_layers") or 0)
    return ["attention"] * n


def _qwen35_layers(config: dict[str, Any]) -> list[str]:
    text = _text_config(config)
    layer_types = [str(x) for x in text.get("layer_types", [])]
    if layer_types:
        return ["attention" if x == "full_attention" else "deltanet" for x in layer_types]
    n = int(text.get("num_hidden_layers") or 0)
    interval = int(text.get("full_attention_interval") or 4)
    return ["attention" if (i + 1) % interval == 0 else "deltanet" for i in range(n)]


def _gemma4_layers(config: dict[str, Any]) -> list[str]:
    text = _text_config(config)
    layer_kinds = text.get("layer_kinds") or config.get("layer_kinds")
    if isinstance(layer_kinds, list):
        return [str(x) for x in layer_kinds]
    n = int(text.get("num_hidden_layers") or 0)
    return ["attention_or_sliding_attention"] * n


def _layer_kinds(arch: str, config: dict[str, Any]) -> list[str]:
    text = _text_config(config)
    if arch == "nemotron_h":
        return _parse_nemotron_h_pattern(str(text.get("hybrid_override_pattern") or ""), int(text.get("num_hidden_layers") or 0))
    if arch == "qwen35":
        return _qwen35_layers(config)
    if arch == "gemma4":
        return _gemma4_layers(config)
    return _generic_dense_layers(config)


def _required_ops(arch: str, config: dict[str, Any], layer_kinds: list[str]) -> list[str]:
    ops = {"embedding", "rmsnorm", "matmul", "residual_add", "logits"}
    if any(kind in {"attention", "full_attention", "sliding_attention", "attention_or_sliding_attention"} for kind in layer_kinds):
        ops.update({"attention", "rope"})
    if any(kind == "deltanet" for kind in layer_kinds):
        ops.update({
            "recurrent_dt_gate",
            "recurrent_conv_state_update",
            "ssm_conv1d",
            "recurrent_qk_l2_norm",
            "gated_deltanet",
            "recurrent_norm_gate",
        })
    if any(kind == "mamba" for kind in layer_kinds):
        ops.update({
            "mamba_in_proj_split",
            "mamba_conv1d_state_update",
            "mamba_dt_softplus",
            "mamba_selective_scan",
            "mamba_rmsnorm_gate",
            "mamba_out_proj",
        })
    if any(kind == "moe" for kind in layer_kinds):
        ops.update({
            "topk_router",
            "topk_softmax",
            "moe_expert_dispatch",
            "moe_expert_combine",
            "shared_expert_mlp",
        })
    act = str(_text_config(config).get("mlp_hidden_act") or _text_config(config).get("hidden_act") or "").lower()
    if act == "relu2":
        ops.add("relu2_mlp")
    else:
        ops.add("swiglu_or_geglu")
    if arch == "gemma4":
        ops.update({"gemma4_per_layer_embed", "gemma4_final_logit_softcap"})
    return sorted(ops)


def _missing_ops(arch: str, required_ops: list[str]) -> list[str]:
    missing = []
    for op in required_ops:
        if op.startswith("mamba_") or op == "relu2_mlp":
            missing.append(op)
        if op in {"topk_router", "topk_softmax", "moe_expert_dispatch", "moe_expert_combine", "shared_expert_mlp"}:
            missing.append(op)
    if arch == "cohere":
        missing.append("cohere_tensor_name_mapping_audit")
    if arch not in SUPPORTED_DENSE_ARCHES | SUPPORTED_HYBRID_ARCHES:
        missing.append("v8_template_contract")
        missing.append("safetensors_to_bump_mapping")
    return sorted(set(missing))


def inspect_config(config: dict[str, Any]) -> dict[str, Any]:
    text = _text_config(config)
    arch = _infer_arch(config)
    layer_kinds = _layer_kinds(arch, config)
    required_ops = _required_ops(arch, config, layer_kinds)
    missing_ops = _missing_ops(arch, required_ops)
    status = "supported" if not missing_ops else "bringup_required"
    return {
        "arch": arch,
        "model_type": text.get("model_type") or config.get("model_type"),
        "architectures": config.get("architectures", []),
        "status": status,
        "num_layers": int(text.get("num_hidden_layers") or len(layer_kinds) or 0),
        "hidden_size": int(text.get("hidden_size") or 0),
        "intermediate_size": int(text.get("intermediate_size") or 0),
        "num_attention_heads": int(text.get("num_attention_heads") or 0),
        "num_key_value_heads": int(text.get("num_key_value_heads") or 0),
        "layer_kind_counts": dict(sorted(Counter(layer_kinds).items())),
        "required_ops": required_ops,
        "missing_ops": missing_ops,
        "notes": _notes(arch, config, layer_kinds, missing_ops),
    }


def _notes(arch: str, config: dict[str, Any], layer_kinds: list[str], missing_ops: list[str]) -> list[str]:
    text = _text_config(config)
    notes: list[str] = []
    if arch == "nemotron_h":
        pattern_chars = [ch for ch in str(text.get("hybrid_override_pattern") or "").strip() if not ch.isspace()]
        if int(text.get("num_hidden_layers") or 0) and len(pattern_chars) != int(text.get("num_hidden_layers") or 0):
            notes.append(
                f"hybrid_override_pattern describes {len(pattern_chars)} layers but num_hidden_layers={text.get('num_hidden_layers')}; "
                "confirm whether the pattern repeats or whether checkpoint code expands it."
            )
        notes.append("Nemotron-H is a hybrid Mamba/attention decoder; CK DeltaNet kernels are not a substitute for Mamba selective scan.")
        notes.append(f"hybrid_override_pattern={text.get('hybrid_override_pattern')}")
        notes.append(f"mamba heads={text.get('mamba_num_heads')} head_dim={text.get('mamba_head_dim')} state={text.get('ssm_state_size')} conv_kernel={text.get('conv_kernel')}")
        notes.append(f"MLP activation={text.get('mlp_hidden_act')}; relu2 needs an explicit forward/backward contract.")
    if arch == "cohere":
        notes.append("Cohere Command configs are gated in this environment; require config/weight access before mapping tensor names.")
        notes.append("Likely first target is dense decoder mapping/audit before any new math kernels.")
    if not missing_ops:
        notes.append("Config-level contract has no known missing op, but weight-name audit and hidden parity are still required.")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect a model config and report CK v8 compatibility contract")
    ap.add_argument("config", type=Path, help="config.json or model directory")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    report = inspect_config(_load_config(args.config))
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "supported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
