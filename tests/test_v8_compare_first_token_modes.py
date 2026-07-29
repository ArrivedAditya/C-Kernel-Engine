#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "compare_first_token_logits_v8.py"
sys.path.insert(0, str(SCRIPT.parent))


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_first_token_modes", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare = _load_module()


def test_auto_llama_mode_mirrors_sequential_contract() -> None:
    assert (
        compare.resolve_llama_decode_mode("auto", "auto", "sequential_decode")
        == "sequential"
    )


def test_auto_llama_mode_mirrors_explicit_ck_mode() -> None:
    assert compare.resolve_llama_decode_mode("auto", "batched", "sequential_decode") == "batched"
    assert compare.resolve_llama_decode_mode("auto", "sequential", "batched") == "sequential"
    assert compare.resolve_llama_decode_mode("auto", "hybrid", "sequential_decode") == "batched"


def test_explicit_llama_mode_is_never_rewritten() -> None:
    assert compare.resolve_llama_decode_mode("batched", "sequential", "sequential_decode") == "batched"
    assert compare.resolve_llama_decode_mode("sequential", "batched", "batched") == "sequential"
