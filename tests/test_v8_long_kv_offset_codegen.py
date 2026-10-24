"""Prevent signed-int overflow in long-context generated KV addresses."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))

from codegen_core_v8 import _normalize_arg_expr, emit_memory_layout  # type: ignore
from codegen_prefill_v8 import emit_prefill_op  # type: ignore


def _compile(source: str) -> subprocess.CompletedProcess[str]:
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    return subprocess.run(
        [compiler, "-std=c11", "-Wall", "-Werror=overflow", "-fsyntax-only", "-x", "c", "-"],
        input=source, text=True, capture_output=True, check=False,
    )


def test_long_prefill_kv_argument_keeps_wide_offset() -> None:
    old_k = "(float*)((model->kv_cache + (26*2)*NUM_KV_HEADS*MAX_SEQ_LEN*128))"
    old_v = "(float*)((model->kv_cache + (26*2+1)*NUM_KV_HEADS*MAX_SEQ_LEN*128))"
    operation = {
        "function": "probe_store", "op": "mla_kv_cache_batch_store", "layer": 26,
        "args": [
            {"name": "k_cache", "source": "state:kv", "expr": old_k},
            {"name": "v_cache", "source": "state:kv", "expr": old_v},
        ],
    }
    emitted = emit_prefill_op(operation, 0, {"embed_dim": 2048, "head_dim": 128})
    assert "1ULL*(26*2)" in emitted
    assert "1ULL*(26*2+1)" in emitted

    prelude = """
        #include <stddef.h>
        #define NUM_KV_HEADS 16
        #define MAX_SEQ_LEN 32768
        typedef struct { float *kv_cache; } Model;
        void probe_store(float *, float *);
        _Static_assert((size_t)52 * 16 * 32768 * 128 == 3489660928ULL,
                       "long KV offset must remain 64-bit");
    """
    good = _compile(prelude + "\nvoid run(Model *model) {\n" + emitted + "\n}\n")
    assert good.returncode == 0, good.stderr

    bad = _compile(prelude + "\nvoid run(Model *model) {\nprobe_store(" +
                   old_k + ", " + old_v + ");\n}\n")
    assert bad.returncode != 0
    assert "overflow" in bad.stderr.lower()


def test_fp16_pointer_and_fallback_cache_size_are_wide() -> None:
    expression = _normalize_arg_expr(
        "(model->kv_cache_f16 + (26*2)*NUM_KV_HEADS*MAX_SEQ_LEN*128)"
    )
    assert "1ULL*(26*2)" in expression
    layout = {"memory": {"weights": {"size": 0}, "activations": {"size": 0}}}
    config = {
        "embed_dim": 2048, "num_heads": 16, "num_kv_heads": 16,
        "head_dim": 128, "intermediate_size": 10944, "vocab_size": 128896,
        "context_length": 32768, "num_layers": 27, "rotary_dim": 32,
        "rope_theta": 8000000.0, "rope_scaling_type": "none",
        "rope_scaling_factor": 1.0,
    }
    emitted = emit_memory_layout(layout, config)
    assert "#define KV_CACHE_SIZE ((size_t)NUM_LAYERS * 2" in emitted
