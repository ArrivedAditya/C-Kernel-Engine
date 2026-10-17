#!/usr/bin/env python3
from __future__ import annotations

"""
Last-vs-full logits layout equivalence on a small local fixture.

After #577, --logits-layout auto plans last-position logits (the runtime
contract); explicit "full" must keep producing the full-sequence logits
matrix whose FINAL row is what "last" computes. This test builds two tiny
runtimes from the same locally cached GGUF (no downloads) with explicit
--logits-layout last and --logits-layout full, replays the same token IDs
through both via the model ABI, and asserts the last-layout logits equal
the final row of the full-layout matrix within tight tolerance.

It SKIPS cleanly when no local GGUF is available (set CK_LOGITS_EQ_GGUF to
point at one). Callers that genuinely need full-sequence logits (training:
run_training_workflow_v8.py) already request "full" explicitly — this test
is the guard that the two layouts cannot silently disagree.
"""

import ctypes
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CKS_RUN = ROOT / "version" / "v8" / "scripts" / "cks-v8-run"

PROMPT_TOKEN_IDS = [2, 105, 2364, 107, 23391, 106, 107, 105, 4368, 107]
MAX_ABS_TOLERANCE = 1e-3

F32P = ctypes.POINTER(ctypes.c_float)


def _default_gguf() -> Path | None:
    env = os.environ.get("CK_LOGITS_EQ_GGUF")
    if env:
        path = Path(env).expanduser()
        return path if path.exists() else None
    candidate = (
        Path.home()
        / ".cache"
        / "ck-engine-v8"
        / "models"
        / "unsloth--gemma-3-270m-it-GGUF"
        / "gemma-3-270m-it-Q5_K_M.gguf"
    )
    return candidate if candidate.exists() else None


def _build_runtime(gguf: Path, run_dir: Path, logits_layout: str) -> Path:
    cmd = [
        str(CKS_RUN),
        "run",
        str(gguf),
        "--run",
        str(run_dir),
        "--context-len",
        "128",
        "--logits-layout",
        logits_layout,
        "--force-compile",
        "--force-convert",
        "--temperature",
        "0.0",
        "--max-tokens",
        "1",
        "--prompt",
        "hello",
        "--chat-template",
        "auto",
    ]
    env = dict(os.environ)
    env.setdefault("CK_NUM_THREADS", "4")
    proc = subprocess.run(
        cmd, cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=1800
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"runtime build failed for layout {logits_layout}:\n"
            f"{proc.stdout[-4000:]}"
        )
    return run_dir


def _load_logits(model_dir: Path) -> tuple[np.ndarray, dict]:
    lib = ctypes.CDLL(str(model_dir / "libmodel.so"), mode=ctypes.RTLD_GLOBAL)
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_forward.argtypes = [F32P]
    lib.ck_model_forward.restype = ctypes.c_int
    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = F32P
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_kv_cache_reset.argtypes = []
    lib.ck_model_kv_cache_reset.restype = None

    rc = lib.ck_model_init(str(model_dir).encode("utf-8"))
    if rc != 0:
        rc = lib.ck_model_init(str(model_dir / "weights.bump").encode("utf-8"))
    assert rc == 0, f"ck_model_init failed for {model_dir}"
    vocab = int(lib.ck_model_get_vocab_size())
    tokens = (ctypes.c_int32 * len(PROMPT_TOKEN_IDS))(*PROMPT_TOKEN_IDS)
    assert lib.ck_model_embed_tokens(tokens, len(PROMPT_TOKEN_IDS)) == 0
    assert lib.ck_model_forward(None) == 0
    ptr = lib.ck_model_get_logits()
    assert ptr, "ck_model_get_logits returned null"

    stride = 0
    if hasattr(lib, "ck_model_get_logits_stride"):
        lib.ck_model_get_logits_stride.argtypes = []
        lib.ck_model_get_logits_stride.restype = ctypes.c_int
        stride = int(lib.ck_model_get_logits_stride())
    active = len(PROMPT_TOKEN_IDS)
    if hasattr(lib, "ck_model_get_active_tokens"):
        lib.ck_model_get_active_tokens.argtypes = []
        lib.ck_model_get_active_tokens.restype = ctypes.c_int
        active = int(lib.ck_model_get_active_tokens())
    if stride > 0 and active > 0:
        flat = np.ctypeslib.as_array(ptr, shape=(active * stride,))
        start = (active - 1) * stride
        logits = flat[start : start + vocab].astype(np.float32, copy=True)
    else:
        logits = np.ctypeslib.as_array(ptr, shape=(vocab,)).astype(np.float32, copy=True)
    if hasattr(lib, "ck_model_free"):
        lib.ck_model_free()
    meta = {"vocab": vocab, "stride": stride, "active_tokens": active}
    return logits, meta


gguf = _default_gguf()
pytestmark = pytest.mark.skipif(gguf is None, reason="no local GGUF fixture available")


@pytest.fixture(scope="module")
def runtimes(tmp_path_factory):
    work = tmp_path_factory.mktemp("logits_layout_eq")
    last_dir = _build_runtime(gguf, work / "last", "last")
    full_dir = _build_runtime(gguf, work / "full", "full")
    return {"last": last_dir, "full": full_dir}


def test_generated_config_records_layout(runtimes):
    for layout, run_dir in runtimes.items():
        layout_doc = json.loads((run_dir / "layout_decode.json").read_text(encoding="utf-8"))
        recorded = str(layout_doc.get("config", {}).get("logits_layout", "")).lower()
        assert recorded == layout, (
            f"{run_dir}: expected resolved logits_layout={layout}, got {recorded!r}"
        )


def test_last_logits_match_final_row_of_full(runtimes):
    last_logits, last_meta = _load_logits(runtimes["last"])
    full_final_row, full_meta = _load_logits(runtimes["full"])
    assert full_meta["stride"] == full_meta["vocab"] > 0, (
        f"full layout must expose a strided logits matrix, got meta {full_meta}"
    )
    assert last_meta["vocab"] == full_meta["vocab"]
    assert int(np.argmax(last_logits)) == int(np.argmax(full_final_row))
    max_abs = float(np.max(np.abs(last_logits - full_final_row)))
    assert max_abs <= MAX_ABS_TOLERANCE, (
        f"last-layout logits diverge from full-layout final row: max-abs {max_abs}"
    )
