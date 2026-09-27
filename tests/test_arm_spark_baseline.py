#!/usr/bin/env python3
from __future__ import annotations

"""
ARM (AArch64) v8 baseline certification fixture — NEW CERTIFICATION lane.

This is the DGX Spark (Grace, Cortex-X925/A725, SVE2) CPU baseline recorded
in docs/notes/artifacts/arm_spark_baseline_certification.json. It is a new
certification, not recovered TDA4VM parity.

The fixture drives a compiled Gemma 3 270M Q5_K_M v8 runtime through
version/v8/scripts/arm_baseline_probe_v8.py and pins:
- prompt token IDs (exact)
- greedy decode trajectory token IDs (exact)
- first-prefill argmax and top-8 logit set (exact)
- top-8 logit values (tolerance LOGIT_VALUE_TOLERANCE)

It SKIPS cleanly on non-ARM hosts and when no compiled runtime is available
(set CK_ARM_SPARK_MODEL_DIR or build the default cache model via
version/v8/scripts/cks-v8-run).
"""

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "version" / "v8" / "scripts" / "arm_baseline_probe_v8.py"

DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache"
    / "ck-engine-v8"
    / "models"
    / "unsloth--gemma-3-270m-it-GGUF"
)

EXPECTED_PROMPT_TOKEN_IDS = [2, 105, 2364, 107, 23391, 106, 107, 105, 4368, 107]
# Trajectory is pinned exactly through the first EOS token (index 10). After
# EOS the model emits newline/EOS near-ties (106/107) whose margins are below
# run-to-run numeric noise, so the tail is bounded to the tie set instead of
# pinned exactly.
EXPECTED_TRAJECTORY_THROUGH_EOS = [
    9259, 236888, 2088, 740, 564, 1601, 611, 3124, 236881, 107, 106,
]
EXPECTED_FIRST_EOS_INDEX = 10
POST_EOS_TIE_TOKENS = {106, 107}
EXPECTED_ARGMAX = 9259
EXPECTED_TOP8_IDS = {9259, 10979, 3910, 236777, 242687, 23391, 125508, 144626}
# Spark-certified top-8 values (CK_NUM_THREADS=4, generic AArch64 baseline).
EXPECTED_TOP8_VALUES = {
    9259: 3.097,
    10979: 1.966,
    3910: 0.257,
    236777: -1.979,
    23391: -2.428,
    242687: -3.046,
    125508: -3.172,
    144626: -3.943,
}
LOGIT_VALUE_TOLERANCE = 0.25


def _load_probe():
    scripts = str(PROBE.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("arm_baseline_probe_v8", PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PROBE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _model_dir() -> Path:
    return Path(os.environ.get("CK_ARM_SPARK_MODEL_DIR", DEFAULT_MODEL_DIR))


pytestmark = pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64"),
    reason="ARM Spark baseline fixture only runs on AArch64 hosts",
)


@pytest.fixture(scope="module")
def probe_report(tmp_path_factory):
    model_dir = _model_dir()
    if not (model_dir / "libmodel.so").exists():
        pytest.skip(f"compiled v8 runtime not found at {model_dir}")
    if not (model_dir / "tokenizer.json").exists():
        pytest.skip(f"tokenizer.json not found at {model_dir}")
    probe = _load_probe()
    out_path = tmp_path_factory.mktemp("arm_spark") / "probe.json"
    argv = sys.argv
    try:
        sys.argv = [
            "arm_baseline_probe_v8",
            "--model-dir",
            str(model_dir),
            "--prompt",
            "hello",
            "--chat-template",
            "auto",
            "--max-tokens",
            "32",
            "--json-out",
            str(out_path),
        ]
        rc = probe.main()
    finally:
        sys.argv = argv
    if rc != 0:
        pytest.fail(f"probe exited with rc={rc}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_prompt_token_ids_exact(probe_report):
    assert probe_report["prompt_token_ids"] == EXPECTED_PROMPT_TOKEN_IDS


def test_decode_trajectory_exact(probe_report):
    decode = probe_report["decode"]
    assert decode["first_eos_index"] == EXPECTED_FIRST_EOS_INDEX
    tokens = decode["token_ids"]
    assert tokens[: EXPECTED_FIRST_EOS_INDEX + 1] == EXPECTED_TRAJECTORY_THROUGH_EOS
    assert set(tokens[EXPECTED_FIRST_EOS_INDEX + 1 :]) <= POST_EOS_TIE_TOKENS


def test_first_prefill_argmax_and_topk(probe_report):
    logits = probe_report["first_token_logits"]
    assert logits["argmax"] == EXPECTED_ARGMAX
    topk = {entry["id"]: entry["value"] for entry in logits["topk"][:8]}
    assert set(topk) == EXPECTED_TOP8_IDS
    for token_id, expected in EXPECTED_TOP8_VALUES.items():
        assert abs(topk[token_id] - expected) <= LOGIT_VALUE_TOLERANCE, (
            f"top-8 logit for token {token_id} drifted: "
            f"{topk[token_id]} vs certified {expected} "
            f"(tol {LOGIT_VALUE_TOLERANCE})"
        )
