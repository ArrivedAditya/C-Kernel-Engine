from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "xray_ck_persistent_replay_v8.py"
PROFILE = ROOT / "version" / "v8" / "parity_profiles" / "text_ck_persistent_replay_v1.json"
spec = importlib.util.spec_from_file_location("xray_ck_persistent_replay_tests", SCRIPT)
assert spec is not None and spec.loader is not None
xray = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = xray
spec.loader.exec_module(xray)
PROFILE_DATA = __import__("json").loads(PROFILE.read_text(encoding="utf-8"))


def test_layout_aware_capture_comparison_finds_first_consumed_edge(tmp_path: Path) -> None:
    persistent = tmp_path / "persistent"
    replay = tmp_path / "replay"
    persistent.mkdir()
    replay.mkdir()
    config = {"num_heads": 2, "num_kv_heads": 1}
    decode_operations = [
        {"idx": 1, "layer": 0, "op": "residual_save", "function": "memcpy"},
        {"idx": 2, "layer": 0, "op": "rmsnorm", "function": "rmsnorm_forward"},
        {"idx": 3, "layer": 0, "op": "q_proj", "function": "q_gemv"},
    ]
    prefill_operations = [
        {"idx": 11, "layer": 0, "op": "residual_save", "function": "memcpy"},
        {"idx": 12, "layer": 0, "op": "rmsnorm", "function": "rmsnorm_forward"},
        {"idx": 13, "layer": 0, "op": "q_proj", "function": "q_gemm"},
    ]
    token_rows = np.arange(12, dtype=np.float32).reshape(3, 4)
    token_rows.tofile(replay / "tok_0000_layer_000_layer_input.f32")
    token_rows[-1].tofile(persistent / "tok_0002_layer_000_layer_input.f32")
    token_rows.tofile(replay / "tok_0000_layer_000_block_rmsnorm.f32")
    token_rows[-1].tofile(persistent / "tok_0002_layer_000_block_rmsnorm.f32")
    q = np.arange(12, dtype=np.float32).reshape(3, 4)
    q.tofile(replay / "tok_0000_layer_000_q_proj_post_bias.f32")
    changed = q[-1].copy()
    changed[1] += np.float32(0.25)
    changed.tofile(persistent / "tok_0002_layer_000_q_proj_post_bias.f32")
    rope_q = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    rope_q.tofile(replay / "tok_0000_layer_000_rope_q.f32")
    persistent_rope_q = rope_q[:, -1, :].copy().reshape(-1)
    persistent_rope_q[0] += np.float32(0.125)
    persistent_rope_q.tofile(persistent / "tok_0002_layer_000_rope_q.f32")

    report = xray.compare_captures(
        persistent_dir=persistent,
        replay_dir=replay,
        decode_call_ir={"config": config, "operations": decode_operations},
        prefill_call_ir={"config": config, "operations": prefill_operations},
        layer=0,
        logical_token=2,
        replay_tokens=3,
        profile=PROFILE_DATA,
    )

    assert report["status"] == "observed"
    assert report["capture_scope"] == {
        "comparison": "CK persistent decode vs CK full replay",
        "phase": "decoder",
        "model_layer_count": 1,
        "selected_layers": [0],
        "covered_layer_count": 1,
        "coverage_complete": True,
        "checkpoint_granularity": "detailed_operator_chain",
    }
    assert report["comparisons"][0]["metrics"]["byte_exact"]
    assert report["first_divergence"] is None
    observed = report["first_observed_divergence"]
    assert observed["checkpoint_id"] == "text.layer.0.q_proj_post_bias"
    assert observed["op_idx"] == 3
    assert observed["subject_execution"]["function"] == "q_gemv"
    assert observed["oracle_execution"]["function"] == "q_gemm"
    assert observed["oracle_execution"]["op_idx"] == 13
    assert observed["metrics"]["max_abs"] == 0.25
    assert observed["classification"] == "PROVIDER_SCHEDULE_DIFFERENCE"
    assert observed["attribution_status"] == "non_causal_mode_change"
    rope = next(
        row for row in report["comparisons"]
        if row["checkpoint_id"] == "text.layer.0.rope_q"
    )
    assert rope["classification"] == "PROPAGATED_PROVIDER_SCHEDULE_DIFFERENCE"
    assert rope["divergent_dependencies"] == [
        "text.layer.0.q_proj_post_bias"
    ]


def test_matching_provider_difference_is_a_causal_candidate(tmp_path: Path) -> None:
    persistent = tmp_path / "persistent"
    replay = tmp_path / "replay"
    persistent.mkdir()
    replay.mkdir()
    replay_values = np.arange(6, dtype=np.float32).reshape(2, 3)
    replay_values.tofile(replay / "tok_0000_layer_000_block_rmsnorm.f32")
    changed = replay_values[-1].copy()
    changed[0] += np.float32(0.5)
    changed.tofile(persistent / "tok_0001_layer_000_block_rmsnorm.f32")
    operation = {
        "idx": 2,
        "layer": 0,
        "op": "rmsnorm",
        "kernel": "rmsnorm_forward",
        "function": "rmsnorm_forward",
    }

    report = xray.compare_captures(
        persistent_dir=persistent,
        replay_dir=replay,
        decode_call_ir={"config": {}, "operations": [operation]},
        prefill_call_ir={"config": {}, "operations": [operation]},
        layer=0,
        logical_token=1,
        replay_tokens=2,
        profile=PROFILE_DATA,
    )

    assert report["status"] == "fail"
    assert report["first_divergence"]["checkpoint_id"] == "text.layer.0.block_rmsnorm"
    assert report["first_divergence"]["classification"] == "PERSISTENT_REPLAY_DIVERGENCE"
    assert report["first_divergence"]["attribution_status"] == "causal_candidate"


def test_coarse_layer_sweep_finds_first_amplification(tmp_path: Path) -> None:
    persistent = tmp_path / "persistent"
    replay = tmp_path / "replay"
    persistent.mkdir()
    replay.mkdir()
    decode_ops = []
    prefill_ops = []
    for layer in range(3):
        decode_ops.append(
            {"idx": layer * 10 + 1, "layer": layer, "op": "residual_add"}
        )
        prefill_ops.append(
            {"idx": layer * 10 + 2, "layer": layer, "op": "residual_add"}
        )
        oracle = np.arange(8, dtype=np.float32).reshape(2, 4)
        oracle.tofile(
            replay / f"tok_0000_layer_{layer:03d}_layer_out.f32"
        )
        subject = oracle[-1].copy()
        subject[0] += np.float32([1e-6, 2e-4, 4e-4][layer])
        subject.tofile(
            persistent / f"tok_0001_layer_{layer:03d}_layer_out.f32"
        )

    report = xray.compare_layer_sweep(
        persistent_dir=persistent,
        replay_dir=replay,
        decode_call_ir={
            "config": {"num_layers": 3},
            "operations": decode_ops,
        },
        prefill_call_ir={
            "config": {"num_layers": 3},
            "operations": prefill_ops,
        },
        logical_token=1,
        replay_tokens=2,
        amplification_ratio=100.0,
    )

    assert len(report["comparisons"]) == 3
    assert report["capture_scope"]["covered_layer_count"] == 3
    assert report["capture_scope"]["coverage_complete"]
    assert report["capture_scope"]["checkpoint_granularity"] == (
        "coarse_layer_out_layer_sweep"
    )
    candidate = report["first_layer_amplification"]
    assert candidate["layer"] == 1
    assert candidate["classification"] == "LAYER_AMPLIFICATION_CANDIDATE"
