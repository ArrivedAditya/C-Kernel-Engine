from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "version/v8/scripts/certify_multimodal_llamacpp_ocr_v8.py"
SPEC = importlib.util.spec_from_file_location("certify_multimodal_llamacpp_ocr_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
import sys

sys.path.insert(0, str(SCRIPT.parent))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_agreement_requires_both_fields_to_be_present():
    truth = {"present": "yes", "missing": "", "different": "one"}
    result = MODULE._field_comparison(
        truth, {"present": "yes", "different": "one"},
        {"present": "YES", "different": "two"},
    )
    assert result == {
        "expected_fields": 3,
        "agreed_present_fields": 1,
        "both_correct": 1,
        "cke_only_correct": 1,
        "oracle_only_correct": 0,
        "raw_json_equal": False,
    }


def test_oracle_command_uses_original_image_and_explicit_generation_contract():
    command = MODULE._oracle_command(
        {"cli": {"path": "/build/llama-mtmd-cli"}},
        {
            "decoder_gguf": {"path": "/model.gguf"},
            "encoder_gguf": {"path": "/projector.gguf"},
            "context_len": 4096,
        },
        Path("/private/image.png"), "read this", 896, 16, "off",
    )
    assert command[command.index("--image") + 1] == "/private/image.png"
    assert command[command.index("-p") + 1] == "read this"
    for flag, value in (("-n", "896"), ("-c", "4096"), ("--temp", "0"), ("--flash-attn", "off")):
        assert command[command.index(flag) + 1] == value
    assert "--log-disable" not in command


def test_resumed_case_rejects_stale_output_and_metrics(tmp_path):
    truth = {"field": "yes"}
    cke = {"parsed_output": {"field": "yes"}, "metrics": MODULE._score(truth, {"field": "yes"})}
    raw = tmp_path / "llama_raw.txt"
    raw.write_text('{"field":"yes"}\n', encoding="utf-8")
    parsed = {"field": "yes"}
    identity = {"source_commit": "pinned"}
    expected = {"global_config_sha256": "selected"}
    row = {
        "status": "complete",
        "case_identity": expected,
        "oracle_identity": identity,
        "oracle_output_sha256": MODULE._sha256_bytes(raw.read_bytes()),
        "oracle_parsed_output": parsed,
        "oracle_metrics": MODULE._score(truth, parsed),
        "cke_metrics": cke["metrics"],
        "comparison": MODULE._field_comparison(truth, cke["parsed_output"], parsed),
        "oracle_wall_sec": 1.0,
    }
    path = tmp_path / "case_result.json"
    path.write_text(json.dumps(row), encoding="utf-8")
    assert MODULE._resumed_case(path, raw, expected, truth, cke, identity) == row
    raw.write_text('{"field":"no"}\n', encoding="utf-8")
    assert MODULE._resumed_case(path, raw, expected, truth, cke, identity) is None
    raw.write_text('{"field":"yes"}\n', encoding="utf-8")
    row["oracle_metrics"]["exact_fields"] = 0
    path.write_text(json.dumps(row), encoding="utf-8")
    assert MODULE._resumed_case(path, raw, expected, truth, cke, identity) is None


def test_aggregate_keeps_execution_and_quality_separate():
    row = {
        "status": "complete",
        "cke_metrics": {"exact_fields": 1},
        "oracle_metrics": {"exact_fields": 2},
        "comparison": {"expected_fields": 3, "agreed_present_fields": 1, "both_correct": 1, "raw_json_equal": False},
        "oracle_wall_sec": 2.0,
    }
    result = MODULE._aggregate([row, {"status": "error"}], 2)
    assert result["completed"] == 1
    assert result["errors"] == 1
    assert result["cke_exact_fields"] == 1
    assert result["oracle_exact_fields"] == 2
    assert result["raw_json_equal_cases"] == 0
