"""The published serving inventory follows circuit declarations, not model names."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "version/v8/scripts/build_serving_coverage_inventory_v8.py"
spec = importlib.util.spec_from_file_location("build_serving_coverage_inventory_v8", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_site_inventory_is_current_and_does_not_promote_declarations():
    report = builder.inventory()
    assert report["circuit_count"] == len(list((builder.V8 / "circuits").glob("*.json")))
    assert report["circuit_linked_count"] == sum(
        row["serving_declaration"] == "circuit_linked" for row in report["rows"])
    assert any(row["circuit"] == "qwen38" and row["output_protocol"] == "qwen_xml"
               for row in report["rows"])
    assert any(row["circuit"] == "muse_glimmer_text"
               and row["chat_scope"] == "text_decoder_candidate" for row in report["rows"])
    assert any(row["circuit"] == "gemma4_vision"
               and row["chat_scope"] == "component" for row in report["rows"])
    assert any(row["circuit"] == "qwen38"
               and row["chat_scope"] == "linked_chat_declaration" for row in report["rows"])
    assert sum(report["chat_scope_counts"].values()) == report["circuit_count"]
    for row in report["rows"]:
        assert row["artifact"]["resolved_bundle_identity"] is None
        assert set(row["evidence"].values()) == {"not_assessed"}
    assert json.loads(builder.REPORT.read_text()) == report
    assert builder.PAGE.read_text() == builder.page(report)


def test_profile_reference_changes_inventory_and_missing_reference_fails(tmp_path):
    v8 = tmp_path / "v8"
    circuits = v8 / "circuits"
    profiles = v8 / "serving_profiles"
    circuits.mkdir(parents=True)
    profiles.mkdir()
    circuit = circuits / "sample.json"
    circuit.write_text(json.dumps({"name": "sample", "family": "test"}))
    assert builder.inventory(v8=v8)["rows"][0]["serving_declaration"] == "missing"
    circuit.write_text(json.dumps({"name": "sample", "family": "test", "serving": {
        "schema": "cke.circuit_serving.v1", "profile_ref": "serving_profiles/test.json",
        "default_variant": "publisher"}}))
    with pytest.raises(ValueError, match="missing or escaping serving profile"):
        builder.inventory(v8=v8)
    (profiles / "test.json").write_text(json.dumps({
        "schema": "cke.serving_profile.v1", "id": "test", "renderer": "jinja-chat-v1",
        "input_modalities": ["text"], "variants": {"publisher": {
            "chat": {"source": "publisher", "variant": "default"},
            "tools": {"source": "publisher", "variant": "tool_use"},
            "output_protocol": "test_protocol"}}}))
    row = builder.inventory(v8=v8)["rows"][0]
    assert row["serving_declaration"] == "circuit_linked"
    assert row["output_protocol"] == "test_protocol"
