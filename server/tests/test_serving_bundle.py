"""Circuit ownership, resolution, and offline serving of immutable assets."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
from fastapi.testclient import TestClient

from server.live import create_app
from server.runtime import load_manifest_templates, load_tool_protocol
from server.serving_bundle import load_resolved_serving, verify_loaded_libraries
from server.tests.test_native_qwen_jinja_contract import RecordingSession

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("resolve_serving_bundle_v8", ROOT / "version/v8/scripts/resolve_serving_bundle_v8.py")
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "source"
    profiles = source / "serving_profiles"
    profiles.mkdir(parents=True)
    shutil.copyfile(ROOT / "version/v8/serving_profiles/qwen_tools_v1.json", profiles / "qwen_tools_v1.json")
    shutil.copyfile(ROOT / "server/tests/fixtures/qwen_code_xml_compat.jinja", profiles / "test_variant.jinja")
    profile = json.loads((profiles / "qwen_tools_v1.json").read_text())
    # A synthetic profile variant tests generic asset resolution. It is not a
    # shipped Qwen workaround or a serving default.
    profile["variants"]["compat"] = {
        "chat": {"source": "publisher", "variant": "default"},
        "tools": {"source": "profile_asset", "path": "test_variant.jinja"},
        "output_protocol": "qwen_code_xml_raw_v2"}
    (profiles / "qwen_tools_v1.json").write_text(json.dumps(profile))
    circuit = source / "circuit.json"
    circuit.write_text(json.dumps({"name": "test-circuit", "serving": {
        "schema": "cke.circuit_serving.v1", "profile_ref": "serving_profiles/qwen_tools_v1.json", "default_variant": "publisher"}}))
    run = tmp_path / "bundle"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"model": "test-circuit"}))
    (run / "layout_decode.json").write_text(json.dumps({"config": {"context_length": 128}}))
    for name in ("libmodel.so", "libckernel_tokenizer.so", "libckernel_engine.so", "weights.bump", "weights_manifest.map"):
        (run / name).write_bytes(b"test-" + name.encode())
    (run / "chat_template.jinja").write_bytes(b"{% for message in messages %}{{ message.role }}:{{ message.content }}\r\n{% endfor %}assistant:")
    return run, circuit, source


def resolve(setup, variant=None):
    run, circuit, source = setup
    return resolver.resolve_serving_bundle(run, circuit, variant=variant, v8_root=source)


def test_publisher_default_and_explicit_compat(setup):
    native = resolve(setup)
    assert native["variant"] == "publisher"
    assert native["output_protocol"] == "qwen_xml"
    chat, variants, _ = load_manifest_templates(setup[0])
    assert "\r\n" in chat and variants["tool_use"] == chat
    compat = resolve(setup, "compat")
    assert compat["identity"] != native["identity"]
    chat, variants, _ = load_manifest_templates(setup[0])
    assert "\r\n" in chat and "function_calls" in variants["tool_use"]
    assert load_tool_protocol(setup[0], chat, variants) == "qwen_code_xml_raw_v2"
    with pytest.raises(ValueError, match="override conflicts"):
        load_tool_protocol(setup[0], chat + "changed", variants)


def test_copied_bundle_serves_without_profile_or_server_templates(setup):
    resolve(setup, "compat")
    run, _, source = setup
    copied = run.parent / "copied"
    shutil.copytree(run, copied)
    shutil.rmtree(run)
    shutil.rmtree(source)
    chat, variants, _ = load_manifest_templates(copied)
    protocol = load_tool_protocol(copied, chat, variants)
    session = RecordingSession([
        '<function_calls><invoke name="read_file"><parameter name="file_path">/tmp/probe</parameter></invoke></function_calls>',
        "Read complete."])
    client = TestClient(create_app(session, model="test", chat_template=chat, chat_templates=variants, tool_protocol=protocol))
    first = client.post("/v1/responses", json={"model": "test", "input": "Read the probe.", "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}]})
    assert first.status_code == 200
    call = next(item for item in first.json()["output"] if item["type"] == "function_call")
    second = client.post("/v1/responses", json={"model": "test", "previous_response_id": first.json()["id"], "input": [{"type": "function_call_output", "call_id": call["call_id"], "output": "probe contents"}]})
    assert second.status_code == 200
    assert "Read complete." in str(second.json())


@pytest.mark.parametrize("asset", ["config.json", "layout_decode.json", "weights.bump", "weights_manifest.map", "libmodel.so", "libckernel_tokenizer.so", "libckernel_engine.so"])
def test_stale_runtime_asset_rejected(setup, asset):
    resolve(setup)
    (setup[0] / asset).write_bytes(b"changed")
    with pytest.raises(ValueError, match="stale serving asset"):
        load_manifest_templates(setup[0])


def test_missing_profile_and_unsupported_variant_leave_previous_contract(setup):
    initial = resolve(setup)
    with pytest.raises(ValueError, match="unsupported serving variant"):
        resolve(setup, "unknown")
    assert load_resolved_serving(setup[0])["identity"] == initial["identity"]
    (setup[2] / "serving_profiles/qwen_tools_v1.json").unlink()
    with pytest.raises(ValueError, match="missing or escaping"):
        resolve(setup)
    assert load_resolved_serving(setup[0])["identity"] == initial["identity"]


def test_conflicting_publisher_assets_rejected(setup):
    (setup[0] / "tokenizer_config.json").write_text(json.dumps({"chat_template": "different"}))
    with pytest.raises(ValueError, match="conflicting publisher"):
        resolve(setup)
    assert not (setup[0] / "serving.json").exists()


def test_safetensors_embedded_publisher_template(setup):
    (setup[0] / "chat_template.jinja").unlink()
    (setup[0] / "tokenizer_config.json").write_text(json.dumps({"chat_template": [{"name": "default", "template": "{{ messages[0].content }}"}, {"name": "tool_use", "template": "tools={{ tools | tojson }}"}]}))
    doc = resolve(setup)
    assert doc["assets"]["chat"]["source"] == "tokenizer_config.json:default"
    assert doc["assets"]["tools"]["source"] == "tokenizer_config.json:tool_use"


def test_legacy_operator_variant_is_not_publisher_default(setup):
    directory = setup[0] / "additional_chat_templates"
    directory.mkdir()
    (directory / "tool_use.jinja").write_text("operator compatibility override")
    doc = resolve(setup)
    assert doc["assets"]["tools"]["sha256"] == doc["assets"]["chat"]["sha256"]


def test_changed_profile_and_circuit_change_bundle_identity(setup):
    original = resolve(setup)
    circuit = json.loads(setup[1].read_text())
    circuit["version"] = 3
    setup[1].write_text(json.dumps(circuit))
    changed_circuit = resolve(setup)
    assert original["identity"] != changed_circuit["identity"]
    path = setup[2] / "serving_profiles/qwen_tools_v1.json"
    profile = json.loads(path.read_text())
    profile["revision"] = 2
    path.write_text(json.dumps(profile))
    assert resolve(setup)["identity"] != changed_circuit["identity"]


def test_corrupt_sidecar_does_not_fall_back(setup):
    resolve(setup)
    sidecar = setup[0] / "serving.json"
    doc = json.loads(sidecar.read_text())
    doc["variant"] = "compat"
    sidecar.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_manifest_templates(setup[0])


def test_incompatible_circuit_and_escaped_asset_rejected(setup):
    (setup[0] / "config.json").write_text(json.dumps({"model": "another-circuit"}))
    with pytest.raises(ValueError, match="does not match"):
        resolve(setup)
    (setup[0] / "config.json").write_text(json.dumps({"model": "test-circuit"}))
    (setup[0] / "weights.bump").unlink()
    external = setup[0].parent / "outside.bump"
    external.write_bytes(b"outside")
    (setup[0] / "weights.bump").symlink_to(external)
    with pytest.raises(ValueError, match="escapes bundle"):
        resolve(setup)
    assert not (setup[0] / "serving.json").exists()


def test_profile_reference_cannot_escape_source_root(setup):
    doc = json.loads(setup[1].read_text())
    doc["serving"]["profile_ref"] = "../elsewhere.json"
    setup[1].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="missing or escaping"):
        resolve(setup)


def test_missing_selected_template_rejects_without_legacy_fallback(setup):
    doc = resolve(setup, "compat")
    (setup[0] / doc["assets"]["tools"]["path"]).unlink()
    with pytest.raises(ValueError, match="invalid resolved serving bundle"):
        load_manifest_templates(setup[0])


def test_loaded_library_identity_rejects_alternate_engine(setup):
    doc = resolve(setup)
    names = ("libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so")
    maps = "\n".join(f"1000-2000 r-xp 00000000 00:01 1 {setup[0] / name}" for name in names)
    verify_loaded_libraries(doc, maps_text=maps)
    elsewhere = setup[0].parent / "elsewhere"
    elsewhere.mkdir()
    alternate = elsewhere / "libckernel_engine.so"
    alternate.write_bytes(b"different engine")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_loaded_libraries(doc, maps_text=maps.replace(str(setup[0] / alternate.name), str(alternate)))
    with pytest.raises(ValueError, match="not loaded"):
        verify_loaded_libraries(doc, maps_text="")
