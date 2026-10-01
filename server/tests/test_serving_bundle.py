"""Circuit ownership, resolution, and offline serving of immutable assets."""
import hashlib
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from server.live import create_app
from server.runtime import load_manifest_templates, load_tool_protocol
from server.serving_bundle import (
    load_resolved_serving, loaded_serving_identity, resolved_renderer_tokens,
    verify_loaded_libraries,
)
from server.tests.test_native_qwen_jinja_contract import RecordingSession

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
from ck_serve_v8 import create_app as create_serving_app
spec = importlib.util.spec_from_file_location("resolve_serving_bundle_v8", ROOT / "version/v8/scripts/resolve_serving_bundle_v8.py")
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


def _session_library(path: Path, *, marker: int = 1):
    source = path.with_suffix(".c")
    source.write_text(f"int ck_session_v8_open(void) {{ return {marker}; }}\n")
    subprocess.run(["cc", "-shared", "-fPIC", "-o", str(path), str(source)], check=True)
    return ctypes.CDLL(str(path))


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
    (run / "weights_manifest.json").write_text(json.dumps({"template": json.loads(circuit.read_text())}))
    (run / "model_v8.c").write_text("generated test source")
    def identity(path):
        return {"path": str(path), "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    ir_outputs = {"decode_layout": identity(run / "layout_decode.json")}
    model = identity(run / "model_v8.c")
    stamps = {
        "ir": {"inputs": {"schema": "ck-v8-ir-bundle-v1", "manifest": identity(run / "weights_manifest.json")}, "outputs": ir_outputs},
        "codegen": {"inputs": {"schema": "ck-v8-codegen-bundle-v1", "artifacts": ir_outputs}, "outputs": {"model_v8.c": model}},
        "runtime": {"inputs": {"schema": "ck-v8-runtime-bundle-v2", "model_source": model}, "outputs": {name: identity(run / name) for name in ("libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so")}}}
    for phase, stamp in stamps.items():
        (run / f".ck_{phase}_bundle.json").write_text(json.dumps(stamp))
    return run, circuit, source


def resolve(setup, variant=None, *, allow_serving_update=False):
    run, circuit, source = setup
    return resolver.resolve_serving_bundle(run, circuit, variant=variant, v8_root=source, allow_serving_update=allow_serving_update)


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


def test_publisher_chat_profile_rejects_tools_before_generation(setup):
    run, circuit_path, source = setup
    shutil.copyfile(
        ROOT / "version/v8/serving_profiles/publisher_chat_v1.json",
        source / "serving_profiles/publisher_chat_v1.json",
    )
    circuit = json.loads(circuit_path.read_text())
    circuit["serving"]["profile_ref"] = "serving_profiles/publisher_chat_v1.json"
    circuit_path.write_text(json.dumps(circuit))
    resolved = resolver.resolve_serving_bundle(
        run, circuit_path, v8_root=source, allow_serving_update=True,
    )
    assert resolved["output_protocol"] == "none"
    chat, variants, _ = load_manifest_templates(run)
    assert load_tool_protocol(run, chat, variants) == "none"
    session = RecordingSession(["Hello.", "Hello again."])
    client = TestClient(create_serving_app(
        session, model="test", chat_template=chat, chat_templates=variants,
        tool_protocol="none", viz=False,
    ))
    assert client.post("/v1/responses", json={"model": "test", "input": "Hello."}).status_code == 200
    assert client.post("/v1/chat/completions", json={
        "model": "test", "messages": [{"role": "user", "content": "Hello."}],
    }).status_code == 200
    prompts_before_tools = list(session.prompts)
    function = {"name": "read_file", "parameters": {"type": "object", "properties": {}}}
    response_tool = client.post("/v1/responses", json={
        "model": "test", "input": "Read a file", "tools": [{"type": "function", **function}],
    })
    chat_tool = client.post("/v1/chat/completions", json={
        "model": "test", "messages": [{"role": "user", "content": "Read a file"}],
        "tools": [{"type": "function", "function": function}],
    })
    assert response_tool.status_code == 501
    assert chat_tool.status_code == 501
    assert session.prompts == prompts_before_tools


def test_publisher_special_tokens_feed_strict_jinja_from_verified_bundle(setup):
    from server.live import TemplateRenderError, _render_with_chat_templates

    run = setup[0]
    (run / "chat_template.jinja").write_text(
        "{{ bos_token }}{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}assistant:")
    (run / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>"}))
    (run / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": {"content": "<|begin_of_text|>"}}))
    doc = resolve(setup)
    tokens = resolved_renderer_tokens(run, load_resolved_serving(run))
    assert tokens == {"bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>"}
    chat, variants, _ = load_manifest_templates(run)
    with pytest.raises(TemplateRenderError, match="bos_token"):
        _render_with_chat_templates(chat, variants,
                                    [{"role": "user", "content": "Hello."}], None)
    assert _render_with_chat_templates(chat, variants,
                                       [{"role": "user", "content": "Hello."}], None,
                                       renderer_tokens=tokens) == "<|begin_of_text|>user:Hello.assistant:"
    session = RecordingSession(["Hi."])
    client = TestClient(create_app(session, model="test", chat_template=chat,
                                   chat_templates=variants, renderer_tokens=tokens))
    response = client.post("/v1/responses", json={"model": "test", "input": "Hello."})
    assert response.status_code == 200
    assert session.prompts == ["<|begin_of_text|>user:Hello.assistant:"]
    assert "tokenizer_config.json" in doc["assets"]
    (run / "tokenizer_config.json").write_text('{"bos_token":"stale"}')
    with pytest.raises(ValueError, match="stale serving asset"):
        load_resolved_serving(run)


def test_conflicting_publisher_special_tokens_fail_closed(setup):
    run = setup[0]
    (run / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<bos-a>"}))
    (run / "special_tokens_map.json").write_text(json.dumps({"bos_token": "<bos-b>"}))
    doc = resolve(setup)
    with pytest.raises(ValueError, match="conflicting publisher token variable"):
        resolved_renderer_tokens(run, doc)


def test_publisher_date_helper_has_upstream_format_contract(monkeypatch):
    import server.live as live

    class FixedDatetime:
        @staticmethod
        def now():
            from datetime import datetime
            return datetime(2026, 9, 30)

    monkeypatch.setattr(live, "datetime", FixedDatetime)
    actual = live._render_with_chat_templates(
        "{{ bos_token }}Current date: {{ strftime_now('%Y-%m-%d') }}.",
        None, [{"role": "user", "content": "Hello."}], None,
        renderer_tokens={"bos_token": "<|begin_of_text|>"})
    assert actual == "<|begin_of_text|>Current date: 2026-09-30."


def test_request_reuses_one_clock_snapshot_across_prefix_renders(monkeypatch):
    import server.live as live
    from datetime import datetime

    class RolloverDatetime:
        calls = 0

        @classmethod
        def now(cls):
            cls.calls += 1
            return (datetime(2026, 9, 30, 23, 59, 59) if cls.calls == 1
                    else datetime(2026, 10, 1, 0, 0, 0))

    monkeypatch.setattr(live, "datetime", RolloverDatetime)
    template = (
        "{{ strftime_now('%Y-%m-%d') }} {{ strftime_now('%Y-%m-%d') }} "
        "{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    session = RecordingSession(["Hi."])
    client = TestClient(create_app(session, model="test", chat_template=template))
    response = client.post("/v1/responses", json={"model": "test", "input": "Hello."})
    assert response.status_code == 200
    assert session.prompts == ["2026-09-30 2026-09-30 user:Hello.assistant:"]
    assert RolloverDatetime.calls == 1


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
    circuit["serving"]["revision"] = 3
    setup[1].write_text(json.dumps(circuit))
    with pytest.raises(ValueError, match="serving-only changes require"):
        resolve(setup)
    changed_circuit = resolve(setup, allow_serving_update=True)
    assert original["identity"] != changed_circuit["identity"]
    path = setup[2] / "serving_profiles/qwen_tools_v1.json"
    profile = json.loads(path.read_text())
    profile["revision"] = 2
    path.write_text(json.dumps(profile))
    assert resolve(setup, allow_serving_update=True)["identity"] != changed_circuit["identity"]


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


def test_loaded_identity_endpoint_reports_only_verified_bundle_hashes(setup):
    doc = resolve(setup)
    names = ("libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so")
    maps = "\n".join(f"1000-2000 r-xp 00000000 00:01 1 {setup[0] / name}" for name in names)
    verify_loaded_libraries(doc, maps_text=maps)
    session_library = setup[0] / "libck_session_v8.so"
    loaded = _session_library(session_library)
    identity = loaded_serving_identity(
        doc, model="test", session_library=loaded,
    )
    chat, variants, _ = load_manifest_templates(setup[0])
    client = TestClient(create_app(
        RecordingSession(["ok"]), model="test", chat_template=chat, context_length=64,
        chat_templates=variants, loaded_identity=identity,
    ))
    first = client.get("/v1/cke/loaded-identity")
    second = client.get("/v1/cke/loaded-identity")
    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["serving_identity"] == doc["identity"]
    assert first.json()["assets_sha256"]["libmodel.so"] == doc["assets"]["libmodel.so"]["sha256"]
    assert first.json()["session_library_sha256"] == hashlib.sha256(session_library.read_bytes()).hexdigest()
    assert first.json()["effective_serving"]["configured_mode"] == "templated"
    assert first.json()["effective_serving"]["active_context_limit"] == 64
    assert first.json()["effective_serving"]["stop_at_eos"] is False
    assert str(setup[0]) not in first.text  # No local bundle paths in HTTP evidence.
    assert client.get("/v1/models").json()["data"][0]["id"] == "test"
    legacy = TestClient(create_app(RecordingSession(["ok"]), model="test", chat_template=chat))
    assert legacy.get("/v1/cke/loaded-identity").status_code == 404
    raw = TestClient(create_app(RecordingSession(["ok"]), model="test",
                                allow_untemplated=True, context_length=64,
                                stop_on_text=("DONE",), loaded_identity=identity))
    raw_identity = raw.get("/v1/cke/loaded-identity").json()
    assert raw_identity["output_protocol"] == doc["output_protocol"]
    assert raw_identity["effective_serving"] == {
        "schema": "cke.effective_serving.v1", "configured_mode": "raw",
        "output_protocol": None, "active_context_limit": 64,
        "default_max_output_tokens": 512, "stop_on_text": ["DONE"],
        "stop_at_eos": False,
    }


def test_loaded_session_library_replacement_cannot_certify_new_path(setup):
    doc = resolve(setup)
    original_path = setup[0] / "libck_session_v8.so"
    loaded = _session_library(original_path, marker=1)
    original_digest = hashlib.sha256(original_path.read_bytes()).hexdigest()
    assert loaded_serving_identity(doc, model="test", session_library=loaded)[
        "session_library_sha256"] == original_digest
    replacement_dir = setup[0] / "replacement"
    replacement_dir.mkdir()
    replacement_path = replacement_dir / "libck_session_v8.so"
    _session_library(replacement_path, marker=2)
    replacement_digest = hashlib.sha256(replacement_path.read_bytes()).hexdigest()
    assert replacement_digest != original_digest
    os.replace(replacement_path, original_path)
    with pytest.raises(ValueError, match="deleted or ambiguous|replaced"):
        loaded_serving_identity(doc, model="test", session_library=loaded)


@pytest.mark.parametrize("allow_serving_update", [False, True])
def test_same_name_computational_change_rejected(setup, allow_serving_update):
    original = resolve(setup)
    circuit = json.loads(setup[1].read_text())
    circuit["operations"] = [{"op": "changed-computation"}]
    setup[1].write_text(json.dumps(circuit))
    with pytest.raises(ValueError, match="circuit differs"):
        resolve(setup, allow_serving_update=allow_serving_update)
    assert load_resolved_serving(setup[0])["identity"] == original["identity"]


@pytest.mark.parametrize("name", ["weights_manifest.json", ".ck_ir_bundle.json", ".ck_codegen_bundle.json", ".ck_runtime_bundle.json", "model_v8.c", "libmodel.so"])
def test_upgrade_requires_unchanged_compilation_provenance(setup, name):
    (setup[0] / name).write_text("{}")
    with pytest.raises((ValueError, KeyError), match="provenance|snapshot"):
        resolve(setup)
    assert not (setup[0] / "serving.json").exists()


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("loaded_mismatch", [False, True])
def test_chat_and_raw_cli_validate_artifact_integrity(setup, monkeypatch, raw, loaded_mismatch):
    import ck_serve_v8
    import server.serving_bundle as bundle
    resolve(setup)
    monkeypatch.setattr(ck_serve_v8, "_ensure_native_session_lib", lambda: None)
    monkeypatch.setattr(ck_serve_v8, "_resolve_run_dir", lambda *_: setup[0])
    argv = [str(setup[0]), "--no-build", "--no-viz"]
    if raw:
        argv += ["--no-chat-template", "--allow-raw-prompt"]
    if loaded_mismatch:
        closed = []
        class Session:
            def close(self):
                closed.append(True)
        monkeypatch.setattr(ck_serve_v8.SessionV8, "open", lambda *a, **k: Session())
        monkeypatch.setattr(bundle, "verify_loaded_libraries", lambda *_: (_ for _ in ()).throw(ValueError("loaded library identity mismatch")))
        with pytest.raises(ValueError, match="loaded library identity mismatch"):
            ck_serve_v8.main(argv)
        assert closed == [True]
    else:
        (setup[0] / "libmodel.so").write_bytes(b"stale")
        monkeypatch.setattr(ck_serve_v8.SessionV8, "open", lambda *a, **k: pytest.fail("opened stale session"))
        with pytest.raises(ValueError, match="stale serving asset"):
            ck_serve_v8.main(argv)


def test_cli_exposes_loaded_identity_after_runtime_verification(setup, monkeypatch):
    import ck_serve_v8
    import server.serving_bundle as bundle
    import uvicorn

    doc = resolve(setup)
    library = setup[0] / "libck_session_v8.so"
    loaded = _session_library(library)
    closed = []
    class Session:
        lib = loaded
        def close(self):
            closed.append(True)
    monkeypatch.setattr(ck_serve_v8, "_ensure_native_session_lib", lambda: None)
    monkeypatch.setattr(ck_serve_v8, "_resolve_run_dir", lambda *_: setup[0])
    monkeypatch.setattr(ck_serve_v8.SessionV8, "open", lambda *a, **k: Session())
    monkeypatch.setattr(bundle, "verify_loaded_libraries", lambda *_: None)
    served = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **_kwargs: served.append(app))

    assert ck_serve_v8.main([str(setup[0]), "--no-build", "--no-viz", "--model-name", "test-model"]) == 0
    assert closed == [True]
    response = TestClient(served[0]).get("/v1/cke/loaded-identity")
    assert response.status_code == 200
    assert response.json()["serving_identity"] == doc["identity"]
    assert response.json()["model"] == "test-model"


def test_raw_build_without_chat_assets_and_normal_chat_rejection(setup, monkeypatch):
    from types import SimpleNamespace
    import ck_run_v8
    import ck_serve_v8
    (setup[0] / "chat_template.jinja").unlink()
    raw = SimpleNamespace(resolve_serving=False, serving_variant=None, no_chat_template=True)
    # This is the same post-compile step used by raw/token-ID build pipelines.
    ck_run_v8.step_resolve_serving(raw, setup[0], setup[0] / "weights_manifest.json")
    assert not (setup[0] / "serving.json").exists()
    chat = SimpleNamespace(resolve_serving=True, serving_variant=None, no_chat_template=False)
    canonical = setup[2] / "circuits"
    canonical.mkdir()
    shutil.copyfile(setup[1], canonical / "test-circuit.json")
    monkeypatch.setattr(ck_run_v8, "V8_ROOT", setup[2])
    monkeypatch.setattr(__import__("resolve_serving_bundle_v8"), "resolve_serving_bundle", lambda run, circuit, **kw: resolver.resolve_serving_bundle(run, circuit, v8_root=setup[2], **kw))
    with pytest.raises(ValueError, match="publisher default template is missing"):
        ck_run_v8.step_resolve_serving(chat, setup[0], setup[0] / "weights_manifest.json")
    monkeypatch.setattr(ck_serve_v8, "_ensure_native_session_lib", lambda: None)
    monkeypatch.setattr(ck_serve_v8, "_resolve_run_dir", lambda *_: setup[0])
    monkeypatch.setattr(ck_serve_v8.SessionV8, "open", lambda *a, **k: pytest.fail("opened without chat assets"))
    with pytest.raises(ValueError, match="normal chat serving requires"):
        ck_serve_v8.main([str(setup[0]), "--no-build", "--no-viz"])
