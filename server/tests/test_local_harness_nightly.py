"""A local harness night cannot turn missing execution into PASS."""
import json

from scripts import local_harness_nightly as lane


def _config(path):
    path.write_text(json.dumps({
        "schema": "cke.harness_repeatability.v1", "attempts": 1,
        "clients": [{
            "name": "qwen-control", "adapter": "qwen_code",
            "endpoint": "http://127.0.0.1:19099/v1", "model": "qwen-test",
            "serving_identity": "a" * 64, "session_library_sha256": "b" * 64,
            "qwen_bin": "qwen", "qwen_version": "0.24.6",
        }],
    }))


def test_missing_config_is_not_tested(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    report = lane.run(tmp_path / "missing.json", tmp_path / "state")
    assert report["status"] == "NOT_TESTED"
    assert json.loads((tmp_path / "state/latest.json").read_text())["status"] == "NOT_TESTED"


def test_unavailable_endpoint_cannot_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    config = tmp_path / "config.json"
    _config(config)
    monkeypatch.setattr(lane, "_preflight", lambda _: ("NOT_TESTED", "connection refused", None))
    report = lane.run(config, tmp_path / "state")
    assert report["status"] == "INCOMPLETE"
    assert report["clients"][0]["status"] == "NOT_TESTED"


def test_wrong_identity_fails_before_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    config = tmp_path / "config.json"
    _config(config)
    monkeypatch.setattr(lane, "_preflight", lambda _: ("FAIL", "wrong identity", "instance"))
    monkeypatch.setattr(lane, "_invoke", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("tools must not run")))
    report = lane.run(config, tmp_path / "state")
    assert report["status"] == "FAIL"
    assert report["clients"][0]["reason"] == "wrong identity"


def test_invalid_identity_schema_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_identity", lambda _: (_ for _ in ()).throw(
        ValueError("unsupported schema")))
    client = {"endpoint": "http://127.0.0.1:19099/v1"}
    status, reason, instance = lane._preflight(client)
    assert status == "FAIL" and "unsupported schema" in reason and instance is None


def test_child_exit_alone_cannot_pass_without_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    config = tmp_path / "config.json"
    _config(config)
    monkeypatch.setattr(lane, "_preflight", lambda _: ("READY", "", "instance"))
    monkeypatch.setattr(lane, "_invoke", lambda *args, **kwargs: 0)
    report = lane.run(config, tmp_path / "state")
    assert report["status"] == "FAIL"
    assert report["clients"][0]["reason"] == "repeatability or reassessment failed"


def test_pause_retains_missing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    config = tmp_path / "config.json"
    _config(config)
    state = tmp_path / "state"
    state.mkdir()
    (state / "PAUSE").touch()
    report = lane.run(config, state)
    assert report["status"] == "NOT_TESTED"
    assert report["clients"][0]["status"] == "NOT_TESTED"
