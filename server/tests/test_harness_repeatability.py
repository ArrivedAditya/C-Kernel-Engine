"""The repeatability coordinator retains failures without inventing certification."""
from pathlib import Path

import pytest

from server import harness_repeatability as matrix


HASH = "a" * 64


def _config():
    return {"schema": matrix.SCHEMA, "attempts": 2, "clients": [{
        "name": "qwen-control", "adapter": "qwen_code",
        "endpoint": "http://127.0.0.1:18103/v1", "model": "qwen-test",
        "serving_identity": HASH, "session_library_sha256": HASH,
        "qwen_bin": "qwen", "qwen_version": "0.24.6",
    }]}


def test_config_rejects_remote_endpoints_and_unpinned_artifacts():
    config = _config()
    config["clients"][0]["endpoint"] = "https://cloud.example/v1"
    with pytest.raises(ValueError, match="localhost"):
        matrix._validate(config)
    config = _config()
    config["clients"][0]["serving_identity"] = "unverified"
    with pytest.raises(ValueError, match="serving_identity"):
        matrix._validate(config)


def test_qwen_child_workspace_is_separate_from_retained_logs(tmp_path):
    command, report, _ = matrix._command(_config()["clients"][0], "task", tmp_path)
    assert command[command.index("--run-dir") + 1] == str(tmp_path / "workspace")
    assert report == tmp_path / "workspace" / "report.json"


def test_matrix_preserves_every_attempt_and_separates_statuses(tmp_path, monkeypatch):
    calls = []

    def fake_step(client, kind, directory: Path):
        directory.mkdir(parents=True)
        calls.append((kind, directory.name))
        return {"client": client["name"], "kind": kind, "timed_out": False,
                "status": "fail" if directory.name == "task-01" else "pass",
                "task_status": "fail" if directory.name == "task-01" else "pass",
                "loaded_identity_status": "pass"}

    monkeypatch.setattr(matrix, "_run_step", fake_step)
    result = matrix.run(_config(), tmp_path / "matrix")
    assert result["status"] == "fail"
    assert calls == [("plain-chat", "plain-chat-01"),
                     ("budget-boundary", "budget-boundary-01"),
                     ("lifecycle", "lifecycle-01"),
                     ("task", "task-01"), ("task", "task-02")]
    assert (tmp_path / "matrix" / "summary.json").is_file()


def test_timeout_stops_reusing_that_endpoint(tmp_path, monkeypatch):
    calls = []

    def fake_step(client, kind, directory: Path):
        calls.append(kind)
        return {"client": client["name"], "kind": kind,
                "timed_out": kind == "task", "status": "fail" if kind == "task" else "pass"}

    monkeypatch.setattr(matrix, "_run_step", fake_step)
    result = matrix.run(_config(), tmp_path / "matrix")
    assert calls == ["plain-chat", "budget-boundary", "lifecycle", "task"]
    assert result["status"] == "fail"


def test_identity_is_independent_of_task_success():
    client = _config()["clients"][0]
    identity = {"model": "qwen-test", "serving_identity": HASH,
                "session_library_sha256": HASH}
    report = {"identity": {"before": identity, "after": identity},
              "result": {"status": "fail"}}
    assert matrix._identity_status(client, report) == "pass"
    report["identity"]["after"] = {**identity, "serving_identity": "b" * 64}
    assert matrix._identity_status(client, report) == "fail"


def test_successful_child_exit_cannot_certify_wrong_report_schema(tmp_path, monkeypatch):
    import sys

    def fake_command(client, kind, directory):
        report = directory / "report.json"
        source = ("import json,sys; open(sys.argv[1], 'w').write(json.dumps("
                  "{'schema':'wrong','model':'qwen-test','status':'pass'}))")
        return [sys.executable, "-c", source, str(report)], report, 10

    monkeypatch.setattr(matrix, "_command", fake_command)
    row = matrix._run_step(_config()["clients"][0], "task", tmp_path / "step")
    assert row["exit_code"] == 0
    assert row["status"] == "fail"
    assert row["report_schema_status"] == "missing_or_invalid"
