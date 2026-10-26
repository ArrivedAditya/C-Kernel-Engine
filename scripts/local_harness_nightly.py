#!/usr/bin/env python3
"""Run the existing Qwen/Gemma repeatability matrix on available local endpoints.

This opt-in client-side nightly never launches models or retries tool tasks. A
missing endpoint is NOT_TESTED; a wrong loaded identity is FAIL. Configure one
exact artifact per client in a private cke.harness_repeatability.v1 JSON file.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cke.local_harness_nightly.v1"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _identity(endpoint: str) -> dict:
    with urlopen(endpoint.rstrip("/") + "/cke/loaded-identity", timeout=5) as response:
        data = json.load(response)
    if not isinstance(data, dict) or data.get("schema") != "cke.loaded_serving_identity.v1":
        raise ValueError("endpoint returned an unsupported identity schema")
    return data


def _preflight(client: dict) -> tuple[str, str, str | None]:
    try:
        observed = _identity(client["endpoint"])
    except HTTPError as exc:
        return "FAIL", f"loaded identity endpoint returned HTTP {exc.code}", None
    except ValueError as exc:
        return "FAIL", f"loaded identity is invalid: {exc}", None
    except (OSError, URLError, TimeoutError) as exc:
        return "NOT_TESTED", f"endpoint unavailable: {type(exc).__name__}", None
    instance = observed.get("server_instance_id")
    if not isinstance(instance, str) or not instance:
        return "FAIL", "loaded server instance is missing", None
    for expected, actual in (
        (client["model"], observed.get("model")),
        (client["serving_identity"], observed.get("serving_identity")),
        (client["session_library_sha256"], observed.get("session_library_sha256")),
    ):
        if expected != actual:
            return "FAIL", "loaded artifact identity differs from configuration", instance
    return "READY", "", instance


def _invoke(command: list[str], *, cwd: Path, log: Path, deadline: int | None) -> int:
    with log.open("wb") as output:
        child = subprocess.Popen(command, cwd=cwd, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return 124


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()


def run(config_path: Path, state_dir: Path) -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from server.harness_repeatability import _validate

    state_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    run_dir = state_dir / "runs" / (started.strftime("%Y%m%dT%H%M%S%fZ"))
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": SCHEMA, "source_commit": _source_commit(),
        "started_at": started.isoformat(), "status": "INCOMPLETE",
        "clients": [],
    }
    if not config_path.is_file():
        report["reason"] = "configuration unavailable"
        report["status"] = "NOT_TESTED"
    else:
        raw = config_path.read_bytes()
        report["config_sha256"] = hashlib.sha256(raw).hexdigest()
        config = _validate(json.loads(raw))
        if (state_dir / "PAUSE").exists():
            report["status"] = "NOT_TESTED"
            report["reason"] = "local harness nightly paused"
            report["clients"] = [{"name": item["name"], "status": "NOT_TESTED"}
                                 for item in config["clients"]]
        else:
            for client in config["clients"]:
                name = client["name"]
                status, reason, instance = _preflight(client)
                row = {"name": name, "status": status, "reason": reason,
                       "server_instance_id": instance}
                if status == "READY":
                    directory = run_dir / name
                    child_config = directory / "config.json"
                    _write(child_config, {
                        "schema": config["schema"], "attempts": config["attempts"],
                        "clients": [client],
                    })
                    outcome = directory / "matrix"
                    code = _invoke([
                        sys.executable, str(ROOT / "server/harness_repeatability.py"),
                        "--config", str(child_config), "--output-dir", str(outcome),
                    ], cwd=ROOT, log=directory / "run.log", deadline=None)
                    summary_path = outcome / "summary.json"
                    if code == 0 and summary_path.is_file():
                        reassess_code = _invoke([
                            sys.executable, str(ROOT / "server/harness_repeatability.py"),
                            "--reassess-dir", str(outcome),
                        ], cwd=ROOT, log=directory / "reassess.log", deadline=120)
                        reassess_path = outcome / "reassessment.json"
                        if reassess_code == 0 and reassess_path.is_file():
                            reassessed = json.loads(reassess_path.read_text())
                            original = json.loads(summary_path.read_text())
                            if (reassessed.get("status") == "pass"
                                    and original.get("status") == "pass"
                                    and original.get("source_commit") == report["source_commit"]):
                                row.update(status="PASS", reason="", steps=len(original.get("steps", [])))
                    if row["status"] != "PASS":
                        row.update(status="FAIL", reason="repeatability or reassessment failed",
                                   exit_code=code)
                report["clients"].append(row)
                _write(run_dir / "summary.json", report)
            statuses = {item["status"] for item in report["clients"]}
            report["status"] = ("FAIL" if "FAIL" in statuses else
                                "PASS" if statuses == {"PASS"} else "INCOMPLETE")
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write(run_dir / "summary.json", report)
    _write(state_dir / "latest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "LOCK").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another local harness nightly owns this state directory")
            return 2
        try:
            result = run(args.config, args.state_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"local harness nightly configuration or report error: {exc}")
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
