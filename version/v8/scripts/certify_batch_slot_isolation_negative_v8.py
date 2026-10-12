#!/usr/bin/env python3
"""Run the live slot-isolation probe and both parked-KV fault directions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PROBE = Path(__file__).with_name("xray_batch_slot_isolation_v8.py")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_files(bundle: Path) -> dict[str, str]:
    return {name: _sha256(bundle / name)
            for name in ("libmodel.so", "libckernel_engine.so", "weights.bump")}


def _run(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError("slot-isolation subprocess timed out") from exc


def certify(bundle: Path, *, layers: tuple[int, ...] = (0, 16, 17),
            threads: int = 1, timeout_seconds: int = 120,
            probe: Path = PROBE) -> dict:
    if timeout_seconds < 1 or threads < 1 or 0 not in layers:
        raise ValueError("negative control requires layer 0 and positive limits")
    bundle = bundle.resolve(strict=True)
    probe = probe.resolve(strict=True)
    bundle_files = _bundle_files(bundle)
    command = [sys.executable, str(probe), str(bundle), "--layers",
               ",".join(map(str, layers)), "--threads", str(threads)]
    with tempfile.TemporaryDirectory(prefix="cke_slot_negative_") as directory:
        control_path = Path(directory) / "control.json"
        control = _run(command + ["--json-out", str(control_path)], timeout_seconds)
        if control.returncode != 0 or not control_path.is_file():
            raise AssertionError("slot-isolation control run did not complete")
        report = json.loads(control_path.read_text())
        if (report.get("schema") != "cke.v8.batch-slot-isolation-xray.v1"
                or report.get("status") != "pass"
                or set(report.get("cases", {})) != {"0", "1"}):
            raise AssertionError("slot-isolation control report is incomplete")
        identity_keys = ("loaded_model_sha256", "loaded_engine_sha256",
                         "loaded_session_sha256", "generated_source_sha256",
                         "weights_sha256", "diagnostic_source_sha256")
        if any(not isinstance(report.get(key), str) or len(report[key]) != 64
               for key in identity_keys):
            raise AssertionError("slot-isolation control omitted artifact identity")
        if (report["loaded_model_sha256"] != bundle_files["libmodel.so"]
                or report["loaded_engine_sha256"] != bundle_files["libckernel_engine.so"]
                or report["weights_sha256"] != bundle_files["weights.bump"]
                or _bundle_files(bundle) != bundle_files):
            raise AssertionError("slot-isolation control artifact files changed")
        negative_cases = {}
        for advancing in (0, 1):
            parked = advancing ^ 1
            result = _run(command + ["--inject-parked-write", str(advancing)],
                          timeout_seconds)
            final_line = next((line.strip() for line in reversed(result.stderr.splitlines())
                               if line.strip()), "")
            expected = ("InjectedParkedWriteDetected: non-advancing slot "
                        f"{parked} logical KV changed")
            if result.returncode != 1 or final_line != expected:
                raise AssertionError(
                    f"parked-slot {parked} injection did not reach its logical-KV guard: "
                    f"exit={result.returncode}, last_error={final_line!r}"
                )
            if _bundle_files(bundle) != bundle_files:
                raise AssertionError("slot-isolation artifact files changed during injection")
            negative_cases[f"advancing_{advancing}/parked_{parked}"] = {
                "status": "detected", "exit_code": result.returncode,
                "guard_error": final_line,
            }
        return {
            "schema": "cke.v8.batch-slot-isolation-negative.v1",
            "status": "pass", "bundle": str(bundle), "layers": list(layers),
            "threads": threads, "timeout_seconds_per_run": timeout_seconds,
            "probe_sha256": _sha256(probe),
            "bundle_file_sha256": bundle_files,
            "control_report_sha256": _sha256(control_path),
            "control_artifact_identity": {key: report[key] for key in identity_keys},
            "control_cases": sorted(report["cases"]),
            "negative_cases": negative_cases,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--layers", default="0,16,17")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = certify(args.bundle, layers=tuple(map(int, args.layers.split(","))),
                     threads=args.threads, timeout_seconds=args.timeout_seconds)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
