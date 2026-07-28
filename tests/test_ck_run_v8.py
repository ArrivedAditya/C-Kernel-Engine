#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "ck_run_v8.py"


def _load_module():
    scripts = str(SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("ck_run_v8_tests", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ck_run_v8 = _load_module()


def test_refresh_manifest_circuit_snapshot_replaces_stale_graph_policy(
    tmp_path: Path, monkeypatch
) -> None:
    v8_root = tmp_path / "v8"
    circuits = v8_root / "circuits"
    circuits.mkdir(parents=True)
    current = {
        "name": "fixture",
        "version": 2,
        "kernels": {"attn_decode": "cache_aware_decode"},
    }
    (circuits / "fixture.json").write_text(json.dumps(current), encoding="utf-8")
    manifest_path = tmp_path / "weights_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config": {"model": "fixture"},
                "template": {
                    "name": "fixture",
                    "version": 1,
                    "kernels": {"attn": "stale_provider"},
                },
                "entries": [{"name": "weight", "offset": 0}],
            }
        ),
        encoding="utf-8",
    )
    original_entries = manifest_path_data(manifest_path)["entries"]
    monkeypatch.setattr(ck_run_v8, "V8_ROOT", v8_root)

    assert ck_run_v8._refresh_manifest_circuit_snapshot(manifest_path)
    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert refreshed["template"] == current
    assert refreshed["entries"] == original_entries
    assert not ck_run_v8._refresh_manifest_circuit_snapshot(manifest_path)


def manifest_path_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
