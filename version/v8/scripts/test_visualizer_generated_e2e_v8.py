#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "version" / "v7" / "scripts" / "test_visualizer_generated_e2e_v7.py"

XRAY_FIXTURES = ROOT / "version" / "v8" / "tests" / "fixtures" / "xray"
XRAY_EXPECTED_KEYS = [
    "xray_whisper_encoder",
    "xray_ranking",
    "xray_execution_trace",
    "xray_execution_state",
    "xray_decoder_pytorch",
    "xray_qwen3vl_pytorch",
    "xray_qwen3vl_llamacpp",
]
XRAY_EXPECTED_SCHEMAS = [
    "cke.whisper_encoder_pytorch_xray",
    "cke.xray_ranking_report",
    "cke.xray_execution_trace",
    "cke.xray_execution_state_report",
    "cke.xray.decoder_pytorch",
    "cke.xray_orchestration_report",
]
XRAY_RUNBOOK_MARKERS = [
    "No X-ray artifacts loaded",
    "compare_whisper_encoder_pytorch_v8.py",
    "xray_execution_state_v8.py",
]


def _generate(open_viz: Path, run_dir: Path, out: Path) -> str | None:
    cmd = [
        sys.executable, str(open_viz),
        "--generate", "--run", str(run_dir),
        "--html-only", "--output", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not out.exists():
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        return None
    return out.read_text(encoding="utf-8")


def run_xray_stage() -> int:
    """v8-only stage: fixture X-ray artifacts must embed into the generated report,
    and the X-Ray tab must render its empty-state runbook without them."""
    open_viz = ROOT / "version" / "v8" / "tools" / "open_ir_visualizer_v8.py"
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="ck-viz-xray-") as tmp:
        # 1) Run dir populated with fixture X-ray artifacts -> keys embedded.
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        for fixture in sorted(XRAY_FIXTURES.glob("*.json")):
            shutil.copy(fixture, run_dir / fixture.name)
        html = _generate(open_viz, run_dir, run_dir / "ir_report.html")
        if html is None:
            failures.append("generate_with_fixtures")
        else:
            for key in XRAY_EXPECTED_KEYS:
                if f'"{key}"' not in html:
                    failures.append(f"missing_key:{key}")
            for marker in XRAY_EXPECTED_SCHEMAS:
                if marker not in html:
                    failures.append(f"missing_schema:{marker}")

        # 2) Empty run dir -> empty-state runbook commands present.
        empty_dir = Path(tmp) / "empty"
        empty_dir.mkdir()
        html_empty = _generate(open_viz, empty_dir, empty_dir / "ir_report.html")
        if html_empty is None:
            failures.append("generate_empty")
        else:
            for marker in XRAY_RUNBOOK_MARKERS:
                if marker not in html_empty:
                    failures.append(f"missing_runbook:{marker}")

    if failures:
        for f in failures:
            print(f"  ✗ {f}")
        print(f"L3_xray_embed  max_diff={len(failures):.2e}  tol=1e+00  [FAIL]")
        return 1
    print(f"  ✓ xray embed: {len(XRAY_EXPECTED_KEYS)} keys + {len(XRAY_EXPECTED_SCHEMAS)} schemas embedded from fixtures")
    print(f"  ✓ xray empty-state runbook: {len(XRAY_RUNBOOK_MARKERS)} markers present")
    print("L3_xray_embed  max_diff=0.00e+00  tol=1e+00  [PASS]")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CK_VIS_VERSION", "v8")
    os.environ.setdefault("CK_VIS_MODELS_ROOT", str(Path.home() / ".cache" / "ck-engine-v8" / "models"))
    os.environ.setdefault("CK_VIS_HEALTH_SCRIPT", str(ROOT / "version" / "v8" / "scripts" / "test_visualizer_health_v8.py"))
    os.environ.setdefault("CK_VIS_OPEN_IR_VIZ", str(ROOT / "version" / "v8" / "tools" / "open_ir_visualizer_v8.py"))
    os.environ.setdefault("CK_VIS_PREPARE_VIEWER", str(ROOT / "version" / "v8" / "tools" / "prepare_run_viewer_v8.py"))
    os.environ.setdefault("CK_VIS_OPEN_IR_HUB", str(ROOT / "version" / "v8" / "tools" / "open_ir_hub_v8.py"))
    sys.argv[0] = str(Path(__file__).resolve())

    base_code = 0
    try:
        runpy.run_path(str(TARGET), run_name="__main__")
    except SystemExit as exc:
        if isinstance(exc.code, int):
            base_code = exc.code
        elif exc.code is not None:
            base_code = 1

    xray_code = run_xray_stage()
    sys.exit(0 if (base_code == 0 and xray_code == 0) else 1)
