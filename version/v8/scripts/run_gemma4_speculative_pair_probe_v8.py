#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def _find_runtime_lib(run_dir: Path) -> Path:
    for name in ("libmodel.so", "ck-kernel-inference.so", "ck-kernel-decode.so"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no generated runtime library found in {run_dir}")


def _find_weights(run_dir: Path) -> Path:
    candidate = run_dir / "weights.bump"
    if not candidate.exists():
        raise FileNotFoundError(f"weights.bump not found in {run_dir}")
    return candidate


class RuntimeProbe:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.lib_path = _find_runtime_lib(run_dir)
        self.weights_path = _find_weights(run_dir)
        self.lib = ctypes.CDLL(str(self.lib_path))
        self.initialized = False

        self.lib.ck_model_init.argtypes = [ctypes.c_char_p]
        self.lib.ck_model_init.restype = ctypes.c_int
        self.lib.ck_model_free.argtypes = []
        self.lib.ck_model_free.restype = None

        self.has_named_activations = False
        try:
            self.lib.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
            self.lib.ck_model_get_named_activation_ptr.restype = ctypes.c_void_p
            self.lib.ck_model_get_named_activation_nbytes.argtypes = [ctypes.c_char_p]
            self.lib.ck_model_get_named_activation_nbytes.restype = ctypes.c_ssize_t
            self.lib.ck_model_get_named_activation_runtime_offset.argtypes = [ctypes.c_char_p]
            self.lib.ck_model_get_named_activation_runtime_offset.restype = ctypes.c_ssize_t
            self.has_named_activations = True
        except AttributeError:
            self.has_named_activations = False

    def init(self) -> None:
        rc = int(self.lib.ck_model_init(str(self.weights_path).encode()))
        if rc != 0:
            raise RuntimeError(f"ck_model_init failed for {self.run_dir} with code {rc}")
        self.initialized = True

    def close(self) -> None:
        if self.initialized:
            self.lib.ck_model_free()
            self.initialized = False

    def activation_info(self, name: str) -> dict[str, Any]:
        if not self.has_named_activations:
            return {"name": name, "present": False, "reason": "runtime lacks named activation API"}
        b_name = name.encode()
        nbytes = int(self.lib.ck_model_get_named_activation_nbytes(b_name))
        offset = int(self.lib.ck_model_get_named_activation_runtime_offset(b_name))
        ptr = int(self.lib.ck_model_get_named_activation_ptr(b_name) or 0)
        return {
            "name": name,
            "present": bool(nbytes > 0 and ptr != 0),
            "nbytes": nbytes,
            "runtime_offset": offset,
            "ptr_nonzero": bool(ptr != 0),
        }


def _load_engine_lib() -> ctypes.CDLL:
    candidates = (
        REPO_ROOT / "build" / "libckernel_engine.so",
        REPO_ROOT / "libckernel_engine.so",
    )
    for path in candidates:
        if path.exists():
            lib = ctypes.CDLL(str(path))
            lib.speculative_verify_greedy_f32.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.speculative_verify_greedy_f32.restype = None
            lib.speculative_commit_one_i32.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.speculative_commit_one_i32.restype = None
            return lib
    raise FileNotFoundError("build/libckernel_engine.so not found; run make build/libckernel_engine.so")


def run_kernel_smoke() -> dict[str, Any]:
    lib = _load_engine_lib()
    vocab = 16
    logits = (ctypes.c_float * vocab)(*[0.0] * vocab)
    logits[5] = 10.0
    accepted = ctypes.c_int(-1)
    verified = ctypes.c_int(-1)
    lib.speculative_verify_greedy_f32(logits, vocab, 5, ctypes.byref(accepted), ctypes.byref(verified))
    accept_case = (accepted.value, verified.value)
    lib.speculative_verify_greedy_f32(logits, vocab, 3, ctypes.byref(accepted), ctypes.byref(verified))
    reject_case = (accepted.value, verified.value)

    token_buf = (ctypes.c_int * 4)(-1, -1, -1, -1)
    token_count = ctypes.c_int(0)
    target_pos = ctypes.c_int(0)
    draft_pos = ctypes.c_int(0)
    accepted_count = ctypes.c_int(0)
    rejected_count = ctypes.c_int(0)
    lib.speculative_commit_one_i32(
        1,
        5,
        token_buf,
        ctypes.byref(token_count),
        4,
        ctypes.byref(target_pos),
        ctypes.byref(draft_pos),
        ctypes.byref(accepted_count),
        ctypes.byref(rejected_count),
    )
    lib.speculative_commit_one_i32(
        0,
        7,
        token_buf,
        ctypes.byref(token_count),
        4,
        ctypes.byref(target_pos),
        ctypes.byref(draft_pos),
        ctypes.byref(accepted_count),
        ctypes.byref(rejected_count),
    )
    passed = (
        accept_case == (1, 5)
        and reject_case == (0, 5)
        and list(token_buf)[:2] == [5, 7]
        and token_count.value == 2
        and target_pos.value == 2
        and draft_pos.value == 2
        and accepted_count.value == 1
        and rejected_count.value == 1
    )
    return {
        "passed": passed,
        "accept_case": list(accept_case),
        "reject_case": list(reject_case),
        "tokens": list(token_buf),
        "token_count": token_count.value,
        "target_position": target_pos.value,
        "draft_position": draft_pos.value,
        "accepted_count": accepted_count.value,
        "rejected_count": rejected_count.value,
    }


def probe_pair(target_run: Path | None, draft_run: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {"kernel_smoke": run_kernel_smoke()}
    probes: list[RuntimeProbe] = []
    try:
        if target_run is not None:
            target = RuntimeProbe(target_run)
            probes.append(target)
            target.init()
            out["target"] = {
                "run_dir": str(target_run),
                "lib": str(target.lib_path),
                "has_named_activations": target.has_named_activations,
                "candidate_hidden_exports": [
                    target.activation_info(name)
                    for name in ("target_hidden_stream", "main_stream", "layer_output")
                ],
            }
        if draft_run is not None:
            draft = RuntimeProbe(draft_run)
            probes.append(draft)
            draft.init()
            out["draft"] = {
                "run_dir": str(draft_run),
                "lib": str(draft.lib_path),
                "has_named_activations": draft.has_named_activations,
                "candidate_backbone_inputs": [
                    draft.activation_info(name)
                    for name in ("draft_backbone_stream", "backbone_stream")
                ],
            }
    finally:
        for probe in reversed(probes):
            probe.close()
    out["passed"] = bool(out["kernel_smoke"]["passed"])
    if target_run is not None:
        out["passed"] = out["passed"] and any(
            item.get("present") for item in out.get("target", {}).get("candidate_hidden_exports", [])
        )
    if draft_run is not None:
        out["passed"] = out["passed"] and any(
            item.get("present") for item in out.get("draft", {}).get("candidate_backbone_inputs", [])
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe Gemma4 target + assistant speculative runtime bridge readiness.")
    ap.add_argument("--target-run", type=Path, help="Generated Gemma4 backbone runtime directory.")
    ap.add_argument("--draft-run", type=Path, help="Generated Gemma4 assistant runtime directory.")
    ap.add_argument("--kernel-smoke-only", action="store_true", help="Only validate speculative verifier/commit kernels.")
    args = ap.parse_args()

    if args.kernel_smoke_only:
        result = {"passed": True, "kernel_smoke": run_kernel_smoke()}
        result["passed"] = bool(result["kernel_smoke"]["passed"])
    else:
        result = probe_pair(args.target_run, args.draft_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
