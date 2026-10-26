#!/usr/bin/env python3
"""Measure pinned official Qwen3-TTS on retained technical text.

This is an asset-backed reference diagnostic, not CKE inference. It records
the raw talker stop decision before the official wrapper trims EOS, so a
capacity-limited waveform cannot be mistaken for natural completion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import resource
import statistics
import sys
import time

if __package__:
    from .capture_qwen3_tts_customvoice_oracle import (
        MODEL_REPO, MODEL_REVISION, REFERENCE_REPO, REFERENCE_REVISION,
        sha256_file, verify_model_dir, verify_reference_code,
    )
else:
    from capture_qwen3_tts_customvoice_oracle import (
        MODEL_REPO, MODEL_REVISION, REFERENCE_REPO, REFERENCE_REVISION,
        sha256_file, verify_model_dir, verify_reference_code,
    )

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = ROOT / "tests/fixtures/tts/accessibility_technical_text_v1.json"
REQUIRED_CATEGORIES = {
    "normal_prose", "path_and_signed_numbers", "signed_numbers", "decimals",
    "dates", "error_codes", "commands_and_filenames",
}


def load_corpus(path: Path) -> dict[str, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "cke.tts.accessibility_technical_text.v1":
        raise ValueError("unsupported accessibility corpus schema")
    cases: dict[str, dict] = {}
    categories = set()
    for case in document.get("cases", []):
        identifier = case.get("id")
        source = case.get("source_text")
        facts = case.get("critical_facts")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", identifier):
            raise ValueError("case ID must be a safe lower-case identifier")
        if identifier in cases or not isinstance(source, str) or not source.strip():
            raise ValueError(f"duplicate ID or empty source text: {identifier}")
        if not isinstance(facts, list) or not facts or any(
                not isinstance(fact, str) or not fact or fact not in source for fact in facts):
            raise ValueError(f"critical facts must occur literally in source text: {identifier}")
        expansion = case.get("spoken_expansion")
        if (expansion is not None and
                (not isinstance(expansion, str) or not expansion.strip() or
                 not isinstance(case.get("expansion_reason"), str) or
                 not case["expansion_reason"].strip())):
            raise ValueError(f"spoken expansion needs text and reason: {identifier}")
        if expansion is not None:
            spoken_facts = case.get("spoken_critical_facts")
            if not isinstance(spoken_facts, list) or len(spoken_facts) != len(facts) or any(
                    not isinstance(fact, str) or not fact or fact not in expansion
                    for fact in spoken_facts):
                raise ValueError(f"spoken critical facts must occur in expansion: {identifier}")
        categories.add(case.get("category"))
        cases[identifier] = case
    if not REQUIRED_CATEGORIES <= categories:
        raise ValueError(f"missing corpus categories: {sorted(REQUIRED_CATEGORIES - categories)}")
    return cases


def classify_termination(raw_sequence: list[int], eos_id: int,
                         max_new_tokens: int) -> str:
    if not raw_sequence:
        return "missing_sequence"
    if eos_id in raw_sequence:
        return "eos"
    if len(raw_sequence) >= max_new_tokens:
        return "capacity"
    return "unknown"


def run(model_dir: Path, corpus_path: Path, case_ids: list[str],
        output_dir: Path, policy: str, variant: str, seed: int,
        max_new_tokens: int, threads: int, repetitions: int,
        require_eos: bool) -> dict:
    if policy not in ("greedy", "sampled") or variant not in ("original", "expanded"):
        raise ValueError("unsupported policy or text variant")
    if max_new_tokens < 2 or max_new_tokens > 4096 or threads < 1 or \
            repetitions < 1 or repetitions > 20:
        raise ValueError("invalid generation budget, thread count, or repetition count")
    corpus = load_corpus(corpus_path)
    corpus_hash = sha256_file(corpus_path)
    selected = case_ids or list(corpus)
    if len(selected) != len(set(selected)) or any(name not in corpus for name in selected):
        raise ValueError("unknown or repeated case ID")
    if variant == "expanded" and any("spoken_expansion" not in corpus[name]
                                     for name in selected):
        raise ValueError("expanded variant requires an explicit expansion for every case")
    model_hashes = verify_model_dir(model_dir)

    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    import qwen_tts
    from qwen_tts import Qwen3TTSModel

    reference_hashes = verify_reference_code(Path(qwen_tts.__file__).resolve().parent)
    torch.set_num_threads(threads)
    load_started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(model_dir), device_map="cpu", dtype=torch.float32,
        attn_implementation="eager",
    )
    load_seconds = time.perf_counter() - load_started
    eos_id = json.loads((model_dir / "config.json").read_text())["talker_config"][
        "codec_eos_token_id"]
    original_talker_generate = model.model.talker.generate
    original_model_generate = model.model.generate
    captured: dict[str, object] = {}

    def record_talker(*args, **kwargs):
        result = original_talker_generate(*args, **kwargs)
        if not hasattr(result, "sequences") or result.sequences.shape[0] != 1:
            raise RuntimeError("reference did not expose one raw talker sequence")
        captured["raw_sequence"] = result.sequences[0].detach().cpu().tolist()
        return result

    def record_codes(*args, **kwargs):
        result = original_model_generate(*args, **kwargs)
        if len(result[0]) != 1:
            raise RuntimeError("reference did not expose one codec sequence")
        captured["codes"] = result[0][0].detach().cpu().numpy().copy()
        return result

    model.model.talker.generate = record_talker
    model.model.generate = record_codes
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for name, iteration in ((name, iteration) for name in selected
                                for iteration in range(1, repetitions + 1)):
            case = corpus[name]
            spoken = (case["source_text"] if variant == "original" else
                      case["spoken_expansion"])
            torch.manual_seed(seed)
            np.random.seed(seed)
            captured.clear()
            started = time.perf_counter()
            with torch.inference_mode():
                wavs, sample_rate = model.generate_custom_voice(
                    text=spoken, speaker="Ryan", language="English",
                    do_sample=policy == "sampled",
                    subtalker_dosample=policy == "sampled",
                    max_new_tokens=max_new_tokens, non_streaming_mode=True,
                )
            generate_seconds = time.perf_counter() - started
            if len(wavs) != 1 or "raw_sequence" not in captured or "codes" not in captured:
                raise RuntimeError(f"incomplete reference capture: {name}")
            waveform = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            codes = np.asarray(captured["codes"], dtype=np.int32)
            raw_sequence = captured["raw_sequence"]
            if not waveform.size or not np.isfinite(waveform).all() or \
                    codes.ndim != 2 or codes.shape[1] != 16 or sample_rate != 24000:
                raise RuntimeError(f"invalid reference audio or codes: {name}")
            prefix = f"{name}-{variant}-{policy}"
            if repetitions > 1:
                prefix += f"-run{iteration}"
            wav_path = output_dir / f"{prefix}.wav"
            codes_path = output_dir / f"{prefix}-codes.npy"
            sf.write(wav_path, waveform, sample_rate, subtype="PCM_16")
            np.save(codes_path, codes)
            termination = classify_termination(raw_sequence, eos_id, max_new_tokens)
            item = {
                "id": name, "iteration": iteration,
                "category": case["category"],
                "source_text": case["source_text"], "spoken_text": spoken,
                "text_variant": variant,
                "text_transformed": spoken != case["source_text"],
                "expansion_reason": case.get("expansion_reason") if variant == "expanded" else None,
                "critical_facts": case["critical_facts"],
                "termination": termination, "raw_talker_tokens": len(raw_sequence),
                "raw_sequence_tail": raw_sequence[-12:], "codec_eos_id": eos_id,
                "codec_frames": int(codes.shape[0]),
                "first_code_tail": codes[-12:, 0].tolist(),
                "audio_samples": int(waveform.size), "sample_rate": sample_rate,
                "audio_seconds": waveform.size / sample_rate,
                "generate_seconds": generate_seconds,
                "real_time_factor": generate_seconds / (waveform.size / sample_rate),
                "waveform_peak": float(np.max(np.abs(waveform))),
                "waveform_rms": float(np.sqrt(np.mean(waveform * waveform))),
                "wav_file": wav_path.name, "wav_sha256": sha256_file(wav_path),
                "codes_file": codes_path.name, "codes_sha256": sha256_file(codes_path),
                "human_listening": "NOT_TESTED", "asr_fidelity": "NOT_TESTED",
                "cke_execution": "NOT_TESTED", "cke_numerical_parity": "NOT_TESTED",
            }
            results.append(item)
            print(json.dumps({"case": name, "iteration": iteration,
                              "termination": termination,
                              "codec_frames": item["codec_frames"],
                              "generate_seconds": round(generate_seconds, 3)},
                             sort_keys=True), flush=True)
    finally:
        model.model.talker.generate = original_talker_generate
        model.model.generate = original_model_generate
    if sha256_file(corpus_path) != corpus_hash:
        raise RuntimeError("accessibility corpus changed during reference execution")
    repeated_timing = {}
    for name in selected:
        times = [item["generate_seconds"] for item in results if item["id"] == name]
        repeated_timing[name] = {
            "runs": len(times), "mean_seconds": statistics.mean(times),
            "population_stddev_seconds": statistics.pstdev(times),
            "min_seconds": min(times), "max_seconds": max(times),
            "wav_hashes_identical": len({item["wav_sha256"] for item in results
                                         if item["id"] == name}) == 1,
        }
    report = {
        "schema": "cke.qwen3_tts.accessibility_reference.v1",
        "scope": "pinned official PyTorch reference; CKE inference NOT_TESTED",
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION,
                  "files_sha256": model_hashes},
        "reference": {"repo": REFERENCE_REPO, "revision": REFERENCE_REVISION,
                      "files_sha256": reference_hashes},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "numpy": np.__version__, "device": "cpu", "dtype": "float32",
                    "attention": "eager", "threads": threads,
                    "machine": platform.machine(),
                    "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
        "request": {"policy": policy, "variant": variant, "seed": seed,
                    "max_new_tokens": max_new_tokens, "speaker": "Ryan",
                    "language": "English", "non_streaming_mode": True,
                    "repetitions": repetitions,
                    "first_playable_audio_seconds": "NOT_MEASURED_NON_STREAMING"},
        "load_seconds": load_seconds,
        "repeated_timing": repeated_timing,
        "corpus_sha256": corpus_hash,
        "results": results,
        "all_natural_eos": all(item["termination"] == "eos" for item in results),
    }
    report_path = output_dir / "reference-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if require_eos and not report["all_natural_eos"]:
        raise RuntimeError(f"one or more cases hit a generation boundary; see {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=("greedy", "sampled"), default="sampled")
    parser.add_argument("--variant", choices=("original", "expanded"), default="original")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat each selected case with the same seed on the loaded model")
    parser.add_argument("--require-eos", action="store_true")
    args = parser.parse_args()
    report = run(args.model_dir, args.corpus, args.case, args.output_dir,
                 args.policy, args.variant, args.seed, args.max_new_tokens,
                 args.threads, args.repeat, args.require_eos)
    print(json.dumps({"report": str(args.output_dir / "reference-report.json"),
                      "all_natural_eos": report["all_natural_eos"],
                      "cases": len(report["results"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
