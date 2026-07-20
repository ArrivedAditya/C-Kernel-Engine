#!/usr/bin/env python3
"""Resume-safe CK BF16 versus PyTorch Qwen3-VL generation certification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
import time
import traceback
from array import array
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_bridge_module():
    path = SCRIPT_DIR / "run_multimodal_bridge_v8.py"
    spec = importlib.util.spec_from_file_location("cke_bf16_corpus_bridge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load bridge module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _custom_prefix_report(prefix_path: Path) -> tuple[Path, dict[str, Any]]:
    candidates = (
        prefix_path.with_suffix(prefix_path.suffix + ".json"),
        prefix_path.parent / "report.json",
        prefix_path.parent.parent / "report.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), _read_json(candidate)
    raise RuntimeError(
        "custom prefix requires a nearby provenance report: expected "
        "<prefix>.json, <prefix-dir>/report.json, or <prefix-parent>/report.json"
    )


def _validate_custom_prefix_provenance(
    prefix_path: Path,
    sample_image: Path,
    expected_shape: tuple[int, int],
    expected_grid: tuple[int, int, int],
) -> Path:
    report_path, report = _custom_prefix_report(prefix_path)
    reported_image = report.get("image")
    if not reported_image:
        raise RuntimeError(f"custom prefix report has no source image: {report_path}")
    source_image = Path(str(reported_image)).resolve()
    if not source_image.is_file():
        raise RuntimeError(f"custom prefix source image is unavailable: {source_image}")
    source_sha = _sha256(source_image)
    sample_sha = _sha256(sample_image)
    if source_sha != sample_sha:
        raise RuntimeError(
            "custom prefix image mismatch: "
            f"prefix_image_sha256={source_sha} sample_image_sha256={sample_sha}"
        )

    tensor = ((report.get("torch") or {}).get("tensors") or {}).get("vision_output") or {}
    reported_shape = tuple(int(value) for value in tensor.get("shape") or ())
    if reported_shape != expected_shape:
        raise RuntimeError(
            f"custom prefix logical shape mismatch: got={reported_shape} expected={expected_shape}"
        )
    grid_rows = (report.get("torch") or {}).get("grid_thw") or []
    reported_grid = tuple(int(value) for value in (grid_rows[0] if grid_rows else ()))
    if reported_grid != expected_grid:
        raise RuntimeError(
            f"custom prefix visual grid mismatch: got={reported_grid} expected={expected_grid}"
        )
    return report_path


def _runtime_from_dir(path: Path, *, encoder: bool) -> dict[str, Any]:
    if encoder:
        libraries = sorted(path.glob("lib*qwen3vl*encoder*.so"))
        layouts = sorted(path.glob("layout.json"))
    else:
        libraries = sorted(path.glob("libmodel.so"))
        layouts = sorted(path.glob("layout_decode.json"))
    if len(libraries) != 1 or len(layouts) != 1:
        raise RuntimeError(f"runtime artifacts are ambiguous in {path}")
    layout = _read_json(layouts[0])
    config = dict(layout.get("config") or {})
    adjacent_engine = path / "libckernel_engine.so"
    canonical_engine = Path(
        os.environ.get("CK_CERT_ENGINE_SO", str(adjacent_engine))
    ).resolve()
    runtime = {
        "so_path": libraries[0],
        "engine_so": canonical_engine,
        "weights_bump": path / "weights.bump",
        "manifest_map": path / "weights_manifest.map",
        "layout_path": layouts[0],
        "context_length": int(config.get("context_length", 0) or 0),
        "embed_dim": int(config.get("embed_dim", 0) or 0),
        "vocab_size": int(config.get("vocab_size", 0) or 0),
        "config": config,
    }
    if not adjacent_engine.is_file() or not canonical_engine.is_file():
        raise RuntimeError(
            "runtime engine provenance is incomplete: "
            f"adjacent={adjacent_engine} canonical={canonical_engine}"
        )
    if _sha256(adjacent_engine) != _sha256(canonical_engine):
        raise RuntimeError(
            "runtime adjacent and canonical engines differ: "
            f"adjacent={adjacent_engine} canonical={canonical_engine}"
        )
    return runtime


def _encoder_grid(runtime: dict[str, Any]) -> tuple[int, int, int]:
    layout = _read_json(Path(runtime["layout_path"]))
    config = dict(layout.get("config") or {})
    return (
        int(config.get("vision_temporal_grid", 1) or 1),
        int(config.get("vision_grid_h", 0) or 0),
        int(config.get("vision_grid_w", 0) or 0),
    )


def _pre_eos(tokens: list[int], eos_ids: set[int]) -> list[int]:
    for index, token in enumerate(tokens):
        if int(token) in eos_ids:
            return [int(value) for value in tokens[:index]]
    return [int(value) for value in tokens]


def _first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _prefix_metrics(torch: Any, ck_values: array, reference: Any) -> dict[str, float | int]:
    ck = torch.frombuffer(ck_values, dtype=torch.float32).clone().reshape(reference.shape)
    ref = reference.float().cpu().contiguous()
    delta = ck - ref
    rmse = float(torch.sqrt(torch.mean(delta * delta)).item())
    reference_rms = float(torch.sqrt(torch.mean(ref * ref)).item())
    raw_cosine = float(
        torch.nn.functional.cosine_similarity(ck.reshape(1, -1), ref.reshape(1, -1)).item()
    )
    cosine = max(-1.0, min(1.0, raw_cosine))
    return {
        "cosine": cosine,
        "rmse": rmse,
        "relative_rmse": rmse / max(reference_rms, 1.0e-30),
        "max_abs": float(torch.max(torch.abs(delta)).item()),
        "exact_elements": int(torch.count_nonzero(ck == ref).item()),
        "elements": int(ref.numel()),
    }


def _load_samples(manifest: Path, limit: int, selected: set[int]) -> list[dict[str, Any]]:
    payload = _read_json(manifest)
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(payload.get("samples") or []):
        if selected and index not in selected:
            continue
        if not isinstance(sample, dict):
            continue
        inputs = sample.get("inputs") or []
        if not inputs or not isinstance(inputs[0], dict) or not inputs[0].get("path"):
            continue
        image = (manifest.parent / str(inputs[0]["path"])).resolve()
        rows.append({"index": index, "id": str(sample.get("id") or index), "image": image})
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def _decoder_worker(request_path: str, response_path: str) -> None:
    try:
        request = _read_json(Path(request_path))
        bridge = _load_bridge_module()
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            request["checkpoint"], local_files_only=True
        )
        runtime = _runtime_from_dir(Path(request["decoder_runtime"]), encoder=False)
        prefix = array("f")
        expected = int(request["prefix_tokens"]) * int(request["prefix_embed_dim"])
        with Path(request["prefix_f32"]).open("rb") as handle:
            prefix.fromfile(handle, expected)
            if handle.read(1):
                raise RuntimeError("decoder worker prefix has trailing data")
        if len(prefix) != expected:
            raise RuntimeError(
                f"decoder worker prefix count mismatch: got={len(prefix)} expected={expected}"
            )
        report = bridge._run_decoder(
            runtime,
            prefix,
            int(request["prefix_tokens"]),
            [int(value) for value in request["tokens_after"]],
            tokens_before=[int(value) for value in request["tokens_before"]],
            prefix_embed_dim=int(request["prefix_embed_dim"]),
            prefix_grid=tuple(int(value) for value in request["prefix_grid"]),
            prefix_text_pos=int(request["prefix_text_pos"]),
            prefix_decode_policy=str(request["prefix_decode_policy"]),
            bridge_runtime="decode-staged",
            bridge_generation_mode="incremental-decode",
            tokenizer=processor.tokenizer,
            stop_token_ids=[int(value) for value in request["stop_token_ids"]],
            max_tokens=int(request["max_tokens"]),
            temperature=0.0,
            sample_top_k=40,
            top_p=1.0,
            min_p=0.0,
            stream_output=False,
            forced_generation_token_ids=(
                [int(value) for value in request["forced_generation_token_ids"]]
                if request.get("forced_generation_token_ids") is not None else None
            ),
            generation_trace_top_k=int(request["trace_top_k"]),
        )
        serializable_report = _serializable_decoder_report(report)
        _write_json(
            Path(response_path), {"status": "ok", "report": serializable_report}
        )
    except BaseException as error:
        _write_json(
            Path(response_path),
            {"status": "error", "error": str(error), "traceback": traceback.format_exc()},
        )
        raise


def _serializable_decoder_report(report: dict[str, Any]) -> dict[str, Any]:
    timings = report.get("timings")
    return {
        "generated_token_ids": [
            int(value) for value in report.get("generated_token_ids", [])
        ],
        "teacher_forced_input_ids": [
            int(value) for value in report.get("teacher_forced_input_ids", [])
        ],
        "generation_logit_trace": report.get("generation_logit_trace", []),
        "timings": dict(timings) if isinstance(timings, dict) else {},
    }


def _run_decoder_isolated(
    *,
    request: dict[str, Any],
    prefix_embeddings: array,
    request_path: Path,
    response_path: Path,
) -> dict[str, Any]:
    prefix_path = request_path.with_suffix(".prefix.f32")
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    with prefix_path.open("wb") as handle:
        prefix_embeddings.tofile(handle)
    request = {**request, "prefix_f32": str(prefix_path.resolve())}
    _write_json(request_path, request)
    response_path.unlink(missing_ok=True)
    process = multiprocessing.get_context("spawn").Process(
        target=_decoder_worker,
        args=(str(request_path.resolve()), str(response_path.resolve())),
    )
    process.start()
    process.join()
    try:
        if not response_path.is_file():
            raise RuntimeError(
                f"decoder worker exited without a response: exitcode={process.exitcode}"
            )
        response = _read_json(response_path)
        if process.exitcode != 0 or response.get("status") != "ok":
            raise RuntimeError(
                "decoder worker failed: "
                f"exitcode={process.exitcode} error={response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        report = response.get("report")
        if not isinstance(report, dict):
            raise RuntimeError("decoder worker returned no report")
        return report
    finally:
        prefix_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder-runtime", type=Path, action="append", required=True)
    parser.add_argument("--encoder-weights-bump", type=Path, required=True)
    parser.add_argument("--decoder-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="Extract visible form fields as compact JSON.")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--sample-index", type=int, action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    parser.add_argument("--ck-prefix-source", choices=("ck", "pytorch"), default="ck")
    parser.add_argument(
        "--ck-prefix-f32",
        type=Path,
        help="Use one exact FP32 prefix artifact for a bounded single-sample causal test",
    )
    parser.add_argument(
        "--dump-torch-prefix-f32",
        type=Path,
        help="Persist the exact PyTorch FP32 visual prefix before CK decoder execution",
    )
    parser.add_argument("--teacher-force-pytorch", action="store_true")
    parser.add_argument("--trace-top-k", type=int, default=0)
    parser.add_argument(
        "--decoder-process",
        choices=("spawn", "in-process"),
        default="spawn",
        help="Run CK decode in a clean process so the reference backend cannot alter runtime state",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ["CK_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    bridge = _load_bridge_module()
    torch.set_num_threads(args.threads)
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    result_dir = output_dir / "images"
    result_dir.mkdir(parents=True, exist_ok=True)

    encoder_runtimes: dict[tuple[int, int, int], dict[str, Any]] = {}
    for runtime_dir in args.encoder_runtime:
        runtime = _runtime_from_dir(runtime_dir.resolve(), encoder=True)
        runtime["weights_bump"] = args.encoder_weights_bump.resolve()
        grid = _encoder_grid(runtime)
        if grid in encoder_runtimes:
            raise RuntimeError(f"duplicate encoder geometry {grid}")
        encoder_runtimes[grid] = runtime
    decoder_runtime = _runtime_from_dir(args.decoder_runtime.resolve(), encoder=False)

    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(checkpoint), torch_dtype=torch.bfloat16, local_files_only=True
    )
    model.eval()

    eos = model.generation_config.eos_token_id
    eos_ids = {int(value) for value in (eos if isinstance(eos, list) else [eos]) if value is not None}
    samples = _load_samples(args.manifest.resolve(), args.max_images, set(args.sample_index))
    if args.ck_prefix_f32 is not None:
        if args.ck_prefix_source != "ck":
            parser.error("--ck-prefix-f32 cannot be combined with --ck-prefix-source=pytorch")
        if len(samples) != 1:
            parser.error("--ck-prefix-f32 requires exactly one selected sample")
    completed: list[dict[str, Any]] = []

    for ordinal, sample in enumerate(samples, start=1):
        result_path = result_dir / f"{int(sample['index']):03d}.json"
        if result_path.exists() and not args.force:
            retained = _read_json(result_path)
            completed.append(retained)
            print(f"[{ordinal}/{len(samples)}] retained {sample['id']}: {retained.get('status')}", flush=True)
            continue

        image_path = Path(sample["image"])
        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": args.prompt},
            ],
        }]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prompt_text], images=[image], return_tensors="pt",
            min_pixels=1, max_pixels=args.image_max_pixels,
        )
        grid = tuple(int(value) for value in inputs["image_grid_thw"][0].tolist())
        runtime = encoder_runtimes.get(grid)
        if runtime is None:
            raise RuntimeError(f"no CK encoder runtime for image grid {grid}: {image_path}")

        input_ids = [int(value) for value in inputs["input_ids"][0].tolist()]
        image_token_id = int(model.config.image_token_id)
        image_indices = [index for index, token in enumerate(input_ids) if token == image_token_id]
        if not image_indices:
            raise RuntimeError(f"processor emitted no image tokens for {image_path}")
        first_image = image_indices[0]
        last_image = image_indices[-1]
        before_ids = input_ids[:first_image]
        after_ids = input_ids[last_image + 1:]

        torch_t0 = time.perf_counter()
        with torch.inference_mode():
            generated_result = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                return_dict_in_generate=bool(args.trace_top_k),
                output_scores=bool(args.trace_top_k),
            )
        generated = generated_result.sequences if args.trace_top_k else generated_result
        torch_sec = time.perf_counter() - torch_t0
        torch_ids_raw = [int(value) for value in generated[0, len(input_ids):].tolist()]
        torch_ids = _pre_eos(torch_ids_raw, eos_ids)
        torch_logit_trace = []
        if args.trace_top_k:
            for step, scores in enumerate(generated_result.scores):
                values, indices = torch.topk(scores[0].float(), k=int(args.trace_top_k))
                torch_logit_trace.append({
                    "step": int(step),
                    "top_k": [
                        {"token_id": int(token_id), "logit": float(logit)}
                        for token_id, logit in zip(indices.tolist(), values.tolist())
                    ],
                })

        ck_encoder_t0 = time.perf_counter()
        from compare_qwen3vl_bf16_vision_hidden_v8 import _qwen3vl_processor_pixels_to_planar
        runtime_config = runtime["config"]
        vision_config = model.config.vision_config
        planar = _qwen3vl_processor_pixels_to_planar(
            inputs["pixel_values"].float().cpu().numpy(),
            grid,
            patch_size=int(vision_config.patch_size),
            temporal_patch_size=int(vision_config.temporal_patch_size),
            height=int(runtime_config["image_height"]),
            width=int(runtime_config["image_width"]),
            merge_size=int(vision_config.spatial_merge_size),
        )
        encoder_report = bridge._run_encoder(
            runtime, "processor-planar", image_path=image_path, planar_override=planar
        )
        ck_encoder_sec = time.perf_counter() - ck_encoder_t0
        prefix_embeddings = encoder_report["embeddings"]
        torch_prefix_t0 = time.perf_counter()
        with torch.inference_mode():
            image_embeds, deepstack = model.get_image_features(
                inputs["pixel_values"], inputs["image_grid_thw"]
            )
        final = image_embeds[0]
        prefix = torch.cat([final, *deepstack], dim=-1).float().cpu().contiguous()
        if args.dump_torch_prefix_f32 is not None:
            if len(samples) != 1:
                parser.error("--dump-torch-prefix-f32 requires exactly one selected sample")
            dump_path = args.dump_torch_prefix_f32.resolve()
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_bytes(prefix.numpy().tobytes(order="C"))
        torch_prefix_sec = time.perf_counter() - torch_prefix_t0
        if tuple(prefix.shape) != (
            int(encoder_report["prefix_tokens"]), int(encoder_report["embed_dim"])
        ):
            raise RuntimeError(
                f"PyTorch prefix shape {tuple(prefix.shape)} does not match CK bridge shape "
                f"{(encoder_report['prefix_tokens'], encoder_report['embed_dim'])}"
            )
        prefix_source = args.ck_prefix_source
        custom_prefix_report = None
        if args.ck_prefix_f32 is not None:
            custom_prefix_report = _validate_custom_prefix_provenance(
                args.ck_prefix_f32.resolve(),
                image_path,
                (int(encoder_report["prefix_tokens"]), int(encoder_report["embed_dim"])),
                tuple(int(value) for value in grid),
            )
            expected_count = int(encoder_report["prefix_tokens"]) * int(encoder_report["embed_dim"])
            imported = array("f")
            with args.ck_prefix_f32.resolve().open("rb") as handle:
                imported.fromfile(handle, expected_count)
                if handle.read(1):
                    raise RuntimeError(f"custom prefix has trailing data: {args.ck_prefix_f32}")
            if len(imported) != expected_count:
                raise RuntimeError(
                    f"custom prefix element count mismatch: got={len(imported)} expected={expected_count}"
                )
            prefix_embeddings = imported
            prefix_source = "custom_f32"
        elif args.ck_prefix_source == "pytorch":
            prefix_embeddings = array("f", prefix.numpy().reshape(-1))
        prefix_metrics = _prefix_metrics(torch, prefix_embeddings, prefix)
        ck_decoder_t0 = time.perf_counter()
        if args.decoder_process == "spawn":
            decoder_report = _run_decoder_isolated(
                request={
                    "checkpoint": str(checkpoint),
                    "decoder_runtime": str(args.decoder_runtime.resolve()),
                    "prefix_tokens": int(encoder_report["prefix_tokens"]),
                    "prefix_embed_dim": int(encoder_report["embed_dim"]),
                    "prefix_grid": [
                        int(encoder_report["prefix_grid_x"]),
                        int(encoder_report["prefix_grid_y"]),
                    ],
                    "prefix_text_pos": len(before_ids) + int(encoder_report["prefix_text_pos"]),
                    "prefix_decode_policy": str(encoder_report["prefix_decode_policy"]),
                    "tokens_before": before_ids,
                    "tokens_after": after_ids,
                    "stop_token_ids": sorted(eos_ids),
                    "max_tokens": args.max_new_tokens,
                    "forced_generation_token_ids": (
                        torch_ids_raw if args.teacher_force_pytorch else None
                    ),
                    "trace_top_k": args.trace_top_k,
                },
                prefix_embeddings=prefix_embeddings,
                request_path=result_dir / f"{int(sample['index']):03d}.decoder-request.json",
                response_path=result_dir / f"{int(sample['index']):03d}.decoder-response.json",
            )
        else:
            decoder_report = bridge._run_decoder(
                decoder_runtime,
                prefix_embeddings,
                int(encoder_report["prefix_tokens"]),
                after_ids,
                tokens_before=before_ids,
                prefix_embed_dim=int(encoder_report["embed_dim"]),
                prefix_grid=(
                    int(encoder_report["prefix_grid_x"]),
                    int(encoder_report["prefix_grid_y"]),
                ),
                prefix_text_pos=len(before_ids) + int(encoder_report["prefix_text_pos"]),
                prefix_decode_policy=str(encoder_report["prefix_decode_policy"]),
                bridge_runtime="decode-staged",
                bridge_generation_mode="incremental-decode",
                tokenizer=processor.tokenizer,
                stop_token_ids=sorted(eos_ids),
                max_tokens=args.max_new_tokens,
                temperature=0.0,
                sample_top_k=40,
                top_p=1.0,
                min_p=0.0,
                stream_output=False,
                forced_generation_token_ids=(
                    torch_ids_raw if args.teacher_force_pytorch else None
                ),
                generation_trace_top_k=args.trace_top_k,
            )
        ck_decoder_sec = time.perf_counter() - ck_decoder_t0
        decoder_timings = decoder_report.get("timings", {})
        decoder_compute_sec = (
            float(decoder_timings.get("decoder_forward_mixed_ms", 0.0))
            + float(decoder_timings.get("decoder_generation_ms", 0.0))
        ) / 1000.0
        ck_ids_raw = [int(value) for value in decoder_report["generated_token_ids"]]
        ck_ids = _pre_eos(ck_ids_raw, eos_ids)
        divergence = _first_difference(torch_ids, ck_ids)
        result = {
            "index": int(sample["index"]),
            "id": sample["id"],
            "image_sha256": _sha256(image_path),
            "grid_thw": list(grid),
            "prefix_shape": [int(encoder_report["prefix_tokens"]), int(encoder_report["embed_dim"])],
            "torch_ids_raw": torch_ids_raw,
            "ck_ids_raw": ck_ids_raw,
            "torch_ids_pre_eos": torch_ids,
            "ck_ids_pre_eos": ck_ids,
            "first_divergence": divergence,
            "exact_pre_eos": divergence is None,
            "status": "pass" if divergence is None else "fail",
            "ck_prefix_source": prefix_source,
            "ck_prefix_f32_sha256": _sha256(args.ck_prefix_f32.resolve()) if args.ck_prefix_f32 else None,
            "ck_prefix_provenance_report": (
                str(custom_prefix_report) if custom_prefix_report is not None else None
            ),
            "teacher_forced": bool(args.teacher_force_pytorch),
            "decoder_process": args.decoder_process,
            "teacher_forced_input_ids": [
                int(value) for value in decoder_report.get("teacher_forced_input_ids", [])
            ],
            "torch_logit_trace": torch_logit_trace,
            "ck_logit_trace": decoder_report.get("generation_logit_trace", []),
            "prefix_metrics": prefix_metrics,
            "torch_text": processor.tokenizer.decode(torch_ids, skip_special_tokens=False),
            "ck_text": processor.tokenizer.decode(ck_ids, skip_special_tokens=False),
            "timings": {
                "torch_generation_sec": torch_sec,
                "torch_prefix_sec": torch_prefix_sec,
                "ck_encoder_sec": ck_encoder_sec,
                "ck_decoder_sec": ck_decoder_sec,
                "ck_decoder_compute_sec": decoder_compute_sec,
                "ck_decoder_init_overhead_sec": max(0.0, ck_decoder_sec - decoder_compute_sec),
                "ck_decoder_stages": decoder_timings,
            },
        }
        _write_json(result_path, result)
        completed.append(result)
        print(
            f"[{ordinal}/{len(samples)}] {sample['id']}: {result['status']} "
            f"first_divergence={divergence} torch={torch_sec:.1f}s "
            f"ck_encoder={ck_encoder_sec:.1f}s ck_decoder={ck_decoder_sec:.1f}s",
            flush=True,
        )

        summary = {
            "schema": "cke.qwen3vl_bf16_corpus_certification",
            "schema_version": 1,
            "checkpoint": str(checkpoint),
            "manifest": str(args.manifest.resolve()),
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "threads": args.threads,
            "completed": len(completed),
            "passed": sum(bool(row.get("exact_pre_eos")) for row in completed),
            "failed": sum(not bool(row.get("exact_pre_eos")) for row in completed),
            "min_prefix_cosine": min(
                (float(row["prefix_metrics"]["cosine"]) for row in completed if row.get("prefix_metrics")),
                default=None,
            ),
            "max_prefix_relative_rmse": max(
                (float(row["prefix_metrics"]["relative_rmse"]) for row in completed if row.get("prefix_metrics")),
                default=None,
            ),
            "results": completed,
        }
        _write_json(output_dir / "summary.json", summary)

    return 0 if all(bool(row.get("exact_pre_eos")) for row in completed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
