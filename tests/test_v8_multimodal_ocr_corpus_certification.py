#!/usr/bin/env python3
"""Portable contracts for the model-agnostic multimodal OCR certifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "certify_multimodal_ocr_corpus_v8.py"
SPEC = importlib.util.spec_from_file_location("certify_multimodal_ocr_corpus_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PUBLIC_SPEC = importlib.util.spec_from_file_location(
    "export_vision_public_summary_v8",
    ROOT / "version" / "v8" / "scripts" / "export_vision_public_summary_v8.py",
)
assert PUBLIC_SPEC is not None and PUBLIC_SPEC.loader is not None
PUBLIC = importlib.util.module_from_spec(PUBLIC_SPEC)
PUBLIC_SPEC.loader.exec_module(PUBLIC)


class MultimodalOcrCorpusCertificationTest(unittest.TestCase):
    def test_gguf_source_bridge_command_uses_shared_generated_workspace(self) -> None:
        args = SimpleNamespace(
            decoder_gguf=Path("decoder.gguf"), encoder_gguf=Path("encoder.gguf"),
            decoder_runtime=None, encoder_runtime=None, composition_circuit=None,
            output_dir=Path("results"), chat_template="auto", thinking_mode="suppressed",
            context_len=4096, max_new_tokens=32, generation_progress_every=8,
            adapt_encoder_geometry=False,
        )
        command = MODULE._bridge_command(args, {"image": Path("image.jpg")}, Path("case"), "extract")
        self.assertEqual(command[command.index("--decoder-gguf") + 1], "decoder.gguf")
        self.assertEqual(command[command.index("--encoder-gguf") + 1], "encoder.gguf")
        self.assertEqual(Path(command[command.index("--workdir") + 1]),
                         Path("results/gguf_runtime").resolve())
        self.assertNotIn("--composition-circuit", command)

    def test_mixed_source_modes_reject_before_manifest_loading(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            MODULE.main([
                "--manifest", "unused.json", "--decoder-gguf", "decoder.gguf",
                "--encoder-runtime", "encoder", "--adapter-id", "fixture",
                "--model-label", "fixture", "--output-dir", "out",
            ])
        self.assertEqual(error.exception.code, 2)

    def test_gguf_execution_rejects_swapped_source_and_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.jpg"
            image.write_bytes(b"image")
            workspace = root / "runtime"
            report = {"status": "ok", "prefix_source": "encoder", "prefix_tokens": 2,
                      "encoder_report": {"image_source": "file", "image_path": str(image),
                                         "prefix_tokens": 2}}
            config = {"source_mode": "gguf", "gguf_runtime_dir": str(workspace)}
            for role in ("encoder", "decoder"):
                gguf = root / f"{role}.gguf"
                gguf.write_bytes(role.encode())
                config[f"{role}_gguf"] = MODULE._gguf_identity(gguf)
                runtime = workspace / role
                runtime.mkdir(parents=True)
                model = runtime / f"lib{role}.so"
                model.write_bytes(role.encode())
                (runtime / "libckernel_engine.so").write_bytes(b"engine")
                report[f"{role}_runtime"] = {
                    "source": "gguf", "gguf": str(gguf),
                    "workdir": str(runtime), "so_path": str(model),
                }
            sample = {"image": image, "image_sha256": MODULE._sha256_file(image)}
            evidence = MODULE._verify_bridge_execution(report, sample, config)
            self.assertEqual(evidence["source_mode"], "gguf")
            self.assertFalse(evidence["loaded_engine_verified"])
            changed = json.loads(json.dumps(report))
            changed["decoder_runtime"]["gguf"] = changed["encoder_runtime"]["gguf"]
            with self.assertRaisesRegex(ValueError, "GGUF differs"):
                MODULE._verify_bridge_execution(changed, sample, config)
            changed = json.loads(json.dumps(report))
            outside = root / "libdecoder.so"
            outside.write_bytes(b"wrong")
            changed["decoder_runtime"]["so_path"] = str(outside)
            with self.assertRaisesRegex(ValueError, "outside the workspace"):
                MODULE._verify_bridge_execution(changed, sample, config)
            (root / "decoder.gguf").write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "changed during OCR execution"):
                MODULE._verify_bridge_execution(report, sample, config)

    def test_gguf_resume_rejects_changed_generated_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "runtime"
            config = {"source_mode": "gguf", "gguf_runtime_dir": str(workspace)}
            evidence = {
                "evidence_kind": "bridge_reported_paths_and_artifact_hashes",
                "source_mode": "gguf", "loaded_engine_verified": False,
                "prefix_source": "encoder", "prefix_tokens": 2,
                "model_library_paths": {}, "model_library_sha256": {},
                "engine_library_paths": {}, "engine_library_sha256": {},
            }
            for role in ("encoder", "decoder"):
                source = root / f"{role}.gguf"
                source.write_bytes(role.encode())
                config[f"{role}_gguf"] = MODULE._gguf_identity(source)
                runtime = workspace / role
                runtime.mkdir(parents=True)
                model = runtime / f"lib{role}.so"
                engine = runtime / "libckernel_engine.so"
                model.write_bytes(role.encode())
                engine.write_bytes(b"engine")
                evidence["model_library_paths"][role] = str(model)
                evidence["model_library_sha256"][role] = MODULE._sha256_file(model)
                evidence["engine_library_paths"][role] = str(engine)
                evidence["engine_library_sha256"][role] = MODULE._sha256_file(engine)
            expected = {"image_index": 1, "image_sha256": "image", "truth_sha256": "truth"}
            truth = {"name": "Ada"}
            output = '{"name":"Ada"}'
            row = {
                "case_config": expected, "status": "complete", "image_index": 1,
                "image_sha256": "image", "truth_sha256": "truth",
                "execution_evidence": evidence, "generated_text": output,
                "output_sha256": MODULE._sha256_bytes(output.encode()),
                "parsed_output": truth, "metrics": MODULE._score(truth, truth),
                "generated_token_ids": [1], "generated_tokens": 1,
                "token_trace_sha256": MODULE._sha256_json([1]),
                "timings": {"wall_sec": 1.0},
            }
            path = root / "case.json"
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertIsNotNone(MODULE._load_resumed(path, expected, config, truth))
            (workspace / "decoder" / "libdecoder.so").write_bytes(b"changed")
            self.assertIsNone(MODULE._load_resumed(path, expected, config, truth))

    def test_gguf_mode_runs_fresh_report_and_invalidates_changed_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.jpg"
            truth = root / "truth.json"
            image.write_bytes(b"image")
            truth.write_text('{"name":"Ada"}', encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"samples": [{
                "inputs": [{"path": "image.jpg"}],
                "groundTruth": [{"path": "truth.json"}],
            }]}), encoding="utf-8")
            decoder = root / "decoder.gguf"
            encoder = root / "encoder.gguf"
            decoder.write_bytes(b"decoder")
            encoder.write_bytes(b"encoder")
            output = root / "output"
            runtime = output / "gguf_runtime"
            runtime.mkdir(parents=True)
            report_path = runtime / "bridge_report.json"
            report_path.write_text('{"status":"stale"}', encoding="utf-8")
            args = [
                "--manifest", str(manifest), "--decoder-gguf", str(decoder),
                "--encoder-gguf", str(encoder), "--adapter-id", "fixture",
                "--model-label", "fixture", "--output-dir", str(output),
                "--limit", "1", "--require-images", "1", "--threads", "7",
            ]
            calls = []

            def run(command, log_path, env):
                self.assertFalse(report_path.exists())
                self.assertEqual(env["CK_NUM_THREADS"], "7")
                self.assertEqual(env["OMP_NUM_THREADS"], "7")
                calls.append(command)
                report = {
                    "status": "ok", "prefix_source": "encoder", "prefix_tokens": 2,
                    "encoder_report": {"image_source": "file", "image_path": str(image),
                                       "prefix_tokens": 2},
                    "generated_text": '{"name":"Ada"}', "generated_token_ids": [1],
                    "generation_stop_reason": "eos", "timings": {},
                }
                for role, source in (("encoder", encoder), ("decoder", decoder)):
                    directory = runtime / role
                    directory.mkdir(exist_ok=True)
                    model = directory / f"lib{role}.so"
                    if not model.exists():
                        model.write_bytes(role.encode())
                    (directory / "libckernel_engine.so").write_bytes(b"engine")
                    report[f"{role}_runtime"] = {
                        "source": "gguf", "gguf": str(source),
                        "workdir": str(directory), "so_path": str(model),
                    }
                report_path.write_text(json.dumps(report), encoding="utf-8")
                return 0.01

            with mock.patch.object(MODULE, "_run", side_effect=run), redirect_stdout(StringIO()):
                self.assertEqual(MODULE.main(args), 0)
                self.assertEqual(MODULE.main(args), 0)
                self.assertEqual(len(calls), 1)
                (runtime / "decoder" / "libdecoder.so").write_bytes(b"changed")
                self.assertEqual(MODULE.main(args), 0)
                self.assertEqual(len(calls), 2)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["quality_status"], "pass")
            self.assertTrue((output / "image01" / "bridge_report.json").is_file())

    def test_required_count_applies_to_selected_distinct_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, payload in enumerate((b"first", b"second"), start=1):
                (root / f"image{index}.jpg").write_bytes(payload)
                (root / f"truth{index}.json").write_text('{"name":"Ada"}', encoding="utf-8")
            manifest = root / "manifest.json"
            def write_manifest(second_image: str) -> None:
                manifest.write_text(json.dumps({"samples": [
                    {"inputs": [{"path": "image1.jpg"}],
                     "groundTruth": [{"path": "truth1.json"}]},
                    {"inputs": [{"path": second_image}],
                     "groundTruth": [{"path": "truth2.json"}]},
                ]}), encoding="utf-8")
            args = [
                "--manifest", str(manifest), "--encoder-runtime", str(root / "encoder"),
                "--decoder-runtime", str(root / "decoder"),
                "--composition-circuit", "fixture", "--adapter-id", "fixture",
                "--model-label", "fixture", "--output-dir", str(root / "out"),
                "--require-images", "2", "--dry-run",
            ]
            write_manifest("image2.jpg")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                MODULE.main([*args, "--start-index", "2"])
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                MODULE.main([*args, "--limit", "1"])
            write_manifest("image1.jpg")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                MODULE.main(args)

    def test_public_ocr_summary_excludes_private_material_and_preserves_verdict(self) -> None:
        private = {
            "schema": "cke.multimodal_ocr_corpus_certification",
            "status": "complete", "quality_status": "fail",
            "aggregate": {"requested": 2, "completed": 2, "passing": 1, "errors": 0,
                          "field_accuracy": 0.5, "nonempty_field_accuracy": 0.5,
                          "total_wall_sec": 4.0},
            "rows": [
                {"status": "complete", "image_path": "/private/SENTINEL.jpg",
                 "generated_text": "SENTINEL-SECRET-OCR", "prompt": "SENTINEL-PRIVATE-PROMPT",
                 "image_sha256": "c" * 64,
                 "execution_evidence": {
                     "evidence_kind": "bridge_reported_paths_and_artifact_hashes",
                     "loaded_engine_verified": False, "prefix_source": "encoder",
                     "model_library_sha256": {"encoder": "a" * 64, "decoder": "b" * 64},
                     "model_library_paths": {"encoder": "/private/SENTINEL-encoder.so"},
                     "engine_library_sha256": {"decoder": "d" * 64},
                 }} for _ in range(2)
            ],
        }
        public = PUBLIC.build_public_summary(private, "ocr-v1")
        serialized = json.dumps(public)
        self.assertEqual(public["selected_count"], 2)
        self.assertEqual(public["completed_count"], 2)
        self.assertEqual(public["passing_count"], 1)
        self.assertEqual(public["status"], "fail")
        for secret in ("SENTINEL", "/private", "c" * 64, "d" * 64):
            self.assertNotIn(secret, serialized)
        minimal = PUBLIC.build_count_free_public_summary(private)
        self.assertEqual(minimal, {
            "schema_version": 1,
            "lane": "independent_image_to_ocr_task_quality",
            "status": "fail",
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "private.json"
            output = root / "public.json"
            source.write_text(json.dumps(private), encoding="utf-8")
            with redirect_stdout(StringIO()):
                self.assertEqual(PUBLIC.main([
                    "--private-summary", str(source), "--public-summary", str(output),
                    "--count-free",
                ]), 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), minimal)
        with self.assertRaisesRegex(ValueError, "inconsistent OCR verdict"):
            PUBLIC.build_public_summary({**private, "quality_status": "pass"}, "ocr-v1")
        with self.assertRaisesRegex(ValueError, "missing decoder generated-library hash"):
            broken = json.loads(json.dumps(private))
            broken["rows"][0]["execution_evidence"]["model_library_sha256"].pop("decoder")
            PUBLIC.build_public_summary(broken, "ocr-v1")
        partial = json.loads(json.dumps(private))
        partial["status"] = "incomplete"
        partial["quality_status"] = "incomplete"
        partial["aggregate"].update(completed=1, passing=1, errors=1)
        partial["rows"] = partial["rows"][:1]
        self.assertEqual(PUBLIC.build_public_summary(partial, "ocr-v1")["status"], "incomplete")

    def test_aggregate_distinguishes_execution_from_field_quality(self) -> None:
        metrics = {"json_valid": True, "expected_fields": 2, "exact_fields": 1,
                   "extra_fields": [], "nonempty_expected_fields": 2,
                   "nonempty_exact_fields": 1}
        aggregate = MODULE._aggregate([
            {"status": "complete", "metrics": metrics, "timings": {"wall_sec": 1.0}}
        ], 1)
        self.assertEqual(aggregate["completed"], 1)
        self.assertEqual(aggregate["passing"], 0)

    def test_extracts_plain_and_fenced_json(self) -> None:
        self.assertEqual(MODULE._extract_json_object('{"name":"Ada"}'), {"name": "Ada"})
        self.assertEqual(
            MODULE._extract_json_object('Result:\n```json\n{"name":"Ada"}\n```'),
            {"name": "Ada"},
        )
        self.assertIsNone(MODULE._extract_json_object("not JSON"))

    def test_scores_alternative_values_without_weakening_field_names(self) -> None:
        metrics = MODULE._score(
            {"sin": ["123-456", "123 456"], "name": "Ada", "blank": ""},
            {"sin": "123456", "name": "Grace", "blank": "", "extra": 1},
        )
        self.assertTrue(metrics["json_valid"])
        self.assertEqual(metrics["exact_fields"], 2)
        self.assertEqual(metrics["mismatched_fields"], ["name"])
        self.assertEqual(metrics["extra_fields"], ["extra"])
        self.assertEqual(metrics["nonempty_exact_fields"], 1)

    def test_loads_existing_private_manifest_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sample.jpg").write_bytes(b"image")
            (root / "sample.json").write_text('{"name":"Ada"}\n', encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "id": "sample",
                                "inputs": [{"path": "sample.jpg", "mimeType": "image/jpeg"}],
                                "groundTruth": [
                                    {"path": "sample.json", "format": "field-key-value-json"}
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = MODULE._load_samples(manifest)
        self.assertEqual(rows[0]["id"], "sample")
        self.assertEqual(rows[0]["truth"], {"name": "Ada"})

    def test_manifest_pins_media_and_truth_and_owns_case_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "sample.jpg"
            truth = root / "sample.json"
            image.write_bytes(b"image")
            truth.write_text('{"name":"Ada"}\n', encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "id": "sample",
                                "inputs": [
                                    {"path": image.name, "sha256": MODULE._sha256_file(image)}
                                ],
                                "groundTruth": [
                                    {"path": truth.name, "sha256": MODULE._sha256_file(truth)}
                                ],
                                "prompt": "Return JSON only.",
                                "comparison": {"max_new_tokens": 4096},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = MODULE._load_samples(manifest)

        self.assertEqual(rows[0]["prompt"], "Return JSON only.")
        self.assertEqual(MODULE._max_new_tokens(128, rows[0]), 4096)

    def test_manifest_rejects_drift_from_pinned_image_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sample.jpg").write_bytes(b"image")
            (root / "sample.json").write_text('{"name":"Ada"}\n', encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "inputs": [{"path": "sample.jpg", "sha256": "0" * 64}],
                                "groundTruth": [{"path": "sample.json"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "image SHA-256 mismatch"):
                MODULE._load_samples(manifest)

    def test_case_prompt_overrides_template_without_model_specific_logic(self) -> None:
        self.assertEqual(
            MODULE._build_prompt("Fields: {fields}", {"name": "Ada"}, "Shared OCR prompt"),
            "Shared OCR prompt",
        )

    def test_case_config_records_the_resolved_generation_budget(self) -> None:
        sample = {
            "index": 1,
            "image_sha256": "image-hash",
            "truth_sha256": "truth-hash",
        }
        config = MODULE._case_config("global-hash", sample, "prompt", 4096)
        self.assertEqual(config["max_new_tokens"], 4096)

    def test_public_row_redacts_private_content(self) -> None:
        private = {
            "image_index": 1,
            "image_sha256": "image-hash",
            "truth_sha256": "truth-hash",
            "status": "complete",
            "output_sha256": "output-hash",
            "token_trace_sha256": "trace-hash",
            "stop_reason": "stop_token",
            "generated_tokens": 2,
            "timings": {"wall_sec": 1.0},
            "metrics": {"json_valid": True},
            "execution_evidence": {"evidence_kind": "bridge_reported_paths_and_artifact_hashes",
                                   "loaded_engine_verified": False,
                                   "prefix_source": "encoder", "prefix_tokens": 4,
                                   "model_library_sha256": {"encoder": "a" * 64,
                                                            "decoder": "b" * 64}},
            "image_path": "/private/image.jpg",
            "truth_path": "/private/truth.json",
            "prompt": "private prompt",
            "generated_text": "private output",
        }
        serialized = json.dumps(MODULE._public_row(private))
        self.assertNotIn("/private", serialized)
        self.assertNotIn("private prompt", serialized)
        self.assertNotIn("private output", serialized)
        self.assertIn('"prefix_source": "encoder"', serialized)

    def test_resume_rejects_cases_without_generated_encoder_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "case_result.json"
            expected = {"image_index": 1, "image_sha256": "image", "truth_sha256": "truth"}
            config = {
                "encoder_runtime": {"model_library": {"sha256": "a" * 64}},
                "decoder_runtime": {"model_library": {"sha256": "b" * 64}},
            }
            truth = {"name": "Ada"}
            row = {"case_config": expected, "status": "complete", "image_index": 1,
                   "image_sha256": "image", "truth_sha256": "truth"}
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertIsNone(MODULE._load_resumed(path, expected, config, truth))
            row["execution_evidence"] = {
                "evidence_kind": "bridge_reported_paths_and_artifact_hashes",
                "loaded_engine_verified": False,
                "prefix_source": "encoder",
                "prefix_tokens": 4,
                "model_library_sha256": {"encoder": "a" * 64, "decoder": "b" * 64},
            }
            row["generated_text"] = '{"name":"Ada"}'
            row["parsed_output"] = {"name": "Ada"}
            row["metrics"] = MODULE._score(truth, row["parsed_output"])
            row["output_sha256"] = MODULE._sha256_bytes(row["generated_text"].encode("utf-8"))
            row["generated_token_ids"] = [1, 2]
            row["generated_tokens"] = 2
            row["token_trace_sha256"] = MODULE._sha256_json([1, 2])
            row["timings"] = {"wall_sec": 1.0}
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertIsNotNone(MODULE._load_resumed(path, expected, config, truth))
            malformed = dict(row, metrics={"json_valid": True})
            path.write_text(json.dumps(malformed), encoding="utf-8")
            self.assertIsNone(MODULE._load_resumed(path, expected, config, truth))
            for mutation in (
                {"metrics": {**row["metrics"], "exact_fields": 0}},
                {"parsed_output": {"name": "Grace"}},
                {"output_sha256": "0" * 64},
                {"generated_token_ids": [1, 3]},
                {"generated_tokens": True},
                {"timings": {"wall_sec": float("nan")}},
            ):
                with self.subTest(mutation=mutation):
                    path.write_text(json.dumps({**row, **mutation}), encoding="utf-8")
                    self.assertIsNone(MODULE._load_resumed(path, expected, config, truth))
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertIsNone(MODULE._load_resumed(path, expected, config, {"name": "Grace"}))
            for mutation in (
                {"prefix_tokens": 0},
                {"prefix_tokens": True},
                {"model_library_sha256": {"encoder": "wrong", "decoder": "b" * 64}},
                {"loaded_engine_verified": True},
            ):
                with self.subTest(mutation=mutation):
                    changed = dict(row, execution_evidence={**row["execution_evidence"], **mutation})
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    self.assertIsNone(MODULE._load_resumed(path, expected, config, truth))
            wrong_case = dict(row, image_index=2)
            path.write_text(json.dumps(wrong_case), encoding="utf-8")
            self.assertIsNone(MODULE._load_resumed(path, expected, config, truth))
            path.write_text("{malformed", encoding="utf-8")
            self.assertIsNone(MODULE._load_resumed(path, expected, config, truth))

    def test_make_target_declares_unsupported_llamacpp_oracle(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("test-cohere-compass-private-ocr-auto", makefile)
        self.assertIn("--adapter-id cohere_compass", makefile)
        self.assertIn("--oracle-id llama.cpp", makefile)
        self.assertIn("--oracle-status unsupported", makefile)
        self.assertIn("--adapt-encoder-geometry", makefile)

    def test_shared_private_manifest_drives_all_vision_family_lanes(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("V8_PRIVATE_VISION_CORPUS_MANIFEST", makefile)
        self.assertIn("test-cohere-compass-private-pytorch-ocr-auto", makefile)
        self.assertIn("test-cohere-compass-private-token-parity-auto", makefile)
        self.assertIn("test-v8-private-vision-corpus-auto:", makefile)
        self.assertIn("certify_cohere_compass_pytorch_ocr_v8.py", makefile)
        self.assertIn("compare_multimodal_corpus_runs_v8.py", makefile)
        self.assertIn("COHERE_COMPASS_PRIVATE_OCR_CONTEXT ?= 8192", makefile)

    def test_bridge_command_uses_shared_geometry_cache_when_requested(self) -> None:
        args = SimpleNamespace(
            decoder_runtime=Path("decoder"),
            encoder_runtime=Path("encoder"),
            composition_circuit="cohere_compass",
            chat_template="auto",
            thinking_mode="suppressed",
            context_len=4096,
            max_new_tokens=32,
            generation_progress_every=8,
            adapt_encoder_geometry=True,
            output_dir=Path("results"),
        )
        sample = {"image": Path("form.jpg")}
        command = MODULE._bridge_command(args, sample, Path("case"), "extract")
        rendered = " ".join(str(item) for item in command)
        self.assertIn("--encoder-geometry-cache-dir", rendered)
        self.assertIn("results/encoder_geometry_cache", rendered)

    def test_ocr_requires_the_pinned_image_fed_generated_runtimes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.jpg"
            image.write_bytes(b"image")
            encoder_dir = root / "encoder"
            decoder_dir = root / "decoder"
            encoder_dir.mkdir()
            decoder_dir.mkdir()
            (encoder_dir / "libvision_encoder.so").write_bytes(b"encoder")
            (encoder_dir / "libckernel_engine.so").write_bytes(b"engine")
            (decoder_dir / "libmodel.so").write_bytes(b"decoder")
            (decoder_dir / "libckernel_engine.so").write_bytes(b"engine")
            config = {
                "encoder_runtime": MODULE._runtime_identity(encoder_dir, "encoder"),
                "decoder_runtime": MODULE._runtime_identity(decoder_dir, "decoder"),
            }
            sample = {"image": image, "image_sha256": MODULE._sha256_file(image)}
            report = {
                "status": "ok", "prefix_source": "encoder", "prefix_tokens": 4,
                "encoder_report": {"image_source": "file", "image_path": str(image),
                                   "prefix_tokens": 4},
                "encoder_runtime": {"source": "prebuilt", "workdir": str(encoder_dir),
                                    "so_path": str(encoder_dir / "libvision_encoder.so")},
                "decoder_runtime": {"source": "prebuilt", "workdir": str(decoder_dir),
                                    "so_path": str(decoder_dir / "libmodel.so")},
            }
            evidence = MODULE._verify_bridge_execution(report, sample, config)
            self.assertEqual(evidence["prefix_tokens"], 4)
            self.assertEqual(evidence["evidence_kind"], "bridge_reported_paths_and_artifact_hashes")
            self.assertIs(evidence["loaded_engine_verified"], False)

            bad = dict(report, prefix_source="synthetic_zero")
            with self.assertRaisesRegex(ValueError, "image-fed encoder"):
                MODULE._verify_bridge_execution(bad, sample, config)
            bad = dict(report, encoder_report=dict(report["encoder_report"], image_path=str(root / "other.jpg")))
            with self.assertRaisesRegex(ValueError, "image path"):
                MODULE._verify_bridge_execution(bad, sample, config)
            bad = dict(report, decoder_runtime=dict(report["decoder_runtime"], source="gguf"))
            with self.assertRaisesRegex(ValueError, "prebuilt decoder"):
                MODULE._verify_bridge_execution(bad, sample, config)
            bad = dict(report, encoder_runtime=dict(report["encoder_runtime"], so_path=str(decoder_dir / "libmodel.so")))
            with self.assertRaisesRegex(ValueError, "encoder model library"):
                MODULE._verify_bridge_execution(bad, sample, config)
            image.write_bytes(b"different image")
            with self.assertRaisesRegex(ValueError, "image bytes changed"):
                MODULE._verify_bridge_execution(report, sample, config)
            image.write_bytes(b"image")
            (encoder_dir / "libvision_encoder.so").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "encoder model library changed"):
                MODULE._verify_bridge_execution(report, sample, config)

    def test_synthetic_prefix_cannot_complete_an_ocr_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "image.jpg").write_bytes(b"image")
            (root / "truth.json").write_text('{"name":"Ada"}', encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({"samples": [{
                "inputs": [{"path": "image.jpg"}],
                "groundTruth": [{"path": "truth.json"}],
            }]}), encoding="utf-8")
            for role, library in (("encoder", "libvision_encoder.so"),
                                  ("decoder", "libmodel.so")):
                runtime = root / role
                runtime.mkdir()
                (runtime / library).write_bytes(role.encode())
                (runtime / "libckernel_engine.so").write_bytes(b"engine")

            def fake_run(_command, log_path, _env):
                bridge = log_path.parent / "runtime" / "bridge_report.json"
                bridge.parent.mkdir()
                bridge.write_text(json.dumps({
                    "status": "ok", "prefix_source": "synthetic_zero",
                    "generated_text": '{"name":"Ada"}',
                }), encoding="utf-8")
                return 0.01

            args = [
                "--manifest", str(root / "manifest.json"),
                "--encoder-runtime", str(root / "encoder"),
                "--decoder-runtime", str(root / "decoder"),
                "--composition-circuit", "fixture", "--adapter-id", "fixture",
                "--model-label", "fixture", "--output-dir", str(root / "out"),
            ]
            with mock.patch.object(MODULE, "_run", side_effect=fake_run), redirect_stdout(StringIO()) as output:
                result = MODULE.main(args)
            summary = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result, 1)
            self.assertNotIn(str(root), output.getvalue())
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["aggregate"]["errors"], 1)
            self.assertEqual(summary["rows"], [])

            stale = root / "out" / "image01" / "runtime" / "bridge_report.json"
            stale.write_text(json.dumps({
                "status": "ok", "prefix_source": "encoder", "prefix_tokens": 4,
                "encoder_report": {"image_source": "file", "image_path": str(root / "image.jpg"),
                                   "prefix_tokens": 4},
                "encoder_runtime": {"source": "prebuilt", "workdir": str(root / "encoder"),
                                    "so_path": str(root / "encoder" / "libvision_encoder.so")},
                "decoder_runtime": {"source": "prebuilt", "workdir": str(root / "decoder"),
                                    "so_path": str(root / "decoder" / "libmodel.so")},
                "generated_text": '{"name":"Ada"}',
            }), encoding="utf-8")
            with mock.patch.object(MODULE, "_run", return_value=0.01):
                result = MODULE.main(args)
            self.assertEqual(result, 1)
            self.assertFalse(stale.exists())
            summary = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "incomplete")

    def test_completed_execution_does_not_promote_wrong_ocr_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "private-image.jpg"
            image.write_bytes(b"image")
            (root / "truth.json").write_text('{"name":"Ada"}', encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({"samples": [{
                "inputs": [{"path": image.name}],
                "groundTruth": [{"path": "truth.json"}],
            }]}), encoding="utf-8")
            for role, library in (("encoder", "libvision_encoder.so"),
                                  ("decoder", "libmodel.so")):
                runtime = root / role
                runtime.mkdir()
                (runtime / library).write_bytes(role.encode())
                (runtime / "libckernel_engine.so").write_bytes(b"engine")

            def fake_run(_command, log_path, _env):
                bridge = log_path.parent / "runtime" / "bridge_report.json"
                bridge.parent.mkdir()
                bridge.write_text(json.dumps({
                    "status": "ok", "prefix_source": "encoder", "prefix_tokens": 4,
                    "encoder_report": {"image_source": "file", "image_path": str(image),
                                       "prefix_tokens": 4},
                    "encoder_runtime": {"source": "prebuilt", "workdir": str(root / "encoder"),
                                        "so_path": str(root / "encoder" / "libvision_encoder.so")},
                    "decoder_runtime": {"source": "prebuilt", "workdir": str(root / "decoder"),
                                        "so_path": str(root / "decoder" / "libmodel.so")},
                    "generated_text": '{"name":"Grace"}',
                }), encoding="utf-8")
                return 0.01

            args = [
                "--manifest", str(root / "manifest.json"),
                "--encoder-runtime", str(root / "encoder"),
                "--decoder-runtime", str(root / "decoder"),
                "--composition-circuit", "fixture", "--adapter-id", "fixture",
                "--model-label", "fixture", "--output-dir", str(root / "out"),
                "--require-images", "1",
            ]
            with mock.patch.object(MODULE, "_run", side_effect=fake_run):
                self.assertEqual(MODULE.main(args), 0)
            summary = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["quality_status"], "fail")
            self.assertEqual(summary["aggregate"]["passing"], 0)
            self.assertEqual(PUBLIC.build_public_summary(summary, "ocr-v1")["status"], "fail")


if __name__ == "__main__":
    unittest.main()
