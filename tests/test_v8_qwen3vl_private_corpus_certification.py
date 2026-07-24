from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "certify_qwen3vl_llamacpp_corpus_v8.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("qwen3vl_private_corpus", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Qwen3VLCorpusCertificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_module()

    def test_manifest_order_and_hashes_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "first.jpg").write_bytes(b"first")
            (root / "second.jpg").write_bytes(b"second")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {"id": "private-name-1", "inputs": [{"path": "first.jpg"}]},
                            {"id": "private-name-2", "inputs": [{"path": "second.jpg"}]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = self.module._load_corpus(manifest)
            self.assertEqual([row["index"] for row in rows], [1, 2])
            self.assertEqual(rows[0]["image_sha256"], self.module._sha256_file(root / "first.jpg"))

    def test_redacted_summary_excludes_paths_text_and_sample_ids(self) -> None:
        report = {
            "pass": True,
            "max_new_tokens": 128,
            "ctx_len": 4096,
            "steps": [{"generated_prefix": [1], "ck_next_text": "private"}],
            "stop_reason": None,
            "first_divergence": None,
            "prefix": {"grid": [36, 28], "tokens": 1008},
            "prompt_tokens_before_image": [1, 2],
            "prompt_tokens_after_image": [3],
            "ck_runtime": {
                "shared_library": {"sha256": "decoder"},
                "engine_library": {"sha256": "engine"},
            },
            "compiler_provenance": {
                "status": "pass",
                "decoder_family": "gcc",
                "engine_family": "gcc",
            },
            "llama_oracle": {"commit": self.module.PINNED_LLAMA_COMMIT},
            "generated_shared_text": "private generated document text",
        }
        row = self.module._redacted_row(
            index=1,
            image_sha256="image",
            prefix_sha256="prefix",
            report=report,
            elapsed={"bridge": 1.0, "parity": 2.0},
            requested_tokens=128,
        )
        encoded = json.dumps(row)
        self.assertNotIn("private", encoded)
        self.assertNotIn("path", encoded)
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["steps"], 1)
        self.assertEqual(row["matched_tokens"], 1)
        self.assertEqual(row["requested_tokens"], 128)
        self.assertEqual(row["prefill_tokens"], 1011)
        self.assertEqual(row["context_tokens_after_comparison"], 1012)
        self.assertEqual(row["elapsed_sec"]["total"], 3.0)
        self.assertEqual(row["elapsed_sec"]["comparison_per_token"], 2.0)

    def test_resume_requires_exact_case_configuration_and_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "case_result.json"
            config = {
                "global_config_sha256": "config",
                "image_index": 1,
                "image_sha256": "image",
            }
            result.write_text(
                json.dumps(
                    {
                        "case_config": config,
                        "redacted_row": {"image_index": 1, "status": "pass"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNotNone(self.module._resumed_row(result, config))
            changed = dict(config, image_sha256="changed")
            self.assertIsNone(self.module._resumed_row(result, changed))

    def test_summary_fails_for_any_divergence(self) -> None:
        selected = [{"index": 1}, {"index": 2}]
        config = self._config()
        summary = self.module._summary(
            selected=selected,
            rows=[
                {"image_index": 1, "status": "pass"},
                {"image_index": 2, "status": "fail"},
            ],
            config=config,
        )
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        encoded = json.dumps(summary)
        self.assertNotIn("/private", encoded)
        self.assertNotIn("llama_root", encoded)
        self.assertNotIn('"decoder"', encoded)
        self.assertNotIn('"mmproj"', encoded)

    def test_summary_treats_execution_errors_as_failures(self) -> None:
        summary = self.module._summary(
            selected=[{"index": 1}],
            rows=[{"image_index": 1, "status": "error", "error_sha256": "hash"}],
            config=self._config(),
        )
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["failed"], 1)

    def test_production_commands_use_batched_exact_runtime_parity(self) -> None:
        parser_values = type(
            "Args",
            (),
            {
                "context_len": 4096,
                "max_new_tokens": 128,
                "top_k": 16,
                "threads": 20,
                "ck_threads": 20,
                "llama_required_isa": "avx2",
            },
        )()
        command = self.module._parity_command(
            parser_values,
            bridge_report=Path("bridge.json"),
            prefix_path=Path("prefix.f32"),
            workdir=Path("work"),
            report_path=Path("report.json"),
        )
        rendered = " ".join(map(str, command))
        self.assertIn("--reuse-bridge-decoder-runtime-exact", rendered)
        self.assertIn("--llama-decode-mode batched", rendered)
        self.assertIn("--append-on-divergence stop", rendered)
        self.assertIn("--max-new-tokens 128", rendered)

    def test_progress_line_distinguishes_tokens_context_and_time(self) -> None:
        row = {
            "image_index": 7,
            "status": "pass",
            "matched_tokens": 128,
            "requested_tokens": 128,
            "prefix_tokens": 1008,
            "prompt_tokens": 27,
            "prefill_tokens": 1035,
            "context_tokens_after_comparison": 1163,
            "context_capacity": 4096,
            "elapsed_sec": {
                "bridge": 75.5,
                "parity": 100.0,
                "total": 175.5,
                "comparison_per_token": 0.78125,
            },
        }
        line = self.module._progress_line(
            row,
            completed=7,
            requested=40,
            resumed=True,
        )
        self.assertIn("matched=128/128", line)
        self.assertIn("prefix=1008", line)
        self.assertIn("prefill=1035", line)
        self.assertIn("context=1163/4096", line)
        self.assertIn("total=175.50s", line)
        self.assertIn("compare=0.781s/token-pair", line)
        self.assertTrue(line.endswith("resumed"))

    def test_timing_summary_reports_corpus_total_and_mean(self) -> None:
        timing = self.module._timing_summary(
            [
                {"elapsed_sec": {"total": 10.0}},
                {"elapsed_sec": {"total": 20.0}},
            ]
        )
        self.assertEqual(timing["total_sec"], 30.0)
        self.assertEqual(timing["mean_sec_per_image"], 15.0)
        self.assertEqual(timing["min_sec_per_image"], 10.0)
        self.assertEqual(timing["max_sec_per_image"], 20.0)

    def test_private_console_requires_tty_or_explicit_opt_in(self) -> None:
        with mock.patch.dict("os.environ", {"CI": "true"}):
            self.assertFalse(
                self.module._private_console_enabled(
                    SimpleNamespace(show_private_details=None)
                )
            )
        self.assertTrue(
            self.module._private_console_enabled(
                SimpleNamespace(show_private_details=True)
            )
        )
        self.assertFalse(
            self.module._private_console_enabled(
                SimpleNamespace(show_private_details=False)
            )
        )

    def test_private_console_renders_local_details_without_changing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case_dir = Path(temporary)
            (case_dir / "bridge_report.json").write_text(
                json.dumps(
                    {
                        "encoder_report": {
                            "source_image_size": [2200, 1700],
                            "image_width": 1152,
                            "image_height": 896,
                        },
                        "timings": {
                            "encoder_execute_ms": 20000.0,
                            "decoder_forward_mixed_ms": 85000.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (case_dir / "parity.json").write_text(
                json.dumps(
                    {
                        "generated_shared_text": '{"title":"Monthly Report"}',
                        "first_divergence": None,
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "image_index": 1,
                "image_sha256": "image-hash",
                "grid": [36, 28],
                "status": "pass",
                "matched_tokens": 128,
                "requested_tokens": 128,
                "prefix_tokens": 1008,
                "prefill_tokens": 1035,
                "context_tokens_after_comparison": 1163,
                "context_capacity": 4096,
                "stop_reason": None,
                "elapsed_sec": {
                    "bridge": 110.0,
                    "parity": 90.0,
                    "total": 200.0,
                },
            }
            output = io.StringIO()
            with mock.patch("sys.stdout", output):
                self.module._print_private_case_details(
                    sample={
                        "index": 1,
                        "image": Path("/private/form.jpg"),
                        "image_sha256": "image-hash",
                    },
                    row=row,
                    case_dir=case_dir,
                    prompt="Extract fields.",
                )
            rendered = output.getvalue()
            self.assertIn("EXACT MATCH", rendered)
            self.assertIn("/private/form.jpg", rendered)
            self.assertIn("source 2200x1700 -> processed 1152x896", rendered)
            self.assertIn("128/128 exact pre-EOS greedy token pairs", rendered)
            self.assertIn('Output (CK == llama.cpp', rendered)
            self.assertIn('{"title":"Monthly Report"}', rendered)
            self.assertNotIn("/private/form.jpg", json.dumps(row))

    def _config(self) -> dict[str, object]:
        return {
            "version": 1,
            "cke_commit": "cke",
            "manifest_sha256": "manifest",
            "decoder": {"path": "/private/decoder.gguf"},
            "mmproj": {"path": "/private/mmproj.gguf"},
            "llama_root": "/private/llama.cpp",
            "llama_commit": self.module.PINNED_LLAMA_COMMIT,
            "expected_llama_commit": self.module.PINNED_LLAMA_COMMIT,
            "compiler": "gcc",
            "prompt_sha256": "prompt",
            "context_len": 4096,
            "image_max_tokens": 1024,
            "max_new_tokens": 128,
            "threads": 20,
            "ck_threads": 20,
            "top_k": 16,
            "llama_required_isa": "avx2",
        }


if __name__ == "__main__":
    unittest.main()
