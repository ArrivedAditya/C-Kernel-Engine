# X-Ray Parity Drift in the IR Visualizer

Status: landed on `feat/v8-xray-visualizer` (commits `33958ace`, `a8d54a35`), PR #250.
Audience: operators and agents doing model bring-up and numerical-parity work on v8.

## Core idea

X-ray parity reports — per-checkpoint numerical drift of the CK runtime vs PyTorch
and llama.cpp oracles — were schema-validated JSON with no visual surface. The IR
visualizer is where the model's stitching is studied, so X-ray belongs there: a
human watching an agent bring up a model should see, in seconds, which backend
diverges, at which op, under which kernel contract, and whether it flips output
tokens.

The design principle: **join, don't instrument**. The call IR already carries every
op with its resolved kernel and contract; X-ray reports already carry per-stop
metrics; the kernel registry is already loaded. The tab connects them.

## What the X-Ray tab shows

1. **Backend status board** — one card per loaded report: backend chip (PyTorch /
   llama.cpp / ggml-oracle), verdict, first non-exact and first material divergence
   stops, thresholds, provenance, source path.
   Verdicts are three-state:
   - green `pass · within gate` — all checkpoints inside thresholds;
   - amber `cosmetic drift` — thresholds crossed but no top-1 logit flips;
   - red `behavioral divergence` — at least one top-1 flip.
2. **Run Context bar** — per report: backend, phase badge (PREFILL / DECODE /
   TEACHER_FORCED), seed, threads, backend version. A phase toggle appears on the
   drift chart when two or more phases are present.
3. **Per-checkpoint numerical drift chart** — x = stop index in execution order,
   y = log-scale max_abs and RMSE, threshold line, layer facet shading, green
   byte-exact prefix, amber pin at first non-exact, red pin at first material
   divergence, purple ▲ where per-edge error jumps > 3x, magenta ◆ where
   monotonicity breaks. Points are clickable (checkpoint detail: op, function,
   buffer, shape, sha256s, required vs resolved contract). Export SVG button
   produces a docs/blog-ready figure.
4. **Circuit X-Ray — op-level edges** — one row per call-IR op in execution order,
   grouped by collapsible layer bands, joined with X-ray checkpoints at the same
   stop: provider/function chip (click jumps to Kernel Flow), required-vs-resolved
   contract chip (amber = substituted), max_abs as a log-severity heat cell, RMSE,
   exact_ratio, phase. Ops without a checkpoint render gray so coverage gaps stay
   visible; the header states coverage explicitly (`N/M ops have X-ray edges`).
5. **Logits ranking vs oracle** — per-position top-1 agreement strips grouped by
   check kind (mixed_prefill / teacher_forced / persistent_vs_replay), with cosine.
6. **Execution trace and state stages** — ordered divergence stages with
   pass/fail styling; the first broken stage and what it poisoned downstream.
7. **X-Ray Operator Runbook** — perf-tab-style command panel, pre-filled with the
   report's run directory, Copy button per command.
8. **Live mode** — X-ray filenames are registered in both the Python `serve_live`
   `/api/snapshot` payload and the JS live polling, so the tab hot-reloads while
   agents emit new reports.

## How to invoke X-ray

There is deliberately no single `xray` command. Each X-ray script pins exactly one
oracle backend; you select the backend by selecting the script.

Quick paths (Make):

```bash
make test-bf16-xray                 # compile-check + unit-test all xray scripts
make xray-vision-parity BACKEND=llamacpp GGUF=<mmproj.gguf> \
  XRAY_OUTPUT_DIR=build/xray/qwen3vl_llamacpp
make xray-vision-parity BACKEND=pytorch CHECKPOINT=<hf_qwen3vl_ckpt> \
  RUNTIME_DIR=<runtime_dir> WEIGHTS_BUMP=<weights.bump> CALL_IR=<call.json> \
  XRAY_OUTPUT_DIR=build/xray/qwen3vl_bf16
make test-xray-validator-selftest   # schema validator self-test report
```

Per-backend scripts:

```bash
# Whisper encoder vs PyTorch (per-checkpoint drift, cke.whisper_encoder_pytorch_xray)
python3 version/v8/scripts/compare_whisper_encoder_pytorch_v8.py \
  --run-dir <run_dir> --checkpoint <hf_whisper_dir> \
  --stops key --output <run_dir>/whisper-encoder-xray.json

# Qwen3-VL vs PyTorch bf16 (orchestration)
python3 version/v8/scripts/xray_qwen3vl_bf16_v8.py \
  --checkpoint <ckpt> --runtime-dir <dir> --weights-bump <b> \
  --call-ir <c> --image <img> --output-dir build/xray/qwen3vl_bf16

# Qwen3-VL vs llama.cpp (orchestration)
python3 version/v8/scripts/xray_qwen3vl_llamacpp_v8.py --gguf <gguf> --image <img>

# Decoder vs PyTorch (token-level; consumes lowered_decode_call.json)
python3 version/v8/scripts/xray_decoder_pytorch_v8.py ...

# Execution state (ordered divergence stages)
python3 version/v8/scripts/xray_execution_state_v8.py \
  --subject-trace <ck_trace> --oracle-trace <oracle_trace> --output <out.json>
```

End-to-end flow: run the same model through N scripts → N schema-validated JSONs,
one per backend → place them in the run dir → regenerate the report
(`python3 version/v8/tools/open_ir_visualizer_v8.py --generate --run <run_dir>
--html-only --output <run_dir>/ir_report.html`) or serve with `--live`. The board
groups cards by backend automatically, the drift chart gets one curve per report,
and the circuit table joins every checkpoint to its op, kernel, and contract.

## Parity scorecard semantics (the honest "100%")

When a backend reaches full parity the board can say so, but the claim must stay
scoped. The honest banner is `79/79 edges within gate — FP32 · PyTorch · fixture
20260725 · 1 thread`, never a bare "100%". Parity is always scoped to model family,
dtype, backend version, fixture/seed, thread count, and sample. Live mode keeps the
scorecard a continuously re-proven state: the next agent run that regresses an edge
flips the card back to amber/red with the first-divergence pin.

## Known gaps and next steps

- Ranking → drift overlay is intentionally not rendered: token positions do not
  map cleanly onto op stops. The panel subtitle says so; a principled mapping
  (decode step → consumed checkpoint edges) is future work.
- Phase toggle is implemented but only appears with two or more phase artifacts;
  it has not been visually verified against real multi-phase data.
- All rendered evidence so far is from fixtures
  (`version/v8/tests/fixtures/xray/`); no real-run X-ray reports have been captured
  into a run dir yet. First real Whisper encoder run is the natural smoke test.
- Failure-narrative sequencing (guided "start here → then here" when a card goes
  red) is designed but not yet built; the board verdict split is its first piece.
- Bug-bundle export (drift SVG + failing checkpoint rows + provenance as a single
  issue-ready artifact) is designed, not built.

## Verification contract

- `python3 version/v8/scripts/test_visualizer_js_units_v8.py`
- `python3 version/v8/scripts/test_visualizer_health_v8.py`
- `python3 version/v8/scripts/test_visualizer_generated_e2e_v8.py`
  (includes `L3_xray_embed`: xray keys + schemas embedded, empty-state runbook
  markers)

Note: the pre-commit / pre-push fast regression is gated on runtime paths and does
not cover this tooling; on hosts where the nanbeige 3B smoke is OOM-killed
(`rc=-9`), `CK_SKIP_PRECOMMIT=1` / `CK_SKIP_PREPUSH=1` are the documented escape
hatches for visualizer-only diffs.
