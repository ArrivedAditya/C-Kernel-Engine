# Quickstart Guide

This guide gets you from a fresh checkout to a deterministic generated-token output with the current **v8** inference lane in a few minutes. The v8 runner downloads a supported model, converts it, generates C, compiles a model runtime, and runs it — all from one command.

For depth, see the [v8 runbook](site/_pages/v8-runbook.html) and [version/v8/README.md](../version/v8/README.md).

## Prerequisites

- **Linux** with a working C toolchain (**GCC** with OpenMP) and **Make**
- **Python 3** with `venv`
- Enough RAM for the model you pick (the example below is a 270M-parameter model)

## 1. Build the Engine

```bash
make
```

This builds `build/libckernel_engine.so`, the core runtime library. The v8 runner also (re)builds the engine and tokenizer libraries itself when you pass `--force-compile`, so a stale build can never be paired with freshly generated model code.

## 2. Set Up the Python Environment

The v8 runner needs the packages in `version/v8/requirements.txt`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r version/v8/requirements.txt
```

If you skip this step, the `cks-v8-run` wrapper detects the missing environment and offers to bootstrap it interactively.

## 3. Convert, Compile, and Run a Model

One command handles the whole flow — download, convert to BUMP weights, build the v8 IR, generate C, compile `libmodel.so`, and generate tokens:

```bash
version/v8/scripts/cks-v8-run run \
  hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
  --context-len 1024 \
  --force-convert --force-compile \
  --chat-template auto \
  --prompt "Give me a concise example of C code." \
  --max-tokens 64 \
  --temperature 0.0
```

What happens, step by step:

1. **Download** — the `hf://repo/file.gguf` source is fetched into the local model cache (`~/.cache/ck-engine-v8/models`, override with `CK_CACHE_DIR`). A local `.gguf` path or an existing run directory works too.
2. **Convert** — weights are converted to the engine's BUMP format (`--force-convert` forces a fresh conversion).
3. **Compile** — the v8 IR is built, C code is generated, and `libmodel.so` is compiled and link-checked (`--force-compile` force-builds the engine and tokenizer libraries first).
4. **Generate** — `--temperature 0.0` gives deterministic greedy decoding, and `--prompt` + `--max-tokens` make the run non-interactive and bounded.

Notes:

- **Gemma 3**: keep `--chat-template auto` for the instruction/chat path. `--chat-template none` is raw continuation mode and requires `--allow-raw-prompt`.
- `--context-len` is context *capacity*, not consumed prompt length — allocate only what fits your RAM.
- Models are also accepted as **safetensors** checkpoints (an `hf://` repo without a `.gguf` filename, or a local checkpoint directory); those convert straight to BUMP via the safetensors path instead of GGUF.
- Add `--generate-visualizer` to emit an interactive IR report for the run.

## 4. Reuse the Runtime

Re-running the same command without `--force-convert --force-compile` reuses the cached conversion and compiled runtime. To see the runtimes you have built:

```bash
./build/ck-cli-v8 --list
```

## 5. Go Deeper

- [v8 runbook](site/_pages/v8-runbook.html) — the operator runbook: promoted text-family bring-up commands, the validated Qwen3-VL multimodal path, audio, and the certified FP32 training starter.
- [version/v8/README.md](../version/v8/README.md) — canonical per-family bring-up commands (Qwen, Gemma, GLM4, Nemotron, and more) and what each support level means.
- `version/v8/scripts/cks-v8-run run --help` — full runner option list.

## v7 SVG Training Docs

For tokenizer/data/model-size ablations and next-step criteria:
- `docs/v7-svg-training-ablation.md` (quick guide)
- `version/v7/reports/SVG_ABLATION_PLAN_2026-02-20.md` (canonical detailed matrix)
