# Version 8

`version/v8` is the active inference lane for text and multimodal bring-up.

Current scope:
- keep the inference runner, visualizer, run hub, and regression surface versioned as `v8`
- isolate circuit and multimodal bridge evolution inside `version/v8`
- preserve stable operator contracts while new vision ops land
- expose a native `v8` operator surface for text and multimodal bring-up

What is included here right now:
- `scripts/ck_run_v8.py`
- `scripts/cks-v8-run`
- `scripts/build_ir_v8.py`
- `scripts/memory_planner_v8.py`
- `scripts/resolve_model_dir_v8.py`
- `tools/open_ir_visualizer_v8.py`
- `tools/open_ir_hub_v8.py`
- `tools/ir_visualizer.html`
- `circuits/*`
- `contracts/*`
- `kernel_maps/*`

The active compiler input is:

```text
weights + circuits + kernel maps -> deterministic DSL lowering -> generated C
```

Circuits declare graph structure and required semantics. Kernel maps advertise
complete numerical contracts. `build_ir_v8.py` resolves those requirements and
then verifies that legacy lowering emitted the same uniquely selected provider.
This keeps migration behavior-preserving while moving parity discoveries out of
ad hoc runtime dispatch.

Canonical text bring-up examples:
- `version/v8/scripts/cks-v8-run run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf --context-len 1024 --force-compile --force-convert --chat-template=auto --generate-visualizer`
- `version/v8/scripts/cks-v8-run run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf --context-len 1024 --force-compile --force-convert --generate-visualizer`
- `version/v8/scripts/cks-v8-run run hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --context-len 1024 --force-compile --force-convert --generate-visualizer`
- `python3 version/v8/scripts/ck_run_v8.py run hf://unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf --force-convert --force-compile --context-len 1034`
- `version/v8/scripts/cks-v8-run run hf://mradermacher/Nanbeige4.1-3B-GGUF/Nanbeige4.1-3B.Q4_K_M.gguf --context-len 1024 --force-compile --force-convert --chat-template auto --generate-visualizer`

Notes:
- `Gemma 3`: use `--chat-template auto` for the instruction/chat path. `--chat-template none` is raw continuation mode now and requires `--allow-raw-prompt` if you intentionally want it.
- `NaanBeige`: if the first reply echoes `<|im_start|>assistant` or starts with `<think>`, keep `--chat-template auto`, do not force `none`, and treat it as a prompt-wrapper/chat-contract symptom rather than a stable expected reply shape.

Canonical vision bring-up example:
- `version/v8/scripts/cks-v8-run run hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf --mmproj hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf --image-path version/v8/test_assets/v8_vision_doc_card_72.ppm --prompt "Explain this image."`

Notes:
- For the validated Qwen3-VL 8B path, omitting `--mmproj` now auto-resolves the matching HF companion projector.
- The `.ppm` regression image is a Portable Pixmap file. CK uses it in CI because the format is simple enough to parse directly: a short header plus raw RGB pixels. That removes PNG/JPEG decoder dependency differences from the vision smoke. It is a test-fixture format, not a requirement for normal use; user-facing runs can still use PNG/JPEG when the local image stack is available.
- Qwen3-VL is an 8B multimodal lane. `make v8-regression-fast` stays text-family focused, while `make test-v8-qwen3vl-e2e-smoke` runs the cached vision E2E path when the decoder and mmproj artifacts are present.

That keeps `v8` small and honest: the version split now includes the inference runner, local kernel registry/maps, multimodal bridge entrypoint, and the `v8`-named operator tooling surface used by the visualizer, hub, and regression entrypoints.
