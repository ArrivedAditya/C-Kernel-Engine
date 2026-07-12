# C-Kernel-Engine

![C-Kernel-Engine cover](assets/cover_image.png)

[![Nightly Tests](https://github.com/C-Kernel-Engine/C-Kernel-Engine/actions/workflows/nightly.yml/badge.svg)](https://github.com/C-Kernel-Engine/C-Kernel-Engine/actions/workflows/nightly.yml)
[![llama.cpp Compatibility](https://github.com/C-Kernel-Engine/C-Kernel-Engine/actions/workflows/llamacpp-rolling-compat.yml/badge.svg)](https://github.com/C-Kernel-Engine/C-Kernel-Engine/actions/workflows/llamacpp-rolling-compat.yml)
[![Documentation](https://github.com/C-Kernel-Engine/C-Kernel-Engine/actions/workflows/docs.yml/badge.svg)](https://github.com/C-Kernel-Engine/C-Kernel-Engine/actions/workflows/docs.yml)

C-Kernel-Engine (CKE) is a C-first compiler, kernel library, and runtime for transformer inference and training on CPUs. It turns model weights, explicit circuits, and kernel capability maps into inspectable generated C rather than hiding execution behind a general-purpose framework.

The project is built around a practical thesis:

> A CPU AI system becomes competitive by making the complete model circuit explicit, proving its numerical behavior, and methodically moving every important stage toward the hardware roofline.

CKE is not only a collection of fast GEMMs. It is an attempt to connect model architecture, numerical semantics, code generation, memory planning, threading, parity testing, and profiler evidence into one reproducible system.

## Why CKE Exists

Modern CPUs provide wide SIMD, large caches, high core counts, mature profiling tools, and inexpensive memory capacity. They are also widely available and straightforward to connect into distributed systems. Much of that capability is lost when a runtime uses the wrong kernel shape, silently changes reduction order, repeatedly materializes intermediates, or schedules heterogeneous cores poorly.

CKE addresses that problem at several levels:

- **Explicit circuits:** Model topology, dimensions, weight policy, position semantics, and required numerical behavior are data, not model-name branches hidden in code generation.
- **Exact kernel resolution:** Kernel maps advertise concrete functions, layouts, dtypes, reductions, threading behavior, and ISA capabilities. Missing or ambiguous providers fail compilation.
- **Generated C:** GraphIR, lowered IR, and call-ready IR preserve the resolved decisions before emitting a standalone model runtime.
- **Numerical evidence:** Leaf kernels, stitched layers, mixed prefill, teacher-forced decode, and end-to-end outputs are compared with independent references such as llama.cpp and PyTorch.
- **Performance evidence:** Linux perf, Intel VTune, Intel Advisor roofline analysis, assembly inspection, and focused microbenchmarks identify where cycles and bandwidth are actually spent.
- **One CPU path from kernels to systems:** The long-term direction is efficient single-node execution followed by distributed CPU inference and training without surrendering observability.

## How It Works

```text
GGUF / safetensors metadata and weights
                 +
       circuit JSON requirements
                 +
      kernel capability maps
                 |
                 v
       fail-closed DSL resolver
                 |
                 v
      GraphIR -> LoweredIR -> Call IR
                 |
                 v
       generated C + weights.bump
                 |
                 v
              libmodel.so
        prefill / decode / backward
```

The ownership boundary is deliberate:

- **Circuits** describe the mathematical graph and required semantics.
- **Kernel maps** describe exact executable capabilities.
- **Kernels** implement and test those capabilities.
- **The DSL** validates, resolves, lowers, and emits. It must not guess model behavior.

This separation is enforced by tests that reject new model-family dispatch in protected compiler paths and reject unsupported or ambiguous numerical contracts.

## Current Scope

| Area | Current state |
|---|---|
| v8 text inference | Active GGUF/safetensors conversion, generated runtimes, quantized prefill, KV-cached decode, and model-family regression lanes |
| v8 vision inference | Qwen3-VL and related vision/compiler support under numerical-parity hardening with bounded first-divergence attribution |
| v7 training | FP32 forward/backward, optimizer, gradient accumulation, and training-kernel parity are the authoritative training lane |
| BF16 | Portable storage/rounding contracts are tested; native practical validation is resource-gated to AVX-512 BF16 or AMX-capable machines |
| Audio | Whisper tiny/base inference is next after the v8 compiler and Qwen3-VL parity gates |
| Distributed CPU | Architectural direction and research target; not yet a completed production runtime |

For the detailed support surface, see the [model and kernel matrix](https://c-kernel-engine.github.io/C-Kernel-Engine/model-kernel-matrix.html), [test report](https://c-kernel-engine.github.io/C-Kernel-Engine/test-report.html), and [roadmap](https://c-kernel-engine.github.io/C-Kernel-Engine/version-history.html).

## Correctness Before Speed

A fast kernel is not eligible if it changes the required mathematics. CKE uses layered validation:

1. Scalar or independent reference for the leaf operation.
2. ISA and threaded implementations against the same numerical contract.
3. Circuit and kernel-map resolution tests.
4. Stitched layer and semantic checkpoint comparison.
5. Mixed-prefill and teacher-forced token parity.
6. Practical end-to-end prompts, images, or audio fixtures.
7. Nightly and rolling llama.cpp compatibility gates.

When a model diverges, the X-ray workflow moves from sparse checkpoints to the first failing layer and then to the first failing operation. Fixes belong in a circuit, kernel map, or tested kernel, not as a special case in the DSL.

Read the [numerical contract architecture](https://c-kernel-engine.github.io/C-Kernel-Engine/v8-numerical-contracts.html) and [divergence harness guide](https://c-kernel-engine.github.io/C-Kernel-Engine/divergence-harness.html) for the method.

## Performance Method

CKE optimization separates kernel throughput from model throughput. The workflow is:

1. Establish numerical parity and a repeatable baseline.
2. Measure prefill and decode independently.
3. Profile the real model path, not only a synthetic loop.
4. Isolate the dominant kernel at a practical shape.
5. Inspect SIMD width, instruction mix, reduction structure, cache behavior, and thread utilization.
6. Change one implementation detail and remeasure both the kernel and end-to-end model.
7. Preserve the gain with numerical and performance evidence.

The goal is not to publish a theoretical peak as an achieved result. The goal is to explain the gap and close it stage by stage. See the [kernel tuning methodology](https://c-kernel-engine.github.io/C-Kernel-Engine/kernel-tuning-methodology.html), [profiling guide](https://c-kernel-engine.github.io/C-Kernel-Engine/profiling.html), and [prefill roadmap](https://c-kernel-engine.github.io/C-Kernel-Engine/prefill-performance-roadmap.html).

## Quick Start

Linux is the supported development and profiling environment.

```bash
git clone --recurse-submodules https://github.com/C-Kernel-Engine/C-Kernel-Engine.git
cd C-Kernel-Engine

./scripts/setup-dev-env.sh
./scripts/setup-hooks.sh
make
```

Run the compiler and contract gates:

```bash
make test-v8-dsl
make v7-kernel-parity-train
```

Run a v8 model through conversion, compilation, and chat. This command downloads the model on first use:

```bash
version/v8/scripts/cks-v8-run run \
  hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
  --context-len 1024 \
  --generate-visualizer
```

Start with the [quickstart](https://c-kernel-engine.github.io/C-Kernel-Engine/quickstart.html), [v8 runbook](https://c-kernel-engine.github.io/C-Kernel-Engine/v8-runbook.html), or [v7 training runbook](https://c-kernel-engine.github.io/C-Kernel-Engine/v7-runbook.html).

## Generated Runtime Capabilities

Depending on the circuit and model lane, generated `libmodel.so` runtimes can provide:

- Prompt prefill over full token batches.
- Autoregressive decode with per-layer KV cache.
- Quantized and floating-point kernel paths selected by explicit capability.
- FP32 teacher-forced forward/backward for supported v7 training circuits.
- Bounded semantic tensor exports for parity attribution.
- IR and memory visualizations for inspecting the generated program.

## Contributing

Useful contributions include:

- A numerically justified kernel capability and independent oracle test.
- A circuit definition that introduces no compiler model-name branch.
- llama.cpp or PyTorch parity adapters at a canonical tensor boundary.
- Reproducible VTune, Advisor, perf, assembly, or cache-analysis evidence.
- Model fixtures and regression cases that expose a real failure mode.
- Documentation that clearly separates measured support from planned work.

Run the repository hooks before opening a pull request:

```bash
./scripts/setup-hooks.sh
```

The pre-commit hook runs scoped staged regressions. The pre-push hook runs the heavier build, parity, inference, and training gates when relevant.

## Engineering and Consulting

CKE is also a working laboratory for CPU AI systems engineering. If your team needs help with:

- CPU inference or training architecture.
- Quantized GEMM/GEMV, attention, or fusion kernels.
- Numerical parity and first-divergence attribution.
- Model conversion, generated runtimes, or deterministic memory planning.
- Thread-pool, SIMD, cache, NUMA, or roofline optimization.
- Intel VTune/Advisor or Linux perf investigation.
- Building an evidence-backed CPU deployment or distributed-compute plan.

open a focused [GitHub issue](https://github.com/C-Kernel-Engine/C-Kernel-Engine/issues) for reproducible project work, or connect through [Antsand](https://antsand.com) for consulting and collaboration.

## Project Principles

- Evidence over benchmark theater.
- Correctness before optimization.
- Exact contracts instead of silent fallback.
- Generated code that engineers can inspect.
- Practical model shapes instead of toy-only claims.
- Performance changes must survive end-to-end testing.
- Planned work is labeled as planned.

The cover image was generated with Google Gemini.
