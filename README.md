# C-Kernel-Engine

![C-Kernel-Engine cover](assets/cover_image.png)

[![Nightly Tests](https://github.com/C-Kernel-Engine/C-Kernel-Engine/actions/workflows/nightly.yml/badge.svg)](https://github.com/C-Kernel-Engine/C-Kernel-Engine/actions/workflows/nightly.yml)
[![llama.cpp Compatibility](https://github.com/C-Kernel-Engine/C-Kernel-Engine/actions/workflows/llamacpp-rolling-compat.yml/badge.svg)](https://github.com/C-Kernel-Engine/C-Kernel-Engine/actions/workflows/llamacpp-rolling-compat.yml)
[![Documentation](https://github.com/C-Kernel-Engine/C-Kernel-Engine/actions/workflows/docs.yml/badge.svg)](https://github.com/C-Kernel-Engine/C-Kernel-Engine/actions/workflows/docs.yml)

C-Kernel-Engine (CKE) is a C-first compiler, kernel library, and runtime for transformer inference and training on CPUs. It turns model weights, explicit circuits, and kernel capability maps into inspectable generated C rather than hiding execution behind a general-purpose framework.

New to the project? Start with [What Is the C Kernel Engine?](https://www.shivasnotes.com/blog/5889/What-Is-the-C-Kernel-Engine), then use this README as the map from the ideas to the [documentation](https://c-kernel-engine.github.io/C-Kernel-Engine/) and [source code](https://github.com/C-Kernel-Engine/C-Kernel-Engine).

The project is built around a practical thesis:

> A CPU AI system becomes competitive by making the complete model circuit explicit, proving its numerical behavior, and methodically moving every important stage toward the hardware roofline.

CKE is not only a collection of fast GEMMs. It is an attempt to connect model architecture, numerical semantics, code generation, memory planning, threading, parity testing, and profiler evidence into one reproducible system.

The longer argument behind that direction is documented in [the CPU and smaller-model strategic bet](https://www.shivasnotes.com/blog/5878/Why-I-Stopped-Getting-High-on-the-Newer-AI-Models-And-Why-My-Strategic-Bet-Is-Still-Consistent-CPUs-Smaller-Models-and-Less-Compute-Will-Win) and the project origin story, [Unimporting PyTorch](https://www.shivasnotes.com/blog/5872/Unimporting-PyTorch-How-Constraint-and-Curiosity-Built-a-C-Kernel-Engine).

## Why CKE Exists

Modern CPUs provide wide SIMD, large caches, high core counts, mature profiling tools, and inexpensive memory capacity. They are also widely available and straightforward to connect into distributed systems. Much of that capability is lost when a runtime uses the wrong kernel shape, silently changes reduction order, repeatedly materializes intermediates, or schedules heterogeneous cores poorly.

CKE addresses that problem at several levels:

- **Explicit circuits:** Model topology, dimensions, weight policy, position semantics, and required numerical behavior are data, not model-name branches hidden in code generation.
- **Exact kernel resolution:** Kernel maps advertise concrete functions, layouts, dtypes, reductions, threading behavior, and ISA capabilities. Missing or ambiguous providers fail compilation.
- **Generated C:** GraphIR, lowered IR, and call-ready IR preserve the resolved decisions before emitting a standalone model runtime.
- **Numerical evidence:** Leaf kernels, stitched layers, mixed prefill, teacher-forced decode, and end-to-end outputs are compared with independent references such as llama.cpp and PyTorch.
- **Performance evidence:** Linux perf, Intel VTune, Intel Advisor roofline analysis, assembly inspection, and focused microbenchmarks identify where cycles and bandwidth are actually spent.
- **One CPU path from kernels to systems:** The long-term direction is efficient single-node execution followed by distributed CPU inference and training without surrendering observability.

For deeper context, read [v8 IR Pipeline Codegen](https://www.shivasnotes.com/blog/5917/v8-IR-Pipeline-Codegen-How-CKE-Hardens-Pure-C-Inference), [Templates Are Circuit Maps](https://www.shivasnotes.com/blog/5934/Templates-Are-Circuit-Maps-How-CKE-Describes-A-Model-Family), and the [architecture documentation](https://c-kernel-engine.github.io/C-Kernel-Engine/architecture.html).

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

Inspect the implementation directly:

- [v8 circuits](version/v8/circuits/) define supported model-family graphs and requirements.
- [v8 kernel maps](version/v8/kernel_maps/) bind exact operations to executable capabilities.
- [v8 compiler scripts](version/v8/scripts/) parse, validate, lower, and generate C.
- [C kernels](src/kernels/) implement the numerical and performance primitives.
- [v7 training](version/v7/) owns the current FP32 forward/backward path.
- [Tests](tests/) and [kernel unit tests](unittest/) preserve compiler, circuit, and numerical behavior.

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

## Read the Work

The articles explain the motivation and math; the documentation records the supported method; the source shows the current implementation.

| Topic | ShivasNotes | Technical reference | Source |
|---|---|---|---|
| Project overview | [What Is the C Kernel Engine?](https://www.shivasnotes.com/blog/5889/What-Is-the-C-Kernel-Engine) | [Concepts](https://c-kernel-engine.github.io/C-Kernel-Engine/concepts.html) | [Repository](https://github.com/C-Kernel-Engine/C-Kernel-Engine) |
| Compiler and generated C | [v8 IR Pipeline Codegen](https://www.shivasnotes.com/blog/5917/v8-IR-Pipeline-Codegen-How-CKE-Hardens-Pure-C-Inference) | [IR pipeline](https://c-kernel-engine.github.io/C-Kernel-Engine/ir-pipeline.html) | [v8 compiler](version/v8/scripts/) |
| Model circuits | [Templates Are Circuit Maps](https://www.shivasnotes.com/blog/5934/Templates-Are-Circuit-Maps-How-CKE-Describes-A-Model-Family) | [Numerical contracts](https://c-kernel-engine.github.io/C-Kernel-Engine/v8-numerical-contracts.html) | [Circuits](version/v8/circuits/) and [kernel maps](version/v8/kernel_maps/) |
| GGUF and model materialization | [GGUF: The File Format That Made Local LLMs Practical](https://www.shivasnotes.com/blog/5933/GGUF-The-File-Format-That-Made-Local-LLMs-Practical) | [GGUF to BUMP](https://c-kernel-engine.github.io/C-Kernel-Engine/gguf-bump.html) | [Conversion scripts](version/v8/scripts/) |
| Quantized kernels | [K-Quants Deep Dive](https://www.shivasnotes.com/blog/5925/K-Quants-Deep-Dive-Q4-K-Q5-K-Q6-K-Q8-K-And-Mixed-Dot-Products) | [Quant formats](https://c-kernel-engine.github.io/C-Kernel-Engine/quant-formats.html) | [Quantized kernels](src/kernels/) |
| Runtime ownership | [Threadpools and Memory Pools](https://www.shivasnotes.com/blog/5924/Threadpools-And-Memory-Pools-Why-CKE-Needs-Runtime-Ownership-For-CPU-AI-Kernels) | [Thread-pool design](https://c-kernel-engine.github.io/C-Kernel-Engine/threadpool.html) | [Thread pool](src/ck_threadpool.c) and [allocator](src/ckernel_alloc.c) |
| Performance engineering | [CPU Rooflines, Flamegraphs, VTune, and Perf Gates](https://www.shivasnotes.com/blog/5915/CPU-Performance-Engineering-for-AI-Rooflines-Flamegraphs-VTune-and-Perf-Gates) | [Kernel tuning methodology](https://c-kernel-engine.github.io/C-Kernel-Engine/kernel-tuning-methodology.html) | [Benchmarks](benchmarks/) and [profiling scripts](scripts/) |
| Distributed CPU AI | [MPI, RDMA, NUMA, and CKE](https://www.shivasnotes.com/blog/5922/Distributed-CPU-AI-MPI-RDMA-NUMA-and-C-Kernel-Engine) and [Pipeline vs Tensor Parallelism](https://www.shivasnotes.com/blog/5923/Pipeline-vs-Tensor-Parallelism-How-CKE-Splits-AI-Across-CPU-Nodes) | [Scaling architecture](https://c-kernel-engine.github.io/C-Kernel-Engine/scaling.html) | [Scaling roadmap](docs/site/_pages/scaling.html) |
| Gemma4 architecture | [Four Attention Paths, Shared KV, and Sliding Windows](https://www.shivasnotes.com/blog/5935/Gemma4-In-CKE-Four-Attention-Paths-Shared-KV-And-Sliding-Windows) | [Gemma4 speculative pair](https://c-kernel-engine.github.io/C-Kernel-Engine/gemma4-speculative-pair.html) | [Gemma4 circuit](version/v8/circuits/gemma4.json) |
| Audio roadmap | [How Audio Transformers Work](https://www.shivasnotes.com/blog/5928/How-Audio-Transformers-Work-The-Encoder-Path-Whisper-Timestamps-And-Why-Audio-Is-Not-A-VLM-Patch) | [Execution roadmap](https://c-kernel-engine.github.io/C-Kernel-Engine/version-history.html) | Audio circuit and kernels are planned after the v8 hardening gate |

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

The optimization strategy also follows [Amdahl's Law and the Theory of Constraints](https://www.shivasnotes.com/blog/5932/Amdahl-s-Law-Theory-Of-Constraints-And-C-Kernel-Engine-Optimization): improve the measured system constraint, then profile again instead of assuming the previous hotspot still dominates.

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
