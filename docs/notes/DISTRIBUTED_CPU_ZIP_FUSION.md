# Distributed CPU: Zip Fusion Beyond Tensor Parallelism

> **Canonical documentation:** [Distributed CPU: Zip Fusion Beyond Tensor
> Parallelism](../site/distributed-cpu.html) ([published
> HTML](https://c-kernel-engine.github.io/C-Kernel-Engine/distributed-cpu.html)).
> This Markdown file is a subordinate engineering and experiment-planning
> note. It must not replace or contradict the canonical HTML page.

Status: research design. CKE does not yet provide distributed execution as a
production feature. Nothing here is a measured result, a supported
configuration, or a committed performance claim. Every quantitative figure in
this note is idealized arithmetic unless explicitly marked otherwise.

## Thesis

CKE's distributed-execution research direction is **whole-graph algebraic
scheduling on CPU-first economics**. Instead of exposing named parallel
strategies (TP, CP, EP, PP) and letting the user compose them, the scheduler
asks, per graph region:

```text
For this exact graph region,
what representation lets N CPU nodes execute
the longest independent local path
before any global synchronization?
```

The unit of distribution is the whole quantized operator subgraph, including
work that most frameworks treat as implementation detail: packed-weight
unpack, scale decode, quantized dot products, local accumulation, SwiGLU,
RMSNorm partial statistics, attention-head work, output projection, expert
routing, and Memory-Tetris residency.

The goal is to make a cluster behave as **one pool of compute that re-shapes
per graph region**, not as N machines each holding 1/N of the model.

## Prior Art, Candidly

Head-shard and context-shard attention are not new, and CKE did not invent
them:

- **Megatron Core** combines tensor, pipeline, context and expert parallelism
  and treats context parallelism as a first-class way to partition long
  sequences.
- **DeepSpeed** provides pipeline, tensor and expert parallelism; its MoE API
  can distribute experts differently per MoE layer.
- **vLLM** supports combinations such as data-parallel attention with expert-
  or tensor-parallel MoE layers.
- **DeepSpeed-Ulysses** partitions along sequence length and uses all-to-all
  communication for attention.
- **Unified Sequence Parallelism** combines Ulysses with Ring Attention.
- **LoongTrain** explicitly combines head parallelism and context parallelism
  as a 2-D attention decomposition.
- **ATTENTION2D** similarly partitions attention along query and key/value
  dimensions.

The potentially distinctive claim is narrower than any of these: on CPU-first
economics (abundant DRAM, measured heterogeneous nodes, explicit storage
tiers), use whole-graph algebra to minimize global synchronization boundaries
per layer — and prove it experimentally. The value CKE could add is the
combination, not any single decomposition.

## The Zip Schedule

Per layer, each node runs a fused 1-D lane. For a two-node, Qwen-like
attention block:

```text
                  replicated hidden state x
                         │
             ┌───────────┴───────────┐
          Node A                  Node B
       unpack W_A               unpack W_B
       QKV shard                QKV shard
       heads 0..H/2             heads H/2..H
       RoPE / QK norm           RoPE / QK norm
       attention                attention
       local Wo shard           local Wo shard
       partial y_A              partial y_B
             └──────────┬────────────┘
                 small reduction
                        │
                 next hidden state
```

Key points:

- Heads are **not** concatenated into one giant intermediate tensor. Each node
  applies its local output-projection shard to its own attention output and
  produces a hidden-state-sized partial. The only communication boundary is
  the final reduction `y = partial_A + partial_B`.
- For attention at long context, the decomposition **unzips** from 1-D to
  2-D: head groups on one axis, context shards on the other. Example: 6 nodes
  = 3 context shards × 2 head groups; each node computes large local
  attention tiles over 1/3 of the context and 1/2 of the heads.
- Cross-context merging uses **online-softmax sufficient statistics** —
  per-partition local max, local normalization sum, and local weighted
  output — never the raw attention matrix. The network sees three small
  vectors per query block instead of a 1M × 1M score matrix.
- After attention the schedule **zips back** to 1-D for the MLP lane:
  gate/up shard → local SwiGLU → local down-projection partial → one merge.

Payload sizes at the boundaries (idealized arithmetic):

```text
hidden state, 8192-dim BF16, 1 token:   8192 × 2 B  ≈  16 KB
hidden state, 32-token batch:                          ≈ 512 KB
online-softmax stats per query block:    local max + local sum
                                         + local weighted output
```

GQA note: modern Qwen-like models may have, conceptually, 32 query heads
against 8 KV heads. Query-head groups must be partitioned so the needed KV
heads are local or cheaply replicated. Because CPU RAM is plentiful,
replicating K/V-head metadata or even some KV data may be preferable to
forcing communication.

## The Metric: Global Synchronization Boundaries per Layer

The useful metric is **global synchronization boundaries per layer**, not
tensor-parallel degree. A conventional TP schedule synchronizes after nearly
every operator:

```text
QKV GEMM → collective → attention → collective → output GEMM → collective
→ MLP up → collective → activation → MLP down → collective
```

A hyper-fused schedule aims for one merge per fused region:

```text
NODE A                         NODE B
------------------------------------------------
unpack W_A                     unpack W_B
QKV shard                      QKV shard
local heads                    local heads
attention                      attention
output projection partial      output projection partial
local reduction                local reduction
          \                   /
           tiny global merge            ← boundary 1

gate/up shard                  gate/up shard
SwiGLU local                   SwiGLU local
down projection partial        down projection partial
          \                   /
             one merge                  ← boundary 2
```

"We support TP=2" says nothing about how often nodes stop to agree. Boundary
count per layer is the statement that matters, and the interesting output of
the experimental program is a **topology-versus-context map**, not a single
tok/s figure:

```text
small context:        1 CPU node
medium:               TP=2
long:                 head parallel + TP
very long (128K–1M):  head parallel × context parallel
                      (+ perhaps TP inside projections)
```

## Communication Must Be Measured

The idealized serialization of 512 KB at a nominal 400 Gb/s (≈ 50 GB/s) link
is about 10 µs. That figure is a **division, not a measurement**. Real
end-to-end synchronization adds RDMA/collective latency, software overhead,
PCIe traversal, memory access and barrier costs.

The thesis stands on the **ratio**, not the absolute number: if each node
performs milliseconds of useful attention/GEMM work between boundaries,
hundreds of KB of communication is not scary — and at 1M context the
compute/communication ratio can become particularly favorable. But this is a
research hypothesis. It must be demonstrated on real NICs and a real switch,
not assumed from link-sheet bandwidth.

## CPU-First Economics

GPU frameworks shard because HBM is scarce; state often *cannot* be
replicated. CPU nodes can carry 150 GB, 256 GB, 512 GB or eventually TBs of
DRAM, so CKE can deliberately **spend memory to eliminate communication**:

- replicate the hidden input;
- replicate norm parameters;
- replicate routing metadata;
- keep multiple packed layouts;
- cache hot experts on several nodes.

This inverts the usual optimization objective. Memory-Tetris residency adds
another dimension the mainstream parallelism literature usually does not
center: tensor residency itself is part of scheduling. Per tensor, the
planner could choose among compute-from-resident, fetch from local NVMe,
fetch from remote DRAM over RDMA, route the activation to a remote resident
expert, recompute, or replicate for future reuse. See
[MEMORY_TETRIS_NVME_DESIGN.md](MEMORY_TETRIS_NVME_DESIGN.md).

## Heterogeneous Topology

Megatron-style systems generally assume fairly regular accelerator groups.
CKE's lab fleet is explicitly heterogeneous: a P3-class Intel i7-14700T node
(active), a Ryzen 9950X3D node (planned), and Xeon AMX hardware (future).
Shards should be assigned **proportionally to measured throughput** — e.g.
Node A takes 30% of a tensor shard and Node B 70% because X-Ray measured them
that way — rather than assuming equal ranks. The target is distributed graph
compilation over heterogeneous CPU resources, with 200/400G RDMA-class links
as the fabric assumption to be validated.

## Open Questions

- What is the real end-to-end cost of one hidden-state reduction on the lab
  fabric (RDMA setup, collective latency, barriers), and how does it scale
  with node count?
- At what context length does unzipping to 2-D attention actually beat a
  1-D head-parallel schedule?
- How expensive is GQA KV replication in practice, and when does it beat
  communicating KV?
- How do stragglers on heterogeneous nodes interact with barrier-heavy
  schedules, and does proportional sharding actually equalize lane time?
- Can switch congestion at 6+ nodes keep the compute/communication ratio
  favorable, or does the fabric become the boundary condition?
- Which Memory-Tetris residency choices (local DRAM, remote DRAM, NVMe,
  recompute, replicate) survive contact with measured latencies?

## Experimental Plan and Promotion Gates

Ordered experiments, each with a promotion gate:

1. **Single-node baseline.** Fully resident model, measured kernel times per
   graph region. Gate: reproducible baseline numbers recorded to CKE's
   evidence standard.
2. **Two-node TP reference.** A conventional TP=2 implementation, measuring
   the fraction of wall time spent in collectives. Gate: collective-time
   fraction measured and published, whatever the number is.
3. **Hyper-fused schedule.** Same two nodes, fused 1-D lanes, measuring
   synchronization boundaries per layer and total sync time. Gate: fewer
   boundaries and lower collective-time fraction than experiment 2, with
   identical numerics.
4. **Head × context 2-D at long context.** 6-node 3×2 grid with
   online-softmax merging. Gate: favorable ratio versus 1-D head-parallel at
   the same context, measured not estimated.
5. **Numerical parity.** Every distributed schedule must match the
   single-node baseline under the existing zero-tolerance numerical contract
   rules. Token parity alone is insufficient; selected layer/checkpoint
   outputs must also match. Gate: parity holds across thread counts, node
   counts and schedules.

**Hypothesis to test (not a result):** a fused schedule might cut collective
time from roughly 25% of wall time (a plausible conventional-TP figure on
this class of hardware) to roughly 5–10%. These numbers are the hypothesis
the experiments are designed to confirm or refute.

**Failure modes to measure:** barrier stalls, stragglers on heterogeneous
nodes, GQA KV replication cost, and switch congestion.

## Canonical Documentation Contract

The canonical [Distributed CPU HTML page](../site/distributed-cpu.html)
presents this work as a research design, not current production support.
Future documentation changes must update the HTML source and generated page
first. This note may retain lower-level experiment plans, but it must link
back to the HTML representation and remain subordinate to it.

Avoid claiming that CKE invented tensor/head/context parallelism (it did
not), that idealized bandwidth divisions are measurements, or that any
distributed schedule is implemented. The defensible story is that whole-graph
algebraic scheduling on CPU-first economics may reduce global synchronization
boundaries per layer, and the hardware gates will determine whether the
compute/communication ratio actually holds.
