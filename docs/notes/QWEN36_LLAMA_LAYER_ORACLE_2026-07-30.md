# Qwen3.6 llama.cpp layer-oracle handoff

## Outcome

The llama.cpp layer observer is a persistent CKE X-ray capability. It does not
patch the llama.cpp source tree. CKE's `llama_token_replay_v8.cpp` installs the
public `ggml_backend_sched_eval_callback` through `llama_context_params::cb_eval`
and observes the stable `l_out-N` layer boundaries.

Use the normal trajectory comparison entry point:

```bash
python version/v8/scripts/compare_multitoken_logits_v8.py \
  --execution-mode trajectory \
  --model-dir /dev/shm/qwen36-q4km-runtime \
  --gguf downloads/Qwen_Qwen3.6-27B-Q4_K_M.gguf \
  --prompt-tokens-file /path/to/prompt-tokens.json \
  --max-new-tokens 8 \
  --threads 14 \
  --llama-profile-layers-out /dev/shm/qwen36-llama-layers.csv \
  --output /dev/shm/qwen36-trajectory.json
```

The output must be a new file. The report records its SHA-256, size, boundary
contract, and observer mechanism so stale evidence cannot silently pass.

## What the hook is for

- Persistent CKE-versus-llama.cpp numerical trajectory comparison.
- Sparse layer-boundary wall-time attribution.
- Comparing per-layer medians over multiple decode tokens.
- Finding the first layer or recurrent/full-attention class that amplifies a
  divergence.

It is deliberately not a full llama.cpp per-node profiler. Asking the callback
to observe every graph node introduces graph splits and materially changes
scheduling. Use selected X-ray captures for numerical attribution and native
profilers or isolated production-shape benchmarks for low-level performance.

## Qwen3.6-27B Q4_K_M decode finding

The host exposes 48 logical CPUs but the container CPU quota is 14 CPUs. A
24-thread comparison therefore measures throttling as well as kernel work.
Quota-matched wall rates for the prompt
`Give an example of C, Python, SQL code in detail.` were:

| Runtime | Threads | Decode |
|---|---:|---:|
| CKE | 14 | 2.65 tokens/s |
| llama.cpp | 14 | 5.5 tokens/s |

The CKE decode trace attributes the largest shares to:

| Kernel family | Mean per token | Share |
|---|---:|---:|
| DeltaNet recurrent core | 113.59 ms | 36.3% |
| Q4 projections / MLP gate-up | 106.05 ms | 33.9% |
| Q6 projections / MLP down | 50.96 ms | 16.3% |

Across eight decode tokens, the median layer-boundary sums were 256.71 ms for
CKE versus 114.78 ms for llama.cpp over the 48 recurrent layers, and 47.62 ms
versus 29.67 ms over the 16 full-attention layers. Sparse llama.cpp callback
timings can contain moving cgroup-throttle stalls, so use per-layer medians,
not a single callback-instrumented token total.

The dominant architectural difference is in production DeltaNet decode. CKE
walks 48 value heads serially and gathers/scatters strided state columns.
llama.cpp partitions heads through its threadpool and stores the corresponding
state rows contiguously for vector scale, dot, and multiply-add operations.
CKE's DeltaNet core alone costs approximately as much as llama.cpp's complete
recurrent layer. The next decode optimization should therefore parallelize the
exact Qwen3.6 grouped `48 heads / 16 groups / 128 state` contract and remove
the strided state-column traffic before further provider tuning.

## Maintenance contract

Future agents should extend the CKE-owned helper and its tests when they need a
new stable boundary. They should not carry a private llama.cpp patch or rebuild
the hook in a one-off script. llama.cpp remains the independent, pinned external
oracle; the CKE reference provider remains the faster internal comparison
baseline only after it has been certified against that oracle.
