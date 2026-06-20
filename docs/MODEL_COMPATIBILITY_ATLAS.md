# CK Model Compatibility Atlas

This is the reference-first workflow for adding new model families to C-Kernel-Engine.
GGUF and safetensors are source formats; CK runtime should converge on BUMP plus
explicit sidecar contracts.  New model work should start by proving the graph and
kernel contract before optimizing speed.

## Bring-up order

1. Inspect `config.json`.
2. Classify the layer pattern and required kernels.
3. Add safetensors-to-BUMP mapping with full source tensor coverage audit.
4. Run single-token and multi-token hidden/logit parity against PyTorch or a known
   reference.
5. Add GGUF conversion only after the BUMP graph contract is understood.
6. Optimize shared hot kernels after correctness is stable.

Use:

```bash
.venv/bin/python version/v8/scripts/inspect_model_contract_v8.py /path/to/config.json
```

The script is intentionally conservative: unsupported architectures should fail
closed with a list of missing kernels/templates.

## Current Families

| Family | Status | First target | Notes |
| --- | --- | --- | --- |
| Qwen2/Qwen3/Gemma3/Llama-style dense decoders | Supported at contract level | parity and perf | Uses standard attention + MLP path. |
| Qwen3.5 text | Bring-up active, usable | safetensors + GGUF parity | Hybrid DeltaNet/full-attention decoder. Safetensors require Qwen3.5 norm `+1` transform except `linear_attn.norm.weight`. |
| Gemma4 text | Bring-up active, usable | GGUF/safetensors parity | Hybrid full/sliding attention with per-layer embedding and per-layer RoPE control. |
| Nematron-H | Not yet lowerable | Mamba kernels | Hybrid Mamba/attention decoder. Existing DeltaNet kernels are not a substitute for Mamba selective scan. |
| Cohere Command-style models | Needs config access | tensor-name mapping audit | Main public checkpoints are gated in this environment. Start with config/weights access, then determine whether dense decoder mapping is enough. |

## Nematron-H Gap

Observed public config properties:

- `model_type: nemotron_h`
- `architectures: ["NemotronHForCausalLM"]`
- `hybrid_override_pattern` as one character per layer. Current Nano uses `M` = Mamba2, `*` = attention, `E` = MoE, and `-` = dense MLP.
- Mamba parameters: `mamba_num_heads`, `mamba_head_dim`, `ssm_state_size`, `time_step_rank`, `conv_kernel`
- MLP activation: `relu2`
- MoE parameters: routed experts, shared expert, top-k routing, group-limited expert selection
- Attention parameters: normal GQA-style attention heads/KV heads

Required new CK contracts before full Nematron-H inference:

- `mamba_in_proj_split`
- `mamba_conv1d_state_update`
- `mamba_dt_softplus`
- `mamba_selective_scan`
- `mamba_rmsnorm_gate`
- `mamba_out_proj`
- `relu2_mlp` forward and backward
- group-limited top-k router and routed ReLU2 expert dispatch/combine are covered by scalar reference kernels; shared expert MLP wiring is still missing
- Nematron-H safetensors-to-BUMP mapping with strict all-weight coverage
- Nematron-H template/layer policy for `mamba` vs attention layers

The first implementation should be scalar FP32/BF16 parity-first, then optimized
with AVX/AVX-512/AMX only after hidden-stream parity is stable.

## Cohere Gap

Cohere Command repositories were gated during this run, including `config.json`.
Do not guess the tensor names.  Once access is available:

1. Run the contract inspector on `config.json`.
2. Dump safetensors headers only.
3. Compare tensor names against the existing Llama-family importer.
4. If the graph is dense attention + GLU MLP, add a Cohere-specific name mapper
   and audit test before adding new math.
5. If the graph has custom attention, sliding windows, logit scaling, or unusual
   norms, model those explicitly in sidecar config.

## Why This Matters

CK should be able to say: given a kernel composition, embedding size, layer
pattern, and training contract, the DSL compiler can build a model independent
of PyTorch, llama.cpp, or Unsloth.  Compatibility with diverse model families is
evidence that the architecture is kernel/template-driven rather than hardcoded
to one lineage.
