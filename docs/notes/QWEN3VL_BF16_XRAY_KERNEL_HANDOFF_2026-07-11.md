# Qwen3-VL BF16 X-ray Kernel Handoff

This note records the Xeon diagnosis and its AVX2 integration through the v8 numerical execution contract resolver. The position edge is now selected declaratively; no model-name branch was added.

## Position Edge

Checkpoint: `vision.frontend.position.output`

The generic kernel `position_embeddings_add_tiled_2d_align_corners_bf16` now matches PyTorch's BF16 storage and arithmetic boundary:

- interpolation coordinates: each rational coordinate evaluated before FP32 storage, matching `torch.linspace`
- position table: BF16
- interpolation weights: BF16 RNE
- products: BF16 RNE
- accumulation: `((v00 + v01) + v10) + v11`, BF16 after each addition
- incoming hidden value: BF16 RNE
- residual result: BF16 RNE
- work partition: independent tokens
- deterministic: yes
- thread count changes arithmetic: no

Validation:

```text
synthetic shapes: 3/3 bit-exact
real grid: 56x72
embedding dimension: 1152
real values compared: 4,644,864
mismatches: 0
max_abs: 0
RMSE: 0
```

The registered contract is `bf16_tiled_2d_align_corners_rne_residual`. The Qwen3-VL vision circuit requests it only when the manifest declares `vision_position_storage_boundary=bf16`. The resolver selects `position_embeddings_add_tiled_2d_align_corners_bf16`, and GraphIR, LoweredIR, and call IR retain the exact contract ID, kernel ID, and function. GGUF manifests do not satisfy that selector and retain their existing FP32 position path.

Nightly coverage:

```text
Vision Position Storage BF16: 3/3 exact
v8 BF16 safetensors lowering: resolved contract and function identity pass
Qwen3-VL GGUF circuit tests: existing FP32 position path pass
```

## Next Edge: Layer-0 Norm1

Using the exact PyTorch position output as input:

```text
existing FP32 matched LayerNorm:
  cosine 0.9999986264
  RMSE 0.0005443210
  max_abs 0.01562357

existing BF16 LayerNorm:
  cosine 0.9999999999
  RMSE 0.0000039821
  max_abs 0.00390625
  mismatches 210 / 4,644,864
```

The existing BF16 kernel already matches storage but not PyTorch CPU's reduction order exactly. This is a `REDUCTION_CONTRACT_MISMATCH`, not justification for a model-specific LayerNorm path.

Public minimal reproduction:

```bash
LD_LIBRARY_PATH=build .venv/bin/python \
  research/numerical_contracts/reproduce_qwen3vl_bf16_layernorm_seed21.py
```

Fixture: `research/numerical_contracts/qwen3vl_bf16_layernorm_seed21.json`

Current seed-21 result:

```text
dimension: 1152
mismatches: 1
index: 332
CK: 0.00022411346435546875
PyTorch: 0.0002231597900390625
max_abs: 9.5367431640625e-7
```

The missing generic capability must specify BF16 input/gamma/beta/output, FP32 accumulation, exact mean and variance reduction order/formula, reciprocal-square-root precision, BF16 output rounding, independent-row threading and thread-count arithmetic stability.
