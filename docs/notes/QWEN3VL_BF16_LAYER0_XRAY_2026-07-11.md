# Qwen3-VL BF16 layer-0 X-ray results

Validated against PyTorch BF16 on the canonical 1152x896 OCR geometry at origin/main a3829c1e.

## Proven kernel fixes

### Position interpolation

The BF16-storage-matched align-corners kernel passes 3/3 focused cases exactly. On the isolated real frontend position input it matches all 4,644,864 values.

### Vision M-RoPE

The production runtime passed a full rotary width of 72. The kernel treated the value as a pair count and indexed the second split half using pair + n_dims. Decoder-style section values also caused the vision kernel to return without rotating.

The corrected contract is:

    rotary_width = n_dims
    rope_pairs = rotary_width / 2
    split-half partner = pair + rope_pairs
    axis sections for head_dim 72 = [18, 18, 0, 0]

Measured layer-0 post-RoPE Q:

| State | Cosine | RMSE |
| --- | ---: | ---: |
| Before correction | 0.616694 | 1.473396 |
| Corrected kernel/sections | 0.999972 | 0.002710 |
| Plus BF16 post-RoPE storage | 0.9999998 | 0.000132 |

Focused kernel tests pass at head dimensions 8 and 72; production-width max absolute error is 2.38e-7.

## Measured numerical-contract requirements

These generated-artifact experiments are evidence, not committed generated-C fixes:

| Boundary | Before RMSE | Matched RMSE | Improvement |
| --- | ---: | ---: | ---: |
| layer-0 LayerNorm output | 5.44e-4 | 1.82e-5 | about 30x |
| layer-0 Q projection | 2.80e-3 | 1.04e-4 | about 27x |
| layer-0 post-RoPE Q | 2.71e-3 | 1.32e-4 | about 20x |

Required generic edge contracts:

1. Position interpolation output uses BF16 storage semantics.
2. LayerNorm consumes BF16-valued input and materializes BF16 output.
3. Packed QKV projection materializes BF16 before head splitting.
4. Vision RoPE consumes full rotary width, uses split-half pairing, and materializes BF16 Q/K.
5. Tolerances remain in the parity profile, not the circuit.

## Remaining infrastructure findings

- Latest strict call-IR generation still lacks generic bindings for use_rope_freq_factors and rope_freq_base.
- The vision_patch_bias exporter emits rows instead of rows times channels, causing a shape mismatch in bounded frontend diagnosis.
- Compiler/circuit fixes belong to the zero-hardcoding DSL work. No model-name branch is required.
