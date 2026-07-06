# Qwen3-VL AVX2 SDPR Prefix Parity - 2026-07-06

## Canonical Artifact

- Image: `ocr/1 81.jpg`
- Image SHA256: `eb3becd507063f184d417a6baccfdb22d86277308f21a679aac4e029ed8f25ff`
- Decoder GGUF: `Qwen3VL-8B-Instruct-Q4_K_M.gguf`
- Decoder SHA256: `67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2`
- MMProj GGUF: `mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`
- MMProj SHA256: `c6ba85508d82f42590e6eb77d5340369ab6fecf107a7561d809523d8aa5f3bfd`
- Prompt: `Extract visible form fields as compact JSON.`
- Context: 512
- Threads: 20
- Image token bounds: `--image-min-tokens 128 --image-max-tokens 128`
- Resolved visual grid: `12 x 9`
- Visual prefix rows: 108
- Prompt tokens before image: 5
- Prompt tokens after image: 14

## Finding

The first material AVX2 Qwen3-VL issue on the canonical JPEG is visual preprocessing parity, not a decoder GEMM speed issue.

With the original bridge path, CK used Pillow bilinear resize for JPEG input while llama.cpp mtmd uses its own Qwen-VL bilinear resize:

```text
x_ratio = (src_w - 1) / dst_w
y_ratio = (src_h - 1) / dst_h
```

The existing CK PPM fallback also used align-corners style `dst - 1` denominators.

## Prefix Parity

llama.cpp mtmd was first extended locally to accept the same image token bounds for the dump helper. Without the bounds, llama produced `3657` visual tokens (`69 x 53`), while CK produced `108` (`12 x 9`), so the prefix comparison was invalid.

After forcing the same `128..128` token policy, the original CK prefix was materially wrong:

| Case | Shape | Cosine | RMSE | MAE | Max Abs |
| --- | --- | ---: | ---: | ---: | ---: |
| Original CK JPEG path vs llama mtmd | `108 x 16384` | `0.559064` | `0.214336` | `0.135761` | `6.444275` |

After changing CK bridge image loading to use the llama-compatible resize for JPEG and PPM input:

| Case | Shape | Cosine | RMSE | MAE | Max Abs |
| --- | --- | ---: | ---: | ---: | ---: |
| Resize-compatible CK vs llama mtmd | `108 x 16384` | `0.992510` | `0.025334` | `0.007695` | `1.437146` |

This is a large improvement but not bit/parity perfect. Worst rows after the resize-compatible run were rows 1, 11, and 26.

## Multi-Token Result After Resize Fix

Using the resize-compatible prefix artifacts:

```text
build/qwen3vl_avx2_step23_parity_resizefix/bridge/bridge_report.json
build/qwen3vl_avx2_step23_parity_resizefix/prefix.f32
```

Persistent multi-token parity reached generated step 20 before the first top-1 mismatch:

| Step | CK token | llama token | Logit cosine | RMSE | Top-k overlap |
| ---: | --- | --- | ---: | ---: | ---: |
| 20 | `5214` (` risk`) | `31074` (` delays`) | `0.992328` | `0.791276` | `13/16` |

The mismatch is a narrow top-1 flip:

- CK top logit `risk`: `20.8574`
- llama logit for `risk`: `19.3493`
- llama top logit `delays`: `20.5490`
- CK logit for `delays`: `20.1250`

Full replay at the mismatch prefix still mismatched, so the remaining issue is not just incremental KV-cache state.

## Tensor Dump Status

The step-20 full replay dump command completed, but the first pass requested alias names (`attn_norm`, `q_proj`, etc.) that did not match llama.cpp's actual callback tensor base names. The result had `llama_dumped=0`, so it is not useful for layer attribution.

`--dump-list-only` with `--dump-dir` showed real llama base names such as:

- `embd (view)`
- `norm-0` ... `norm-35`
- `Qcur-0`, `Kcur-0`, `Vcur-0`
- `cache_k_l0 (view)`, `cache_v_l0 (view)`

Next attribution run should use these concrete llama names or add a name-alias map in the dump comparison harness.

## Recommendation

Land the resize/preprocessing fix separately from any AVX2 speed work. Then rerun:

1. Prefix parity on the canonical SDPR image.
2. Persistent multi-token parity.
3. Step-20 full replay tensor attribution with actual llama dump names.
4. OCR quality smoke on `ocr/1 81.jpg`, `ocr/2 81.jpg`, `ocr/3 81.jpg`.

Do not use VTune speed work as the next move until the resize-compatible preprocessing patch is accepted and parity baselines are refreshed.
