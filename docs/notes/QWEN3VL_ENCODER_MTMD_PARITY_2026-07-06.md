# Qwen3-VL Encoder vs llama.cpp mtmd Prefix Parity - 2026-07-06

## Scope

Compared CK v8 Qwen3-VL encoder bridge prefixes against llama.cpp `libmtmd`
image embeddings for several local images.

Reference helper:

```bash
build/mtmd_dump_embeddings \
  --model models/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  --mmproj models/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf \
  --image <image> \
  --out <llama_prefix.f32> \
  --threads 20
```

CK prefix source:

```bash
CK_NUM_THREADS=20 OMP_NUM_THREADS=20 \
.venv/bin/python version/v8/scripts/run_multimodal_bridge_v8.py \
  --decoder-gguf models/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  --encoder-gguf models/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf \
  --image-path <image> \
  --prompt 'OCR TEST' \
  --decoder-context-len 512 \
  --max-tokens 0 \
  --workdir <ck_workdir> \
  --dump-prefix-f32 <ck_prefix.f32> \
  --no-stream-output
```

## Results

| sample | shape | CK grid | cosine | RMSE | MAE | max abs |
|---|---:|---:|---:|---:|---:|---:|
| clean_text | 2097152 / 2097152 | 16x8 | 0.999259174 | 0.00627279 | 0.00353828 | 0.170078 |
| receipt | 2097152 / 2097152 | 16x8 | 0.999641299 | 0.00428494 | 0.00282775 | 0.0712301 |
| table | 2097152 / 2097152 | 16x8 | 0.999097347 | 0.00723644 | 0.00317368 | 0.66362 |
| paragraph | 3932160 / 3932160 | 24x10 | 0.996389270 | 0.0130378 | 0.00343413 | 1.01251 |
| card72 | 147456 / 147456 | 3x3 | 0.799037158 | 0.180574 | 0.109638 | 2.64726 |

## Interpretation

CK and llama.cpp agree on prefix shape/token count for all tested images, including
dynamic grids (`24x10`, `3x3`). The prefix values are not exact.

For OCR-sized inputs the difference is small but not negligible. This is enough
to explain why decoder step 0 logits are already looser than normal Qwen text
parity. For the tiny `card72` input, encoder parity is poor and should be treated
as a preprocessing/resize/normalization or mmproj lowering issue before judging
decoder accuracy.

## Current Takeaway

The Qwen3-VL issue is not only an autoregressive step-23 problem. Drift exists in
the visual prefix before decode starts. Decoder drift then accumulates until the
first visible greedy-token mismatch at generated step 23 on the ctx512 OCR run.
