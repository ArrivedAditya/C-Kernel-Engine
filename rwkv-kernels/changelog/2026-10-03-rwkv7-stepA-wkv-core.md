# 2026-10-03 — RWKV-7 Step A: WKV core aligned to modeling_rwkv7.py eager

## Change
- Replaced the pseudocode-derived additive WKV (`S + decay_row - ...`, scalar
  `bonus_mult`, `eps = H*1e-5`, pre-gate output) with the `rwkv7_recurrent` /
  `rwkv7_eager` math from `rwkv-kernels/modeling_rwkv7.py`:
  - `S_new[i,j] = S[i,j]*exp(w_log[j]) - (S[i,:].kk)*(kk[j]*a[j]) + v[i]*k[j]`,
    `y = S_new @ r`, state axes (value, key).
  - GroupNorm `eps = norm_eps * head_dim` (reference hardcodes `64e-5` at
    head_dim 64; the old `H*1e-5` was wrong whenever `H != N`).
  - Per-head bonus `s_h = sum(r_h*k_h*rk_h)`, `y_h += s_h*v_h` with `rk [H,N]`.
  - Output is `y_out = (groupnorm(y) + bonus) * g`; final CxC output projection
    stays with the caller's GEMV.
  - `v`-mixing and the four LoRA chains stay with the caller (Step B).
- New helpers so the `kk`/`k` transforms are covered in C too:
  - `ck_rwkv7_norm_kk_ref`: per-head L2 normalize, divisor `max(norm, 1e-12)`.
  - `ck_rwkv7_scale_k_ref`: `k_out = k_in * (1 + (a-1) * k_a)`.
- Old `ck_rwkv_wkv_decode_ref` removed (wrong contract); token-shift and
  channelmix kernels unchanged.

## Files
- `include/ckernel_rwkv_decode.h`, `src/kernels/rwkv_decode.c`,
  `unittest/test_rwkv_decode.py` (torch oracle mirrors `rwkv7_recurrent`
  one-token order, plus 8-step state-carry rollout and null-arg guards).

## Verification
- `make build/libckernel_engine.so` — builds clean.
- `.venv/bin/python unittest/test_rwkv_decode.py` — all pass:
  - `H=2,N=8`: kk `2.98e-08`, kscale `2.98e-08`, wkv_y `5.96e-08`,
    wkv_s `5.96e-08`, rollout y `8.94e-08` / s `7.15e-07`.
  - `H=4,N=8`: wkv_y `1.49e-07`, rollout y `2.38e-07` / s `2.38e-07`.
  - `H=4,N=16`: wkv_y `2.38e-07`, rollout y `2.98e-07` / s `7.15e-07`.
  - null guards return `-1`.
