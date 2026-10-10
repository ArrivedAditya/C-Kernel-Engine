# 2026-10-03 — RWKV T=1 FP32 decode reference (inference kernel)

## Scope
- Inference only: `B=1, T=1`, FP32 scalar reference. No prefill/chunked `T>1`, no batching, no quant, no SIMD/threadpool yet.

## Pseudocode diagnosis (`rwkv-kernels/psudocode.py`)
- GPT vs RNN: inference is strictly RNN — one token updates `shift (C)` + `wkv (H,N,N)` in place and carries them forward. Chunked/GPT mode is the same step looped over `T`.
- Blocking faults found:
  - `timemix:15`, `channelmix:81`: `cat([shift, x[:,-1]], dim=1)` yields `(B,2C)`; correct chunked shift is `x_shifted[:,0]=shift`, `x_shifted[:,1:]=x[:,:-1]`.
  - `timemix:47,56-61`: wkv update / `y` / `out[:,t]` dedented out of the `for t` loop — only the last `t` runs.
  - `timemix:64-70`: GroupNorm applied to last `y` instead of `out`.
  - `timemix:61`: `y.view[B,T,C]` invalid syntax.
  - `timemix:34-38,77`: returns `vprime` instead of `vprime_0`, so `v0` is overwritten by each layer instead of staying as layer-0 `vprime`.
  - `rwkv_model:98-108`: single shared shift/wkv state overwritten per layer (needs per-layer state); typo `channelmix_shiftcase`.
  - `timemix:42`: `normalize(..., dim=1)` normalizes over `T`; want `dim=-1` over head dim `N`.
  - `channelmix:84-85`: `W @ x` order reversed for `(B,T,C)` inputs.
  - `timemix:19-24`: `lerp(x, x_shift, mu)` inverts the standard `lerp(x_shift, x, mu)` convention.
  - `timemix:10`: `x.shape()` should be the `x.shape` property.

## Added
- `include/ckernel_rwkv_decode.h` — API + layout contract (`vectors [C]`, `state [H,N,N]`, `Wk [K,C]`, `Wv [C,K]`); documents corrections above.
- `src/kernels/rwkv_decode.c` — scalar FP32 reference, no malloc/free:
  - `ck_rwkv_token_shift_lerp_ref`: `xm = shift + mu*(x-shift)`, `shift_out = x`.
  - `ck_rwkv_wkv_decode_ref`: per-head `m = l2norm(k+removal)`, `p = k+iclr*iclr_mix`, `S_half = S + decay_row - (S@m)(e+m)^T`, `S_new = S_half + v@p^T`, `y = S_new@r`, then GroupNorm (`eps = H*1e-5`) + `bonus_mult*dot(r,k)*v`.
  - `ck_rwkv_channelmix_decode_ref`: `hidden = relu(Wk@xk)^2`, `out = Wv@hidden`.
- `unittest/test_rwkv_decode.py` — torch oracle mirroring C exactly, per-step parity + 8-step state/shift-carry rollout + null-arg guard.
- Wiring: `src/kernels/rwkv_decode.c` added to `Makefile` `SRCS`; `ckernel_engine.h` includes `ckernel_rwkv_decode.h`.

## Verification
- `make build/libckernel_engine.so` — builds clean.
- `.venv/bin/python unittest/test_rwkv_decode.py` — all pass:
  - `H=2,N=8,K=24`: shift `2.98e-08`, wkv_y `2.98e-07`, wkv_state `5.96e-08`, channelmix `7.15e-07`.
  - `H=4,N=8,K=48`: shift `2.98e-08`, wkv_y `3.13e-07`, wkv_state `1.19e-07`, channelmix `1.43e-06`.
  - `H=4,N=16,K=64`: shift `5.96e-08`, wkv_y `7.45e-07`, wkv_state `5.96e-08`, channelmix `9.54e-06`.
  - null-guard `rc=-1`; 8-step C-kernel rollout deterministic.

## Known simplifications / next steps
- `decay` is implemented as the additive `+decay[j]` broadcast exactly as written in the pseudocode; switch to multiplicative (`S*decay`) if real RWKV weights require it (one-line change + oracle update).
- `gate_lora` output used as-is (no SiLU); `mu`/`nu`/`mix`/`bonus` conventions follow the pseudocode as documented in the header.
- Next: per-layer state struct + full single-token layer-step composition, then prefill (`T>1`) equivalence (`chunk(T=8) == 8x step(T=1)`), then quant/SIMD.
