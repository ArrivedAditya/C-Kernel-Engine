# 2026-10-04 — RWKV-7: split state math out of wkv_decode

## Change
- Extracted the per-head state math from `ck_rwkv7_wkv_decode` into two
  separately testable units in `src/kernels/rwkv7_decode.c`
  (declared in `include/ckernel_rwkv7_decode.h`):
  - `ck_rwkv7_state_step`: single-head transition
    `S_new[i,j] = S[i,j]*exp(w_log[j]) - (S[i,:].kk)*(kk[j]*a[j]) + v[i]*k[j]`
    (the unit a SIMD/chunked variant would replace).
  - `ck_rwkv7_state_readout`: single-head `y = S_new.r`.
- `ck_rwkv7_wkv_decode` is now composition only (per-head step + readout,
  then unchanged GroupNorm + per-head bonus + gate). Same loop order, so
  numerics are bit-identical to before.
- `unittest/test_rwkv_decode.py`: new `run_state_case` with direct
  single-head step/readout parity vs the torch reference.

## Verification
- `make build/libckernel_engine.so` — builds clean.
- `.venv/bin/python unittest/test_rwkv_decode.py` — all pass:
  - `state N=8`: step `5.96e-08`, readout `1.19e-07`.
  - `state N=16`: step `5.96e-08`, readout `7.45e-08`.
  - composed WKV diffs unchanged (extraction preserved numerics exactly).
