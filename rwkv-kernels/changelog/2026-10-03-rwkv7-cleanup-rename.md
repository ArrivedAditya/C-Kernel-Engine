# 2026-10-03 — RWKV-7 cleanup: rename to rwkv7, drop external references

## Change
- Renamed `include/ckernel_rwkv_decode.h` -> `include/ckernel_rwkv7_decode.h`
  and `src/kernels/rwkv_decode.c` -> `src/kernels/rwkv7_decode.c`.
- Unified the two remaining `ck_rwkv_*` symbols to `ck_rwkv7_*`:
  `ck_rwkv7_token_shift_lerp_ref`, `ck_rwkv7_channelmix_decode_ref`.
- Removed all mentions of `rwkv-kernels/psudocode.py` and
  `rwkv-kernels/modeling_rwkv7.py` from both files; the header now states the
  math contract self-contained (recurrence, `eps = norm_eps*head_dim`,
  per-head `rk` bonus, `(norm+bonus)*g` output, layouts, return codes).
- Reworded the `F.normalize` comment to plain `divisor max(norm,1e-12)`.
- No math changes. Mechanical updates only: `Makefile` `SRCS`, the `#include`
  in `ckernel_engine.h`, and the `ctypes` symbol names in
  `unittest/test_rwkv_decode.py`.

## Verification
- `make build/libckernel_engine.so` — builds clean, no rwkv warnings/errors.
- `.venv/bin/python unittest/test_rwkv_decode.py` — all pass, diffs unchanged
  from Step B (wkv_y `5.96e-08`–`2.38e-07`, lora `~3e-08`–`9e-08`,
  e2e_y `<= 2.98e-07`).
