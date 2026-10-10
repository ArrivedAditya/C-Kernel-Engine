# 2026-10-03 — RWKV-7 cleanup: drop `_ref` suffix from symbols

## Change
- Renamed all seven exported symbols in `include/ckernel_rwkv7_decode.h` /
  `src/kernels/rwkv7_decode.c` to drop the `_ref` suffix
  (`ck_rwkv7_wkv_decode`, `ck_rwkv7_lora_gates`, `ck_rwkv7_vgate_vmix`,
  `ck_rwkv7_norm_kk`, `ck_rwkv7_scale_k`, `ck_rwkv7_token_shift_lerp`,
  `ck_rwkv7_channelmix_decode`), plus the file-local `ck_rwkv7_sigmoid`
  helper. Updated the `ctypes` bindings in `unittest/test_rwkv_decode.py`.
- No math changes.

## Verification
- `make build/libckernel_engine.so` — builds clean, no rwkv warnings/errors.
- `.venv/bin/python unittest/test_rwkv_decode.py` — all pass, diffs unchanged
  (wkv_y `5.96e-08`–`2.38e-07`, lora `~3e-08`–`9e-08`,
  e2e_y `<= 2.98e-07`).
