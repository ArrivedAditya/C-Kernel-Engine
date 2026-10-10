# 2026-10-03 — RWKV-7 Step B: LoRA gates + v-mix (unfused, exact)

## Change
- Added two unfused FP32 reference units against
  `rwkv-kernels/modeling_rwkv7.py` (`Rwkv7Attention.lora_gates`, `:546-571`,
  and the `v_first` branch, `:485-488`):
  - `ck_rwkv7_lora_gates_ref`: `w_log = -e^-0.5*sigmoid(tanh(xw@w1)@w2+w0)`,
    `a = sigmoid((xa@a1)@a2+a0)`, `g = sigmoid(xg@g1)@g2` (no trailing
    bias on `g`, per reference). Layouts `w1/a1/g1 [C,R]`, `w2/a2/g2 [R,C]`.
  - `ck_rwkv7_vgate_vmix_ref`: layer 0 produce-only path (`xv == NULL` ->
    `v_out = v_first_out = v_in`, no gate read) vs layer >0
    (`v_gate = sigmoid((xv@v1)@v2+v0)`,
    `v_out = v_in + (v_first-v_in)*v_gate`, `v_first` passed through).
- Deliberately unfused; fusion is a later perf change behind its own parity
  gate (per plan discussion).

## Files
- `include/ckernel_rwkv_decode.h`, `src/kernels/rwkv_decode.c`,
  `unittest/test_rwkv_decode.py` (new `run_lora_case`: gates/vmix parity,
  layer-0 produce-only check, and gates->kk/k->vmix->WKV end-to-end).

## Verification
- `make build/libckernel_engine.so` — builds clean.
- `.venv/bin/python unittest/test_rwkv_decode.py` — all pass:
  - Step A cases unchanged (wkv_y `5.96e-08`–`2.38e-07`).
  - `lora H=4 N=8 R=(6,5,4,3)`: w/a/g/v/gate `~6e-08`, vf/l0 exact `0`,
    e2e_y `1.04e-07`, e2e_s `5.96e-08`.
  - `lora H=4 N=16 R=(8,6,5,4)`: w/a/g `3e-08`–`6e-08`, v/gate `8.94e-08`,
    e2e_y `2.98e-07`, e2e_s `1.19e-07`.
  - null guards return `-1`.
