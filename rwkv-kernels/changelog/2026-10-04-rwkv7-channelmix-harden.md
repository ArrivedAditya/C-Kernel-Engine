# 2026-10-04 — RWKV-7 channelmix: caller scratch, aliasing contract, parity

## Change
- `ck_rwkv7_channelmix_decode` now takes caller-provided `scratch` holding
  `ck_rwkv7_channelmix_scratch_elems(dim, hidden_dim)` (= dim + hidden_dim)
  floats for the hoisted `xk` vector and `hidden` activations. The 32KB
  `hidden[8192]` stack array is gone (it risked stack overflow on
  small-stack lanes such as threadpool workers / embedded ARM).
- Fail-closed overlap rules (exact-pointer checks, `-1`): `shift_out ==
  shift_in` is rejected (it previously corrupted the shift read and silently
  computed the wrong answer), as are null/short/aliased scratch buffers.
  `shift_out == x`, `out == x`, and `shift_out == out` remain allowed and are
  safe by operation order (documented in the header).
- Hoisted the `xk = x + mu*(shift-x)` mix out of the `(K, C)` double loop
  (previously recomputed per element).
- `unittest/test_rwkv_decode.py`: new `run_channelmix_case` — torch-oracle
  parity for out/shift plus all four guard cases. Channelmix (and the
  token-shift binding) previously had zero coverage.

## Verification
- `make build/libckernel_engine.so` — builds clean.
- `.venv/bin/python unittest/test_rwkv_decode.py` — all pass:
  - `channelmix C=32 K=48`: out `6.56e-07`, shift exact, guards `-1,-1,-1,-1`.
  - `channelmix C=64 K=96`: out `1.43e-06`, shift exact, guards `-1,-1,-1,-1`.
  - all prior WKV/state/LoRA cases unchanged.
