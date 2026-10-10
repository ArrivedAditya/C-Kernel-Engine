# 2026-10-04 — RWKV-7 tests: token-shift parity, alias and carry chain

## Change
- `unittest/test_rwkv_decode.py`: new `run_shift_case` covering the
  previously untested `ck_rwkv7_token_shift_lerp`:
  - parity at `mu = 0 / 1 / 0.25` with distinct current/previous inputs
    (`xm` vs numpy, `shift_out == x` bit-exact).
  - in-place support: same buffer for `shift_in`/`shift_out` returns 0 with
    correct results (locks in the supported side of the alias contract;
    channelmix keeps rejecting it).
  - two-step carry chain proving the next shift equals the original current
    input (`carry == x0`, then `xm1 = x1 + 0.25*(x0-x1)`, `carry == x1`).
- Channelmix oracle (`xk = x+mu*(shift-x)`, `hidden = relu(Wk@xk)^2`,
  `expected = Wv@hidden`), explicit channelmix `argtypes`/`restype`, and
  exit-code propagation (`assert` + `ok` → `SystemExit(1)`) were already in
  place and are preserved unchanged.

## Verification
- `.venv/bin/python unittest/test_rwkv_decode.py` — all pass, exit 0:
  `mu=0/1/0.25` all bit-exact (`0.00e+00`), in-place and carry-chain PASS.
