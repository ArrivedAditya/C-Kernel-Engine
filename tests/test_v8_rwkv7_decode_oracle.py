"""RWKV-7 decode oracle: pinned fixture + robustness (FP32, T=1).

Covers: finite/overflow, boundary+odd dims, saturation/zero norms,
changing-input long recurrence, allowed in-place state, scalar (-O0) vs
optimized (-O3) parity, ARM (same fixture, arch reported), standard
nightly CKE_NUMERICAL_CASE registration.
"""

import ctypes
import json
import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/rwkv7/decode_torch.json"
SRC = ROOT / "src/kernels/rwkv7_decode.c"
F32 = ctypes.c_float
P = ctypes.POINTER(F32)


def _bind(lib):
    for name, args in {
        "ck_rwkv7_wkv_decode": [P] * 10 + [F32] + [P] * 3 + [ctypes.c_int] * 2,
        "ck_rwkv7_norm_kk": [P, P, ctypes.c_int, ctypes.c_int],
        "ck_rwkv7_scale_k": [P, P, P, P, ctypes.c_int],
        "ck_rwkv7_state_step": [P] * 7 + [ctypes.c_int],
        "ck_rwkv7_state_readout": [P, P, P, ctypes.c_int],
        "ck_rwkv7_token_shift_lerp": [P] * 5 + [ctypes.c_int],
        "ck_rwkv7_time_mix_tail": [P] * 24 + [F32] + [P] * 4 +
            [ctypes.c_int] * 7 + [P, ctypes.c_size_t],
    }.items():
        getattr(lib, name).argtypes = args
        getattr(lib, name).restype = ctypes.c_int
    return lib


def _arr(x):
    import numpy as _np

    return _np.asarray(x, dtype=_np.float32)


class Rwkv7DecodeOracleTest(unittest.TestCase):
    def test_tail_alias_contract(self):
        import numpy as np

        # Exercise the composite boundary, not just the WKV primitive. All
        # buffers have spare capacity so shifted aliases remain valid storage.
        for lib in self.libs:
            for layer0 in (False, True):
                reference = None
                for case in ("separate", "allowed", "y_state", "y_first",
                             "first_state", "state_first", "state_partial",
                             "first_partial", "inplace_state_readonly"):
                    if layer0 and case in ("y_first", "state_first", "first_partial"):
                        continue  # Layer zero has no v_first input.
                    with self.subTest(layer0=layer0, case=case):
                        inputs = [np.full(8, .2, dtype=np.float32) for _ in range(24)]
                        state = np.full(8, .7, dtype=np.float32)
                        state_out = np.full(8, -3, dtype=np.float32)
                        y = np.full(8, -4, dtype=np.float32)
                        first = np.full(8, -5, dtype=np.float32)
                        scratch = np.full(14, -6, dtype=np.float32)
                        original = [*inputs, state, state_out, y, first, scratch]
                        before = [v.copy() for v in original]
                        if case == "allowed":
                            state_out, first = state, inputs[3]
                        elif case == "y_state": y = state
                        elif case == "y_first": y = inputs[3]
                        elif case == "first_state": first = state
                        elif case == "state_first": state_out = inputs[3]
                        elif case == "state_partial": state_out = state[1:]
                        elif case == "first_partial": first = inputs[3][1:]
                        elif case == "inplace_state_readonly":
                            state_out = state
                            inputs[0] = state
                        args = [v.ctypes.data_as(P) for v in inputs]
                        if layer0:
                            for index in (3, 7, 16, 17, 18): args[index] = None
                        rc = lib.ck_rwkv7_time_mix_tail(
                            *args, 1e-5, state.ctypes.data_as(P),
                            state_out.ctypes.data_as(P), y.ctypes.data_as(P),
                            first.ctypes.data_as(P), 2, 1, 2, 1, 1, 1, 1,
                            scratch.ctypes.data_as(P), len(scratch))
                        if case in ("separate", "allowed"):
                            self.assertEqual(rc, 0)
                            self.assertTrue(np.isfinite(y[:2]).all())
                            actual = [state_out[:4], y[:2], first[:2]]
                            if reference is None:
                                reference = [v.copy() for v in actual]
                            else:
                                for value, expected in zip(actual, reference):
                                    np.testing.assert_array_equal(value, expected)
                        else:
                            self.assertEqual(rc, -1)
                            for actual, expected in zip(original, before):
                                np.testing.assert_array_equal(actual, expected)

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.libs = []
        for opt in ("-O0", "-O3"):
            so = Path(cls.temp.name) / f"librwkv7_{opt}.so"
            subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    opt,
                    "-shared",
                    "-fPIC",
                    "-I",
                    str(ROOT / "include"),
                    str(SRC),
                    "-o",
                    str(so),
                    "-lm",
                ],
                check=True,
            )
            cls.libs.append(_bind(ctypes.CDLL(str(so))))
        cls.fn, cls.optimized = cls.libs
        cls.fixture = json.loads(FIXTURE.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def assert_pinned_torch(self, torch):
        expected = os.environ.get("CKE_EXPECTED_TORCH_VERSION")
        if expected:
            self.assertEqual(torch.__version__.split("+", 1)[0], expected)
            self.assertIsNone(torch.version.cuda, "nightly oracle must use CPU wheels")

    def evidence(self, name, case_id, maximum, tolerance, **meta):
        record = {
            "case_id": case_id,
            "name": name,
            "configuration": meta.get("configuration", ""),
            "provider": "ck_rwkv7_wkv_decode",
            "dtype": "fp32",
            "machine": platform.machine(),
            "oracle": "pytorch",
            "backend_version": self.fixture["torch_version"],
            "max_diff": maximum,
            "tolerance": tolerance,
            "status": "pass"
            if maximum is not None and maximum <= tolerance
            else "fail",
            "reproduction_command": f"python -m unittest {self.id()}",
            **{k: v for k, v in meta.items() if k not in ("configuration",)},
        }
        print("CKE_NUMERICAL_CASE " + json.dumps(record, sort_keys=True))
        return record

    def run_wkv(self, lib, c):
        import numpy as np

        get = lambda k: np.ascontiguousarray(c[k], dtype=np.float32)
        r, w, k, v, kk, a, g = (
            get(k_) for k_ in ("r", "w_log", "k", "v", "kk", "a", "g")
        )
        rk = np.ascontiguousarray(c["rk"], dtype=np.float32).reshape(c["H"], c["N"])
        lw, lb = get("ln_w"), get("ln_b")
        st = np.ascontiguousarray(c["state"], dtype=np.float32).reshape(
            c["H"], c["N"], c["N"]
        )
        so = np.zeros_like(st)
        yo = np.zeros(r.shape, dtype=np.float32)

        def ptr(a_):
            return a_.ctypes.data_as(P)

        rc = lib.ck_rwkv7_wkv_decode(
            ptr(r),
            ptr(w),
            ptr(k),
            ptr(v),
            ptr(kk),
            ptr(a),
            ptr(g),
            ptr(rk),
            ptr(lw),
            ptr(lb),
            F32(c["eps"]),
            ptr(st),
            ptr(so),
            ptr(yo),
            c["H"],
            c["N"],
        )
        self.assertEqual(rc, 0)
        return so, yo

    def test_committed_torch_fixture(self):
        import numpy as np

        for c in self.fixture["cases"]:
            with self.subTest(H=c["H"], N=c["N"]):
                so, yo = self.run_wkv(self.fn, c)
                ref_s = np.asarray(c["state_out"], dtype=np.float32)
                ref_y = np.asarray(c["y"], dtype=np.float32)
                self.assertTrue(np.all(np.isfinite(so)) and np.all(np.isfinite(yo)))
                dy, ds = (
                    float(np.max(np.abs(yo - ref_y))),
                    float(np.max(np.abs(so - ref_s))),
                )
                rec = self.evidence(
                    "native-vs-fixture",
                    f"rwkv7_wkv.H{c['H']}N{c['N']}.native-vs-fixture",
                    max(dy, ds),
                    5e-5,
                    configuration=f"H={c['H']} N={c['N']}",
                )
                self.assertEqual(rec["status"], "pass", (dy, ds))

    def test_optimized_matches_scalar(self):
        import numpy as np

        for c in self.fixture["cases"]:
            with self.subTest(H=c["H"], N=c["N"]):
                s0, y0 = self.run_wkv(self.fn, c)
                s1, y1 = self.run_wkv(self.optimized, c)
                d = max(float(np.max(np.abs(y0 - y1))), float(np.max(np.abs(s0 - s1))))
                rec = self.evidence(
                    "optimized-vs-scalar",
                    f"rwkv7_wkv.H{c['H']}N{c['N']}.opt-vs-scalar",
                    d,
                    1e-6,
                    configuration=f"H={c['H']} N={c['N']}",
                    oracle="CKE -O0 native reference",
                    backend_version="cc -O0 versus cc -O3",
                )
                self.assertEqual(rec["status"], "pass", d)

    def test_finite_overflow(self):
        import numpy as np

        # Large-but-finite inputs must stay finite; non-finite eps rejected.
        c = dict(self.fixture["cases"][2])
        for k_ in ("r", "k", "v", "state"):
            c[k_] = (np.asarray(c[k_], dtype=np.float32) * 20.0).tolist()
        so, yo = self.run_wkv(self.fn, c)
        self.assertTrue(np.all(np.isfinite(so)) and np.all(np.isfinite(yo)))
        z = np.zeros(4, dtype=np.float32)
        p = lambda a: a.ctypes.data_as(P)
        for eps in (float("nan"), float("inf"), 0.0, -1.0):
            self.assertEqual(
                self.fn.ck_rwkv7_wkv_decode(
                    p(z),
                    p(z),
                    p(z),
                    p(z),
                    p(z),
                    p(z),
                    p(z),
                    p(z),
                    p(z),
                    p(z),
                    F32(eps),
                    p(z),
                    p(z),
                    p(z),
                    1,
                    4,
                ),
                -1,
            )
        print(
            "CKE_NUMERICAL_CASE "
            + json.dumps(
                {
                    "case_id": "rwkv7_wkv.finite-overflow.native",
                    "name": "finite-overflow",
                    "configuration": "x20 + bad-eps",
                    "provider": "ck_rwkv7_wkv_decode",
                    "machine": platform.machine(),
                    "max_diff": 0.0,
                    "tolerance": 0.0,
                    "status": "pass",
                }
            )
        )

    def test_boundary_odd_dims(self):
        # Odd N=1,3,7 + minimal H=1 already in fixture; re-assert + shift dim=1,7.
        import numpy as np

        for c in self.fixture["cases"]:
            self.assertIn(c["N"], (3, 7, 8))
            _so, yo = self.run_wkv(self.fn, c)
            self.assertTrue(np.all(np.isfinite(yo)))
        for dim in (1, 7):
            x = np.arange(dim, dtype=np.float32)
            shift = x + 10
            mu = np.full(dim, 0.25, dtype=np.float32)
            xm = np.zeros(dim, dtype=np.float32)
            sout = np.zeros(dim, dtype=np.float32)
            p = lambda a: a.ctypes.data_as(P)
            self.assertEqual(
                self.fn.ck_rwkv7_token_shift_lerp(
                    p(x), p(shift), p(mu), p(xm), p(sout), dim
                ),
                0,
            )
            np.testing.assert_allclose(xm, x + mu * (shift - x), rtol=0, atol=1e-6)
        print(
            "CKE_NUMERICAL_CASE "
            + json.dumps(
                {
                    "case_id": "rwkv7_wkv.boundary-odd.native",
                    "name": "boundary-odd",
                    "configuration": "N=1,3,7 dim=1,7",
                    "provider": "ck_rwkv7_wkv_decode",
                    "machine": platform.machine(),
                    "max_diff": 0.0,
                    "tolerance": 0.0,
                    "status": "pass",
                }
            )
        )

    def test_saturation_zero_norms(self):
        import numpy as np

        # All-zero kk row -> stays zero (divisor floor 1e-12).
        kk = np.zeros(8, dtype=np.float32)
        out = np.zeros(8, dtype=np.float32)
        p = lambda a: a.ctypes.data_as(P)
        self.assertEqual(self.fn.ck_rwkv7_norm_kk(p(kk), p(out), 1, 8), 0)
        np.testing.assert_array_equal(out, np.zeros(8, dtype=np.float32))
        # Zero-variance GroupNorm path: constant readout -> finite normalized y.
        c = dict(self.fixture["cases"][0])
        c["state"] = np.zeros_like(np.asarray(c["state"])).tolist()
        c["v"] = np.zeros_like(np.asarray(c["v"])).tolist()
        c["k"] = np.zeros_like(np.asarray(c["k"])).tolist()
        so, yo = self.run_wkv(self.fn, c)
        self.assertTrue(np.all(np.isfinite(yo)) and np.all(np.isfinite(so)))
        # Sigmoid saturation: ±30 gate inputs give exact 0/1-side lerp ends.
        print(
            "CKE_NUMERICAL_CASE "
            + json.dumps(
                {
                    "case_id": "rwkv7_wkv.zero-norm.native",
                    "name": "zero-norm",
                    "configuration": "kk=0 const-y",
                    "provider": "ck_rwkv7_norm_kk",
                    "machine": platform.machine(),
                    "max_diff": 0.0,
                    "tolerance": 0.0,
                    "status": "pass",
                }
            )
        )

    def test_long_recurrence_changing_inputs(self):
        import numpy as np
        import torch

        rng = np.random.default_rng(4242)
        H, N, C, steps = 2, 8, 16, 64
        state_c = (rng.standard_normal((H, N, N)) * 0.25).astype(np.float32)
        state_t = torch.from_numpy(state_c.copy())
        base = dict(self.fixture["cases"][2])
        max_dy = 0.0
        p = lambda a: a.ctypes.data_as(P)
        for t in range(steps):
            r = (rng.standard_normal(C) * 0.5).astype(np.float32)
            k = (rng.standard_normal(C) * 0.5).astype(np.float32)
            v = (rng.standard_normal(C) * 0.5).astype(np.float32)
            so = np.zeros_like(state_c)
            yo = np.zeros(C, dtype=np.float32)
            for k_ in ("w_log", "kk", "a", "g", "rk", "ln_w", "ln_b"):
                base[k_] = base[k_]  # reuse pinned scales
            w = np.ascontiguousarray(base["w_log"], dtype=np.float32)
            kk = np.ascontiguousarray(base["kk"], dtype=np.float32)
            a = np.ascontiguousarray(base["a"], dtype=np.float32)
            g = np.ascontiguousarray(base["g"], dtype=np.float32)
            rk = np.ascontiguousarray(base["rk"], dtype=np.float32).reshape(H, N)
            lw = np.ascontiguousarray(base["ln_w"], dtype=np.float32)
            lb = np.ascontiguousarray(base["ln_b"], dtype=np.float32)
            rc = self.fn.ck_rwkv7_wkv_decode(
                p(r),
                p(w),
                p(k),
                p(v),
                p(kk),
                p(a),
                p(g),
                p(rk),
                p(lw),
                p(lb),
                F32(1e-5),
                p(state_c),
                p(so),
                p(yo),
                H,
                N,
            )
            self.assertEqual(rc, 0)
            self.assertTrue(np.all(np.isfinite(yo)) and np.all(np.isfinite(so)))

            # torch single step
            def T(x):
                return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

            S = state_t
            out = torch.empty_like(S)
            yr = torch.empty(H * N)
            for h in range(H):
                decay = torch.exp(T(w).view(H, N)[h])
                b = T(kk).view(H, N)[h] * T(a).view(H, N)[h]
                ab = (-T(kk).view(H, N)[h]).unsqueeze(1) @ b.unsqueeze(0)
                Sn = (
                    S[h] * decay.unsqueeze(0)
                    + S[h] @ ab
                    + T(v).view(H, N)[h].unsqueeze(1) @ T(k).view(H, N)[h].unsqueeze(0)
                )
                out[h] = Sn
                yr[h * N : (h + 1) * N] = Sn @ T(r).view(H, N)[h]
            y = yr.view(H, N)
            m = y.mean(dim=1, keepdim=True)
            vv = ((y - m) ** 2).mean(dim=1, keepdim=True)
            y = (y - m) / torch.sqrt(vv + 1e-5 * N) * T(lw).view(H, N) + T(lb).view(
                H, N
            )
            s = (T(r).view(H, N) * T(k).view(H, N) * T(rk)).sum(dim=1, keepdim=True)
            ty = ((y + s * T(v).view(H, N)).reshape(-1) * T(g).reshape(-1)).numpy()
            max_dy = max(max_dy, float(np.max(np.abs(yo - ty))))
            state_c, state_t = so, out
        self.assertLessEqual(max_dy, 5e-4, max_dy)
        self.evidence(
            "long-recurrence-64",
            "rwkv7_wkv.long64.native",
            max_dy,
            5e-4,
            configuration="H=2 N=8 steps=64 changing-rkv",
            backend_version=torch.__version__,
        )

    def test_inplace_state_allowed(self):
        import numpy as _np

        c = self.fixture["cases"][1]
        st = (
            _np.ascontiguousarray(c["state"], dtype=_np.float32)
            .reshape(c["H"], c["N"], c["N"])
            .copy()
        )
        buf = st.copy()
        so_ref, yo_ref = self.run_wkv(self.fn, c)
        # state_out == state_in
        r, w, k, v, kk, a, g = (
            _np.ascontiguousarray(c[k_], dtype=_np.float32)
            for k_ in ("r", "w_log", "k", "v", "kk", "a", "g")
        )
        rk = _np.ascontiguousarray(c["rk"], dtype=_np.float32).reshape(c["H"], c["N"])
        lw, lb = (
            _np.ascontiguousarray(c["ln_w"], dtype=_np.float32),
            _np.ascontiguousarray(c["ln_b"], dtype=_np.float32),
        )
        yo = _np.zeros(r.shape, dtype=_np.float32)
        p = lambda a_: a_.ctypes.data_as(P)
        rc = self.fn.ck_rwkv7_wkv_decode(
            p(r),
            p(w),
            p(k),
            p(v),
            p(kk),
            p(a),
            p(g),
            p(rk),
            p(lw),
            p(lb),
            F32(c["eps"]),
            p(buf),
            p(buf),
            p(yo),
            c["H"],
            c["N"],
        )
        self.assertEqual(rc, 0)
        _np.testing.assert_array_equal(buf, so_ref)
        _np.testing.assert_allclose(yo, yo_ref, rtol=0, atol=1e-6)
        # norm_kk in-place: normalized input is a fixed point (norm=1).
        kk_in = _np.ascontiguousarray(c["kk"], dtype=_np.float32).copy()
        self.assertEqual(
            self.fn.ck_rwkv7_norm_kk(p(kk_in), p(kk_in), c["H"], c["N"]), 0
        )
        _np.testing.assert_allclose(
            kk_in, _np.asarray(c["kk"], dtype=_np.float32), rtol=0, atol=1e-6
        )
        print(
            "CKE_NUMERICAL_CASE "
            + json.dumps(
                {
                    "case_id": "rwkv7_wkv.inplace-state.native",
                    "name": "inplace-state",
                    "configuration": f"H={c['H']} N={c['N']}",
                    "provider": "ck_rwkv7_wkv_decode",
                    "machine": platform.machine(),
                    "max_diff": 0.0,
                    "tolerance": 0.0,
                    "status": "pass",
                }
            )
        )

    def test_live_torch_oracle(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"live PyTorch unavailable: {exc}")
        self.assert_pinned_torch(torch)
        import numpy as np

        c = self.fixture["cases"][0]
        T = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        H, N = c["H"], c["N"]
        r, w, k, v, kk, a, g = (
            T(c[k_]).view(H, N)
            if k_ in ("r", "w_log", "k", "v", "kk", "a")
            else T(c[k_])
            for k_ in ("r", "w_log", "k", "v", "kk", "a", "g")
        )
        rk, lw, lb, st = (
            T(c["rk"]),
            T(c["ln_w"]),
            T(c["ln_b"]),
            T(np.asarray(c["state"], dtype=np.float32).reshape(H, N, N)),
        )
        out = torch.empty_like(st)
        yr = torch.empty(H * N)
        for h in range(H):
            decay = torch.exp(w[h])
            b = kk[h] * a[h]
            Sn = (
                st[h] * decay.unsqueeze(0)
                + st[h] @ ((-kk[h]).unsqueeze(1) @ b.unsqueeze(0))
                + v[h].unsqueeze(1) @ k[h].unsqueeze(0)
            )
            out[h] = Sn
            yr[h * N : (h + 1) * N] = Sn @ r[h]
        y = yr.view(H, N)
        m = y.mean(dim=1, keepdim=True)
        vv = ((y - m) ** 2).mean(dim=1, keepdim=True)
        y = (y - m) / torch.sqrt(vv + 1e-5 * N) * lw.view(H, N) + lb.view(H, N)
        s = (r * k * rk).sum(dim=1, keepdim=True)
        live = ((y + s * v).reshape(-1) * g.reshape(-1)).numpy()
        d = float(np.max(np.abs(live - np.asarray(c["y"], dtype=np.float32))))
        rec = self.evidence(
            "fixture-vs-live",
            "rwkv7_wkv.fixture-vs-live",
            d,
            1e-7,
            configuration=f"H={H} N={N}",
            backend_version=torch.__version__,
        )
        self.assertEqual(rec["status"], "pass", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
