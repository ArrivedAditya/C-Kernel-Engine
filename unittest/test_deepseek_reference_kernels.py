import ctypes
import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from lib_loader import load_lib


lib = load_lib("libckernel_engine.so")

fptr = ctypes.POINTER(ctypes.c_float)
iptr = ctypes.POINTER(ctypes.c_int)

lib.deepseek_mhc_mix_f32.argtypes = [fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.deepseek_mhc_mix_f32.restype = None
lib.deepseek_mhc_mix_backward_f32.argtypes = [fptr, fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.deepseek_mhc_mix_backward_f32.restype = None
lib.deepseek_dsa_topk_softmax_f32.argtypes = [fptr, iptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.deepseek_dsa_topk_softmax_f32.restype = None
lib.deepseek_dsa_topk_softmax_backward_f32.argtypes = [iptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.deepseek_dsa_topk_softmax_backward_f32.restype = None
lib.topk_softmax_backward_f32.argtypes = [iptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.topk_softmax_backward_f32.restype = None
lib.deepseek_csa_attention_f32.argtypes = [fptr, fptr, fptr, iptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
lib.deepseek_csa_attention_f32.restype = None
lib.deepseek_csa_attention_backward_f32.argtypes = [fptr, fptr, fptr, fptr, iptr, fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
lib.deepseek_csa_attention_backward_f32.restype = None
lib.deepseek_hybrid_attention_f32.argtypes = [fptr, fptr, fptr, iptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
lib.deepseek_hybrid_attention_f32.restype = None


def ptr(a):
    return a.ctypes.data_as(fptr)


def iptr_np(a):
    return a.ctypes.data_as(iptr)


class TestDeepSeekReferenceKernels(unittest.TestCase):
    def test_mhc_mix_forward_backward(self):
        torch.manual_seed(7)
        tokens, streams, dim = 3, 4, 5
        x = torch.randn(tokens, streams, dim, dtype=torch.float32, requires_grad=True)
        mix = torch.randn(tokens, streams, streams, dtype=torch.float32, requires_grad=True)
        y = torch.einsum("toi,tid->tod", mix, x)
        grad = torch.randn_like(y)
        y.backward(grad)

        x_np = np.ascontiguousarray(x.detach().numpy())
        mix_np = np.ascontiguousarray(mix.detach().numpy())
        out = np.empty_like(x_np)
        lib.deepseek_mhc_mix_f32(ptr(x_np), ptr(mix_np), ptr(out), tokens, streams, dim)
        np.testing.assert_allclose(out, y.detach().numpy(), rtol=1e-6, atol=1e-6)

        grad_np = np.ascontiguousarray(grad.numpy())
        dx = np.empty_like(x_np)
        dm = np.empty_like(mix_np)
        lib.deepseek_mhc_mix_backward_f32(ptr(grad_np), ptr(x_np), ptr(mix_np), ptr(dx), ptr(dm), tokens, streams, dim)
        np.testing.assert_allclose(dx, x.grad.numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(dm, mix.grad.numpy(), rtol=1e-6, atol=1e-6)

    def test_dsa_topk_softmax(self):
        torch.manual_seed(11)
        tokens, heads, keys, top_k = 2, 3, 7, 3
        scores = torch.randn(tokens, heads, keys, dtype=torch.float32)
        vals, idx = torch.topk(scores, top_k, dim=-1, largest=True, sorted=True)
        weights = torch.softmax(vals, dim=-1)

        scores_np = np.ascontiguousarray(scores.numpy())
        idx_np = np.empty((tokens, heads, top_k), dtype=np.int32)
        weights_np = np.empty((tokens, heads, top_k), dtype=np.float32)
        lib.deepseek_dsa_topk_softmax_f32(ptr(scores_np), iptr_np(idx_np), ptr(weights_np), tokens, heads, keys, top_k)
        np.testing.assert_array_equal(idx_np, idx.numpy().astype(np.int32))
        np.testing.assert_allclose(weights_np, weights.numpy(), rtol=1e-6, atol=1e-6)

    def test_topk_softmax_backward_matches_pytorch(self):
        torch.manual_seed(12)
        rows, n, top_k = 5, 9, 4
        scores = torch.randn(rows, n, dtype=torch.float32, requires_grad=True)
        values, indices = torch.topk(scores, top_k, dim=-1, largest=True, sorted=True)
        weights = torch.softmax(values, dim=-1)
        d_weights = torch.randn_like(weights)
        weights.backward(d_weights)

        idx_np = np.ascontiguousarray(indices.numpy().astype(np.int32))
        weights_np = np.ascontiguousarray(weights.detach().numpy())
        d_weights_np = np.ascontiguousarray(d_weights.numpy())
        d_scores = np.empty((rows, n), dtype=np.float32)
        lib.topk_softmax_backward_f32(iptr_np(idx_np), ptr(weights_np), ptr(d_weights_np), ptr(d_scores), rows, n, top_k)
        np.testing.assert_allclose(d_scores, scores.grad.numpy(), rtol=1e-6, atol=1e-6)

    def test_dsa_topk_softmax_backward_wrapper(self):
        torch.manual_seed(15)
        tokens, heads, keys, top_k = 2, 3, 8, 3
        scores = torch.randn(tokens, heads, keys, dtype=torch.float32, requires_grad=True)
        values, indices = torch.topk(scores, top_k, dim=-1, largest=True, sorted=True)
        weights = torch.softmax(values, dim=-1)
        d_weights = torch.randn_like(weights)
        weights.backward(d_weights)

        idx_np = np.ascontiguousarray(indices.numpy().astype(np.int32))
        weights_np = np.ascontiguousarray(weights.detach().numpy())
        d_weights_np = np.ascontiguousarray(d_weights.numpy())
        d_scores = np.empty((tokens, heads, keys), dtype=np.float32)
        lib.deepseek_dsa_topk_softmax_backward_f32(iptr_np(idx_np), ptr(weights_np), ptr(d_weights_np), ptr(d_scores), tokens, heads, keys, top_k)
        np.testing.assert_allclose(d_scores, scores.grad.numpy(), rtol=1e-6, atol=1e-6)

    def test_csa_attention_forward_backward(self):
        torch.manual_seed(13)
        tq, tk, heads, dim, top_k = 3, 5, 2, 4, 3
        scale = 1.0 / math.sqrt(dim)
        q = torch.randn(tq, heads, dim, dtype=torch.float32, requires_grad=True)
        k = torch.randn(tk, heads, dim, dtype=torch.float32, requires_grad=True)
        v = torch.randn(tk, heads, dim, dtype=torch.float32, requires_grad=True)
        indices = torch.tensor([
            [[0, 1, 3], [1, 2, 4]],
            [[1, 3, 4], [0, 2, 3]],
            [[0, 2, 4], [1, 3, 4]],
        ], dtype=torch.long)

        outs = []
        attns = []
        for ti in range(tq):
            per_head = []
            per_attn = []
            for h in range(heads):
                sel = indices[ti, h]
                logits = (q[ti, h].unsqueeze(0) * k[sel, h]).sum(-1) * scale
                a = torch.softmax(logits, dim=-1)
                per_attn.append(a)
                per_head.append((a.unsqueeze(-1) * v[sel, h]).sum(0))
            outs.append(torch.stack(per_head, dim=0))
            attns.append(torch.stack(per_attn, dim=0))
        ref = torch.stack(outs, dim=0)
        ref_attn = torch.stack(attns, dim=0)
        grad = torch.randn_like(ref)
        ref.backward(grad)

        q_np = np.ascontiguousarray(q.detach().numpy())
        k_np = np.ascontiguousarray(k.detach().numpy())
        v_np = np.ascontiguousarray(v.detach().numpy())
        idx_np = np.ascontiguousarray(indices.numpy().astype(np.int32))
        out_np = np.empty((tq, heads, dim), dtype=np.float32)
        attn_np = np.empty((tq, heads, top_k), dtype=np.float32)
        lib.deepseek_csa_attention_f32(ptr(q_np), ptr(k_np), ptr(v_np), iptr_np(idx_np), ptr(out_np), ptr(attn_np), tq, tk, heads, dim, top_k, scale)
        np.testing.assert_allclose(out_np, ref.detach().numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(attn_np, ref_attn.detach().numpy(), rtol=1e-5, atol=1e-6)

        grad_np = np.ascontiguousarray(grad.numpy())
        dq = np.empty_like(q_np)
        dk = np.empty_like(k_np)
        dv = np.empty_like(v_np)
        lib.deepseek_csa_attention_backward_f32(ptr(grad_np), ptr(q_np), ptr(k_np), ptr(v_np), iptr_np(idx_np), ptr(attn_np), ptr(dq), ptr(dk), ptr(dv), tq, tk, heads, dim, top_k, scale)
        np.testing.assert_allclose(dq, q.grad.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dk, k.grad.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dv, v.grad.numpy(), rtol=1e-5, atol=1e-6)

    def test_hybrid_dense_mode_matches_dense_selection(self):
        torch.manual_seed(17)
        tq, tk, heads, dim = 2, 4, 2, 3
        scale = 1.0 / math.sqrt(dim)
        q = np.ascontiguousarray(torch.randn(tq, heads, dim).numpy().astype(np.float32))
        k = np.ascontiguousarray(torch.randn(tk, heads, dim).numpy().astype(np.float32))
        v = np.ascontiguousarray(torch.randn(tk, heads, dim).numpy().astype(np.float32))
        dense_idx = np.tile(np.arange(tk, dtype=np.int32), (tq, heads, 1)).copy()
        out_a = np.empty((tq, heads, dim), dtype=np.float32)
        attn_a = np.empty((tq, heads, tk), dtype=np.float32)
        out_b = np.empty_like(out_a)
        attn_b = np.empty_like(attn_a)
        lib.deepseek_hybrid_attention_f32(ptr(q), ptr(k), ptr(v), iptr_np(dense_idx), ptr(out_a), ptr(attn_a), tq, tk, heads, dim, tk, scale, 0)
        lib.deepseek_csa_attention_f32(ptr(q), ptr(k), ptr(v), iptr_np(dense_idx), ptr(out_b), ptr(attn_b), tq, tk, heads, dim, tk, scale)
        np.testing.assert_allclose(out_a, out_b, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(attn_a, attn_b, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
