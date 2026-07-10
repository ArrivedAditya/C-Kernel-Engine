"""Regression tests for llama.cpp-compatible FP16 split-KV decode attention.

The Qwen3-VL mixed-prefix decode path crosses a semantic boundary at KV=512:
llama.cpp rounds Q/K through FP16, computes FP16 online-softmax partials per
worker chunk, then combines those partials in FP32 chunk order.  A conventional
single FP32 reduction is mathematically reasonable, but is not the same model
contract and can change long-decode top-1 tokens.
"""

import ctypes
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np

from lib_loader import load_lib


lib = load_lib("libckernel_engine.so")

_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_U16_P = ctypes.POINTER(ctypes.c_uint16)
_SPLIT_ARGS = [
    _FLOAT_P, _U16_P, _U16_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.attention_forward_decode_head_major_gqa_flash_f16cache_split.argtypes = _SPLIT_ARGS
lib.attention_forward_decode_head_major_gqa_flash_f16cache_split.restype = None
lib.ck_get_num_threads.argtypes = []
lib.ck_get_num_threads.restype = ctypes.c_int


_LLAMA_HELPER_SOURCE = r"""
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool read_exact(const char * path, void * dst, size_t bytes) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    const size_t got = fread(dst, 1, bytes, f);
    fclose(f);
    return got == bytes;
}

static bool write_exact(const char * path, const void * src, size_t bytes) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;
    const size_t put = fwrite(src, 1, bytes, f);
    fclose(f);
    return put == bytes;
}

static size_t tensor_offset(const ggml_tensor * t, int i0, int i1, int i2, int i3) {
    return (size_t) i0 * t->nb[0] + (size_t) i1 * t->nb[1] +
           (size_t) i2 * t->nb[2] + (size_t) i3 * t->nb[3];
}

int main(int argc, char ** argv) {
    if (argc != 10) {
        fprintf(stderr, "usage: %s q.f32 k.f16 v.f16 out.f32 H Hkv KV D threads\n", argv[0]);
        return 2;
    }
    const int H = atoi(argv[5]);
    const int Hkv = atoi(argv[6]);
    const int KV = atoi(argv[7]);
    const int D = atoi(argv[8]);
    const int threads = atoi(argv[9]);
    if (H <= 0 || Hkv <= 0 || KV <= 0 || D <= 0 || threads <= 0 || H % Hkv != 0) return 2;

    const size_t q_count = (size_t) H * (size_t) D;
    const size_t kv_count = (size_t) Hkv * (size_t) KV * (size_t) D;
    const size_t memory = 64u * 1024u * 1024u +
                          q_count * sizeof(float) + 2u * kv_count * sizeof(ggml_fp16_t);
    ggml_init_params init = { memory, nullptr, false };
    ggml_context * ctx = ggml_init(init);
    if (!ctx) return 3;
    ggml_cpu_init();

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 1, H, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, KV, Hkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, KV, Hkv, 1);
    if (!read_exact(argv[1], q->data, q_count * sizeof(float)) ||
        !read_exact(argv[2], k->data, kv_count * sizeof(ggml_fp16_t)) ||
        !read_exact(argv[3], v->data, kv_count * sizeof(ggml_fp16_t))) {
        ggml_free(ctx);
        return 4;
    }

    ggml_tensor * out = ggml_flash_attn_ext(
        ctx, q, k, v, nullptr, 1.0f / sqrtf((float) D), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_threadpool_params tp_params = ggml_threadpool_params_default(threads);
    tp_params.paused = false;
    ggml_threadpool * threadpool = ggml_threadpool_new(&tp_params);
    ggml_cplan plan = ggml_graph_plan(graph, threads, threadpool);
    plan.work_data = plan.work_size ? (uint8_t *) malloc(plan.work_size) : nullptr;
    if (plan.work_size && !plan.work_data) return 5;
    const ggml_status status = ggml_graph_compute(graph, &plan);
    if (status != GGML_STATUS_SUCCESS) return 6;

    float * host = (float *) malloc(q_count * sizeof(float));
    if (!host) return 7;
    for (int h = 0; h < H; ++h) {
        for (int d = 0; d < D; ++d) {
            memcpy(host + (size_t) h * (size_t) D + (size_t) d,
                   (const char *) out->data + tensor_offset(out, d, h, 0, 0), sizeof(float));
        }
    }
    const bool ok = write_exact(argv[4], host, q_count * sizeof(float));
    free(host);
    free(plan.work_data);
    ggml_threadpool_free(threadpool);
    ggml_free(ctx);
    return ok ? 0 : 8;
}
"""


def _llama_root():
    value = os.environ.get("V8_QWEN3VL_ENCODER_PARITY_LLAMA_CPP_ROOT", "").strip()
    return Path(value).resolve() if value else (Path(__file__).resolve().parents[1] / "llama.cpp")


def _ensure_llama_helper():
    root = _llama_root()
    bin_dir = root / "build" / "bin"
    required = [bin_dir / "libggml.so", bin_dir / "libggml-cpu.so", bin_dir / "libggml-base.so"]
    if not (root / "ggml" / "include" / "ggml.h").is_file() or not all(p.is_file() for p in required):
        raise RuntimeError(f"llama.cpp GGML headers/libraries not found under {root}")
    helper_dir = Path(tempfile.gettempdir()) / "ck_attention_f16_split_kv"
    helper_dir.mkdir(parents=True, exist_ok=True)
    source = helper_dir / "llama_f16_split_kv.cpp"
    binary = helper_dir / "llama_f16_split_kv"
    if not source.is_file() or source.read_text() != _LLAMA_HELPER_SOURCE:
        source.write_text(_LLAMA_HELPER_SOURCE)
    if not binary.is_file() or binary.stat().st_mtime < source.stat().st_mtime:
        command = [
            "g++", "-O2", "-std=c++11",
            "-I", str(root / "ggml" / "include"),
            "-o", str(binary), str(source),
            "-L", str(bin_dir), "-lggml", "-lggml-cpu", "-lggml-base",
            "-lm", "-lpthread", f"-Wl,-rpath,{bin_dir}",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"llama.cpp split-KV helper compile failed:\n{completed.stderr}")
    return binary, bin_dir


def _llama_split_output(q, k_bits, v_bits, head_dim, threads):
    helper, bin_dir = _ensure_llama_helper()
    heads, _ = q.shape
    kv_heads, kv_tokens, _ = k_bits.shape
    q_compact = np.ascontiguousarray(q[:, :head_dim], dtype=np.float32)
    k_compact = np.ascontiguousarray(k_bits[:, :, :head_dim], dtype=np.uint16)
    v_compact = np.ascontiguousarray(v_bits[:, :, :head_dim], dtype=np.uint16)
    with tempfile.TemporaryDirectory(prefix="ck_f16_split_kv_") as tmp:
        tmp = Path(tmp)
        q_path, k_path, v_path, out_path = [tmp / name for name in ("q.f32", "k.f16", "v.f16", "out.f32")]
        q_compact.tofile(q_path)
        k_compact.tofile(k_path)
        v_compact.tofile(v_path)
        command = [
            str(helper), str(q_path), str(k_path), str(v_path), str(out_path),
            str(heads), str(kv_heads), str(kv_tokens), str(head_dim), str(threads),
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        completed = subprocess.run(command, capture_output=True, text=True, env=env)
        if completed.returncode != 0:
            raise RuntimeError(
                f"llama.cpp split-KV helper failed ({completed.returncode}):\n{completed.stderr}"
            )
        return np.fromfile(out_path, dtype=np.float32).reshape(heads, head_dim)


def _f32_ptr(a):
    return a.ctypes.data_as(_FLOAT_P)


def _u16_ptr(a):
    return a.ctypes.data_as(_U16_P)


def _half_bits(a):
    return np.ascontiguousarray(a.astype(np.float16)).view(np.uint16)


def _split_oracle(q, k_bits, v_bits, head_dim, chunks):
    """Independent scalar-contract oracle with explicit FP16 accumulators."""
    num_heads, aligned = q.shape
    num_kv_heads, kv_tokens, _ = k_bits.shape
    chunks = max(1, min(chunks, kv_tokens))
    chunk_size = (kv_tokens + chunks - 1) // chunks
    scale = np.float32(1.0 / math.sqrt(head_dim))
    q_half = q.astype(np.float16)
    k_half = k_bits.view(np.float16)
    v_half = v_bits.view(np.float16)
    partial_max = np.full((num_heads, chunks), -np.inf, dtype=np.float32)
    partial_sum = np.zeros((num_heads, chunks), dtype=np.float32)
    partial_acc = np.zeros((num_heads, chunks, aligned), dtype=np.float32)

    for h in range(num_heads):
        kv_head = h * num_kv_heads // num_heads
        q32 = q_half[h, :head_dim].astype(np.float32)
        for chunk in range(chunks):
            begin = chunk * chunk_size
            end = min(begin + chunk_size, kv_tokens)
            maximum = np.float32(-np.inf)
            total = np.float32(0.0)
            acc = np.zeros(head_dim, dtype=np.float16)
            for token in range(begin, end):
                # np.sum(dtype=float32) preserves the intended FP32 reduction
                # contract while keeping the real Qwen3-VL shape test practical.
                dot = np.sum(
                    q32 * k_half[kv_head, token, :head_dim].astype(np.float32),
                    dtype=np.float32,
                )
                score = np.float32(dot * scale)
                old_max = maximum
                max_scale = np.float32(1.0)
                value_scale = np.float32(1.0)
                if score > maximum:
                    maximum = score
                    max_scale = (
                        np.float32(math.exp(float(old_max - maximum)))
                        if np.isfinite(old_max)
                        else np.float32(0.0)
                    )
                    acc = (acc.astype(np.float32) * max_scale).astype(np.float16)
                else:
                    value_scale = np.float32(math.exp(float(score - maximum)))
                acc = (
                    acc.astype(np.float32)
                    + value_scale * v_half[kv_head, token, :head_dim].astype(np.float32)
                ).astype(np.float16)
                total = np.float32(total * max_scale + value_scale)

            partial_max[h, chunk] = maximum
            partial_sum[h, chunk] = total
            partial_acc[h, chunk, :head_dim] = acc.astype(np.float32)

    out = np.zeros((num_heads, aligned), dtype=np.float32)
    for h in range(num_heads):
        maximum = np.float32(-np.inf)
        total = np.float32(0.0)
        for chunk in range(chunks):
            chunk_sum = partial_sum[h, chunk]
            if chunk_sum == 0.0:
                continue
            new_max = np.float32(max(maximum, partial_max[h, chunk]))
            old_scale = (
                np.float32(math.exp(float(maximum - new_max)))
                if np.isfinite(maximum)
                else np.float32(0.0)
            )
            chunk_scale = np.float32(math.exp(float(partial_max[h, chunk] - new_max)))
            out[h, :head_dim] = (
                out[h, :head_dim] * old_scale
                + partial_acc[h, chunk, :head_dim] * chunk_scale
            ).astype(np.float32)
            total = np.float32(total * old_scale + chunk_sum * chunk_scale)
            maximum = new_max
        if total > 0.0:
            out[h, :head_dim] = (out[h, :head_dim] / total).astype(np.float32)
    return out


def _fp32_single_reference(q, k_bits, v_bits, head_dim):
    """The previous CK-style FP32 reduction, used only as a rejection oracle."""
    num_heads, aligned = q.shape
    num_kv_heads, _, _ = k_bits.shape
    k = k_bits.view(np.float16).astype(np.float32)
    v = v_bits.view(np.float16).astype(np.float32)
    out = np.zeros((num_heads, aligned), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(head_dim))
    for h in range(num_heads):
        kv_head = h * num_kv_heads // num_heads
        scores = np.sum(k[kv_head, :, :head_dim] * q[h, :head_dim], axis=1, dtype=np.float32)
        scores = scores * scale
        weights = np.exp(scores - np.max(scores)).astype(np.float32)
        weights /= np.sum(weights, dtype=np.float32)
        out[h, :head_dim] = weights @ v[kv_head, :, :head_dim]
    return out


def _inputs(seed, heads, kv_heads, kv_tokens, head_dim, aligned):
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 0.55, (heads, aligned)).astype(np.float32)
    k = rng.normal(0.0, 0.55, (kv_heads, kv_tokens, aligned)).astype(np.float32)
    v = rng.normal(0.0, 0.75, (kv_heads, kv_tokens, aligned)).astype(np.float32)
    if head_dim < aligned:
        q[:, head_dim:] = 0.0
        k[:, :, head_dim:] = 0.0
        v[:, :, head_dim:] = 0.0
    return q, _half_bits(k), _half_bits(v)


def _run_explicit(q, k, v, head_dim, chunks):
    heads, aligned = q.shape
    kv_heads, kv_tokens, _ = k.shape
    out = np.full_like(q, np.nan)
    lib.attention_forward_decode_head_major_gqa_flash_f16cache_split(
        _f32_ptr(q), _u16_ptr(k), _u16_ptr(v), _f32_ptr(out),
        heads, kv_heads, kv_tokens, kv_tokens, head_dim, aligned, chunks,
    )
    return out


@dataclass
class Result:
    name: str
    diff: float
    tolerance: float
    passed: bool


def _case(name, seed, heads, kv_heads, kv_tokens, head_dim, aligned, chunks, tolerance):
    q, k, v = _inputs(seed, heads, kv_heads, kv_tokens, head_dim, aligned)
    actual = _run_explicit(q, k, v, head_dim, chunks)
    expected = _split_oracle(q, k, v, head_dim, chunks)
    diff = float(np.max(np.abs(actual - expected)))
    return Result(name, diff, tolerance, diff <= tolerance), (q, k, v, actual)


def main():
    results = []
    below, below_data = _case(
        "f16_split_below_threshold(KV=511,C=1)", 511, 8, 2, 511, 64, 64, 1, 2.0e-5,
    )
    results.append(below)
    threshold, threshold_data = _case(
        "f16_split_threshold(KV=512,C=4)", 512, 8, 2, 512, 64, 64, 4, 2.0e-5,
    )
    results.append(threshold)
    qwen, qwen_data = _case(
        "f16_split_qwen3vl(KV=1058,H=32,D=128,C=20)",
        1058, 32, 8, 1058, 128, 128, 20, 2.0e-5,
    )
    results.append(qwen)
    padded, _ = _case(
        "f16_split_padded_gqa(KV=513,D=80,A=128,C=4)",
        513, 8, 2, 513, 80, 128, 4, 2.0e-5,
    )
    results.append(padded)

    threads = max(1, int(lib.ck_get_num_threads()))
    try:
        for name, data, chunks in (
            ("llama_oracle_below(KV=511)", below_data, 1),
            ("llama_oracle_threshold(KV=512)", threshold_data, threads),
            ("llama_oracle_qwen3vl(KV=1058)", qwen_data, threads),
        ):
            q, k, v, _ = data
            ck = _run_explicit(q, k, v, q.shape[1], chunks)
            llama = _llama_split_output(q, k, v, q.shape[1], threads)
            diff = float(np.max(np.abs(ck[:, :q.shape[1]] - llama)))
            results.append(Result(name, diff, 2.0e-5, diff <= 2.0e-5))
    except RuntimeError as exc:
        print(f"llama.cpp oracle error: {exc}")
        results.append(Result("llama_oracle_available", 1.0, 0.0, False))

    q, k, v, _ = qwen_data
    v_sensitive = _half_bits(v.view(np.float16).astype(np.float32) * np.float32(8.0))
    split_out = _run_explicit(q, k, v_sensitive, 128, 20)
    fp32_out = _fp32_single_reference(q, k, v_sensitive, 128)
    contract_gap = float(np.max(np.abs(split_out - fp32_out)))
    sensitivity_ok = contract_gap >= 5.0e-4
    results.append(Result(
        "reject_single_fp32_reduction(KV=1058)",
        0.0 if sensitivity_ok else 1.0,
        0.0,
        sensitivity_ok,
    ))

    print(f"FP16 split-KV attention contract: CK threads={threads}, fp32_contract_gap={contract_gap:.6e}")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{result.name}  max_diff={result.diff:.6e}  "
            f"tol={result.tolerance:.1e}  [{status}]"
        )
    passed = sum(result.passed for result in results)
    print(f"FP16 split-KV attention: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
