#!/usr/bin/env python3
"""Byte-exact PyTorch CPU native-GQA decode attention contract."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = Path(
    os.environ.get("CK_ENGINE_SO", ROOT / "build" / "libckernel_engine.so")
).resolve()
TORCH_CPU = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cpu.so"
os.environ.setdefault("CK_SLEEF_LIBRARY", str(TORCH_CPU))

LIB = ctypes.CDLL(str(LIB_PATH))
KERNEL = LIB.attention_forward_decode_head_major_gqa_bf16cache_pytorch_contract
FLOAT_P = ctypes.POINTER(ctypes.c_float)
U16_P = ctypes.POINTER(ctypes.c_uint16)
KERNEL.argtypes = [
    FLOAT_P,
    U16_P,
    U16_P,
    FLOAT_P,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
KERNEL.restype = ctypes.c_int
PREFILL_KERNEL = (
    LIB.attention_forward_causal_head_major_gqa_prefill_append_bf16cache_pytorch_contract
)
PREFILL_KERNEL.argtypes = [
    FLOAT_P,
    U16_P,
    U16_P,
    FLOAT_P,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
PREFILL_KERNEL.restype = ctypes.c_int
FULL_PREFILL_KERNEL = (
    LIB.attention_forward_causal_head_major_gqa_prefill_full_bf16cache_pytorch_contract
)
FULL_PREFILL_KERNEL.argtypes = PREFILL_KERNEL.argtypes
FULL_PREFILL_KERNEL.restype = ctypes.c_int

CK_ATTN_REDUCTION_BF16_PYTORCH_SDPA = 4
CK_ATTENTION_STATUS_OK = 0


def cpu_flags() -> set[str]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return set()
    for line in cpuinfo.read_text().splitlines():
        if line.startswith("flags"):
            return set(line.partition(":")[2].split())
    return set()


def run_case(kv_tokens: int, seed: int) -> None:
    heads = 32
    kv_heads = 8
    head_dim = 128
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(
        (1, heads, 1, head_dim), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    k = torch.randn(
        (1, kv_heads, kv_tokens, head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    v = torch.randn(
        (1, kv_heads, kv_tokens, head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        enable_gqa=True,
        scale=head_dim**-0.5,
    )[0, :, 0, :].float().numpy()
    q_f32 = q.float().numpy().reshape(-1).copy()
    k_bf16 = k.view(torch.uint16).numpy().reshape(-1).copy()
    v_bf16 = v.view(torch.uint16).numpy().reshape(-1).copy()
    actual = np.zeros((heads, head_dim), dtype=np.float32)

    status = KERNEL(
        q_f32.ctypes.data_as(FLOAT_P),
        k_bf16.ctypes.data_as(U16_P),
        v_bf16.ctypes.data_as(U16_P),
        actual.ctypes.data_as(FLOAT_P),
        heads,
        kv_heads,
        kv_tokens,
        kv_tokens,
        head_dim,
        head_dim,
        CK_ATTN_REDUCTION_BF16_PYTORCH_SDPA,
    )
    if status != CK_ATTENTION_STATUS_OK:
        raise AssertionError(
            f"native-GQA provider rejected KV={kv_tokens}: status={status}"
        )

    actual_bits = actual.view(np.uint32)
    expected_bits = expected.view(np.uint32)
    mismatch = np.flatnonzero(actual_bits != expected_bits)
    if mismatch.size:
        flat = int(mismatch[0])
        head, channel = np.unravel_index(flat, actual.shape)
        raise AssertionError(
            "BF16 native-GQA decode mismatch "
            f"KV={kv_tokens} head={head} channel={channel} "
            f"actual={actual[head, channel]:.9g} "
            f"expected={expected[head, channel]:.9g} "
            f"mismatches={mismatch.size}/{actual.size}"
        )
    print(f"PASS KV={kv_tokens}: byte-exact {actual.size}/{actual.size}")


def run_prefill_tail_case(seed: int) -> None:
    heads = 32
    kv_heads = 8
    head_dim = 128
    kv_tokens = 4
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(
        (1, heads, 1, head_dim), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    k = torch.randn(
        (1, kv_heads, kv_tokens, head_dim), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    v = torch.randn(
        (1, kv_heads, kv_tokens, head_dim), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, enable_gqa=True, scale=head_dim**-0.5
    )[0].float().numpy()
    q_f32 = q.float().numpy()[0].copy()
    k_bf16 = k.view(torch.uint16).numpy().reshape(-1).copy()
    v_bf16 = v.view(torch.uint16).numpy().reshape(-1).copy()
    actual = np.zeros((heads, 1, head_dim), dtype=np.float32)
    status = PREFILL_KERNEL(
        q_f32.ctypes.data_as(FLOAT_P),
        k_bf16.ctypes.data_as(U16_P),
        v_bf16.ctypes.data_as(U16_P),
        actual.ctypes.data_as(FLOAT_P),
        heads,
        kv_heads,
        1,
        3,
        kv_tokens,
        head_dim,
        head_dim,
        CK_ATTN_REDUCTION_BF16_PYTORCH_SDPA,
    )
    if status != CK_ATTENTION_STATUS_OK:
        raise AssertionError(f"prefill provider rejected scalar-tail case: status={status}")
    mismatch = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
    if mismatch.size:
        flat = int(mismatch[0])
        head, token, channel = np.unravel_index(flat, actual.shape)
        raise AssertionError(
            "BF16 native-GQA prefill scalar-tail mismatch "
            f"head={head} token={token} channel={channel} "
            f"actual={actual[head, token, channel]:.9g} "
            f"expected={expected[head, token, channel]:.9g} "
            f"mismatches={mismatch.size}/{actual.size}"
        )
    print(f"PASS prefill past=3 q=1: byte-exact {actual.size}/{actual.size}")


def run_full_prefill_case(tokens: int, seed: int) -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    heads = 32
    kv_heads = 8
    head_dim = 128
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(
        (1, heads, tokens, head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    k = torch.randn(
        (1, kv_heads, tokens, head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    v = torch.randn(
        (1, kv_heads, tokens, head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=True,
            scale=head_dim**-0.5,
        )[0].float().numpy()
    q_f32 = q.float().numpy()[0].copy()
    k_bf16 = k.view(torch.uint16).numpy().reshape(-1).copy()
    v_bf16 = v.view(torch.uint16).numpy().reshape(-1).copy()
    actual = np.zeros((heads, tokens, head_dim), dtype=np.float32)
    status = FULL_PREFILL_KERNEL(
        q_f32.ctypes.data_as(FLOAT_P),
        k_bf16.ctypes.data_as(U16_P),
        v_bf16.ctypes.data_as(U16_P),
        actual.ctypes.data_as(FLOAT_P),
        heads,
        kv_heads,
        tokens,
        0,
        tokens,
        head_dim,
        head_dim,
        CK_ATTN_REDUCTION_BF16_PYTORCH_SDPA,
    )
    if status != CK_ATTENTION_STATUS_OK:
        raise AssertionError(
            f"full-matrix prefill provider rejected T={tokens}: status={status}"
        )
    mismatch = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
    if mismatch.size:
        flat = int(mismatch[0])
        head, token, channel = np.unravel_index(flat, actual.shape)
        delta = np.abs(actual - expected)
        raise AssertionError(
            "BF16 native-GQA full-matrix prefill mismatch "
            f"T={tokens} head={head} token={token} channel={channel} "
            f"actual={actual[head, token, channel]:.9g} "
            f"expected={expected[head, token, channel]:.9g} "
            f"max_abs={float(delta.max()):.9g} "
            f"mismatches={mismatch.size}/{actual.size}"
        )
    print(
        f"PASS full-matrix prefill T={tokens}: "
        f"byte-exact {actual.size}/{actual.size}"
    )


def run_qwen3vl_activation_case(checkpoint: Path, image_path: Path) -> None:
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3vl

    prompt = os.environ.get(
        "CK_QWEN3VL_PROMPT",
        "Extract visible form fields as compact JSON.",
    )
    processor = AutoProcessor.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).eval()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[Image.open(image_path).convert("RGB")],
        return_tensors="pt",
        min_pixels=1,
        max_pixels=1048576,
    )

    original_attention = qwen3vl.ALL_ATTENTION_FUNCTIONS["sdpa"]
    capture: dict[str, torch.Tensor] = {}

    def capture_attention(
        module,
        query,
        key,
        value,
        attention_mask,
        **kwargs,
    ):
        result = original_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            **kwargs,
        )
        if (
            not capture
            and getattr(module, "layer_idx", -1) == 0
            and query.shape[-1] == 128
        ):
            if attention_mask is not None:
                raise AssertionError(
                    "real-activation oracle currently requires causal SDPA "
                    "without an explicit additive mask"
                )
            capture["q"] = query.detach().cpu()
            capture["k"] = key.detach().cpu()
            capture["v"] = value.detach().cpu()
            capture["output"] = result[0].detach().cpu()
        return result

    qwen3vl.ALL_ATTENTION_FUNCTIONS["sdpa"] = capture_attention
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=True)
    finally:
        qwen3vl.ALL_ATTENTION_FUNCTIONS["sdpa"] = original_attention
    if not capture:
        raise AssertionError("failed to capture Qwen3-VL layer-0 SDPA inputs")

    q = capture["q"]
    k = capture["k"]
    v = capture["v"]
    expected = capture["output"].transpose(1, 2).contiguous().float().numpy()[0]
    _, heads, tokens, head_dim = q.shape
    kv_heads = int(k.shape[1])
    if k.shape[2] != tokens or v.shape[2] != tokens:
        raise AssertionError(
            f"unexpected Q/K/V token shapes: {q.shape}, {k.shape}, {v.shape}"
        )

    q_f32 = q.float().numpy()[0].copy()
    k_bf16 = k.view(torch.uint16).numpy().reshape(-1).copy()
    v_bf16 = v.view(torch.uint16).numpy().reshape(-1).copy()
    actual = np.zeros((heads, tokens, head_dim), dtype=np.float32)
    status = FULL_PREFILL_KERNEL(
        q_f32.ctypes.data_as(FLOAT_P),
        k_bf16.ctypes.data_as(U16_P),
        v_bf16.ctypes.data_as(U16_P),
        actual.ctypes.data_as(FLOAT_P),
        heads,
        kv_heads,
        tokens,
        0,
        tokens,
        head_dim,
        head_dim,
        CK_ATTN_REDUCTION_BF16_PYTORCH_SDPA,
    )
    if status != CK_ATTENTION_STATUS_OK:
        raise AssertionError(
            f"real-activation provider rejected T={tokens}: status={status}"
        )
    mismatch = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
    if mismatch.size:
        flat = int(mismatch[0])
        head, token, channel = np.unravel_index(flat, actual.shape)
        delta = np.abs(actual - expected)
        raise AssertionError(
            "Qwen3-VL layer-0 full-matrix attention mismatch "
            f"T={tokens} head={head} token={token} channel={channel} "
            f"actual={actual[head, token, channel]:.9g} "
            f"expected={expected[head, token, channel]:.9g} "
            f"max_abs={float(delta.max()):.9g} "
            f"mismatches={mismatch.size}/{actual.size}"
        )
    print(
        "PASS Qwen3-VL layer-0 real activation "
        f"T={tokens}: byte-exact {actual.size}/{actual.size}"
    )


def main() -> None:
    flags = cpu_flags()
    if "avx512f" not in flags or not ({"avx512_bf16", "amx_bf16"} & flags):
        print("SKIP: native AVX-512 BF16 or AMX-BF16 hardware is required")
        return
    if not TORCH_CPU.is_file():
        print("SKIP: PyTorch CPU SLEEF oracle library is unavailable")
        return

    torch.set_num_threads(int(os.environ.get("CK_NUM_THREADS", "24")))
    for kv_tokens, seed in ((4, 44), (17, 1), (511, 2), (512, 3), (1083, 4), (1307, 5)):
        run_case(kv_tokens, seed)
    run_prefill_tail_case(44)
    run_full_prefill_case(4, 44)
    total = 8
    if os.environ.get("CK_BF16_FULL_SHAPES") == "1":
        run_full_prefill_case(1026, 22010)
        total += 1
    checkpoint = os.environ.get("CK_QWEN3VL_CHECKPOINT")
    image_path = os.environ.get("CK_QWEN3VL_IMAGE")
    if checkpoint and image_path:
        run_qwen3vl_activation_case(Path(checkpoint), Path(image_path))
        total += 1
    print(
        "BF16 PyTorch native-GQA decode/prefill parity: "
        f"{total}/{total} byte-exact"
    )


if __name__ == "__main__":
    main()
