#!/usr/bin/env python3
"""Regression checks for rebased stitched-parity dump alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import decoder_first_token_parity_v8 as parity  # noqa: E402


def main() -> int:
    dump = parity.parity_test_v7.ParityDump
    llama = [
        dump(0, "kqv_wo", np.array([float(token), 2.0], dtype=np.float32), token, "fp32",
             source_token_id=token,
             source_name=f"kqv_wo-0-token-{token:06d}-occ-000")
        for token in range(41, 86)
    ]
    ck_rows = np.array([[float(token), 2.0] for token in range(41, 86)], dtype=np.float32)
    ck = [
        dump(0, "attn_out", ck_rows, 0, "fp32"),
    ]

    llama = parity._trim_llama_prefill_decode_dumps(
        llama, prompt_start_token=41, prompt_token_count=45,
    )
    ck = parity._expand_ck_prefill_decode_dumps(
        ck, llama, prompt_start_token=1013, prompt_token_count=45,
    )
    report = parity._compare_dump_sets(
        ck, llama, atol=0.0, rtol=0.0, pass_filter="decode",
    )
    row = report["results"][0]
    assert row["op"] == "out_proj", row
    assert row["status"] == "PASS", row
    assert row["ck_token"] == 44 and row["llama_token"] == 44, row
    assert row["ck_source_token"] == 1057, row
    assert row["llama_source_token"] == 85, row
    assert row["llama_source_name"] == "kqv_wo-0-token-000085-occ-000", row

    filters = parity._ck_dump_filter_names("kqv_wo-0")
    assert filters == "kqv_wo-0,attn_out-0", filters
    print("v8 dump alignment: PASS (out_proj alias + absolute token provenance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
