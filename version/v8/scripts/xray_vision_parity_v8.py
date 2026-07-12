#!/usr/bin/env python3
"""Unified v8 numerical X-ray surface for vision reference backends."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import xray_qwen3vl_bf16_v8 as pytorch_adapter
import xray_qwen3vl_llamacpp_v8 as llamacpp_adapter


BACKENDS = {
    "llamacpp": llamacpp_adapter,
    "pytorch": pytorch_adapter,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Backend options are parsed by the selected adapter. Use "
            "--backend BACKEND --help for backend-specific arguments."
        ),
        add_help=False,
    )
    parser.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    parser.add_argument("-h", "--help", action="store_true")
    return parser


def dispatch(argv: Sequence[str] | None = None) -> int:
    args, remaining = _parser().parse_known_args(list(argv) if argv is not None else None)
    adapter = BACKENDS[args.backend]
    if args.help:
        remaining = ["--help", *remaining]
    return int(adapter.main(remaining))


def main() -> int:
    return dispatch(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
