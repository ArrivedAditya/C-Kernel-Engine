#!/usr/bin/env python3
"""Compose the bounded runtime-length Kokoro phoneme-to-waveform graph."""

import argparse
import json
from pathlib import Path

from build_kokoro_generator_stage1_pool_circuit import add_stage1_pool
from build_kokoro_runtime_stage1_source_join_circuit import (
    build_circuit as runtime_join_circuit,
)
from build_kokoro_waveform_circuit import add_waveform


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'version/v8/circuits/kokoro_runtime_waveform_bounded.json'


def build_circuit():
    graph = add_waveform(add_stage1_pool(runtime_join_circuit()))
    graph['name'] = 'kokoro_runtime_waveform_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_runtime_waveform_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_runtime_waveform=True,
        captured_gaussian_input=True,
        native_rng=False,
        connected_numerical_parity=False,
        generated_checked_tokens=[1, 64],
        native_host_voice_rows_require_tokens=[3, 40],
    )
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('runtime waveform circuit differs from authoring')
    else:
        args.output.write_text(content)


if __name__ == '__main__':
    main()
