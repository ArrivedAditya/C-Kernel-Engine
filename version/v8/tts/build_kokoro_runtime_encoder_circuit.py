#!/usr/bin/env python3
"""Declare one capacity-64 Kokoro phoneme encoder with checked token lengths."""
import json

from build_kokoro_encoder_circuit import build_circuit as fixed_encoder
from build_kokoro_runtime_albert_circuit import bind_runtime_tokens


def build_circuit():
    return bind_runtime_tokens(
        fixed_encoder(64), name='kokoro_runtime_phoneme_encoder',
        entry_function='ck_kokoro_runtime_phoneme_encoder')


if __name__ == '__main__':
    print(json.dumps(build_circuit(), indent=2))
