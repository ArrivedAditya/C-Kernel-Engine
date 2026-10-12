#!/usr/bin/env python3
"""Reproduce checked Conv1D scalar versus hardware-FMA timings.

This measures the same validated kernel implementation and includes output
comparison. It does not claim complete Kokoro synthesis speed.
"""

import argparse
from array import array
import ctypes
import json
import platform
import random
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
F = ctypes.POINTER(ctypes.c_float)
Z = ctypes.c_size_t
SIGNATURE = [F, Z, Z, F, Z, F, Z, F, Z, Z, F, Z] + [Z] * 7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    compiler = subprocess.check_output(['cc', '--version'], text=True).splitlines()[0]
    with tempfile.TemporaryDirectory() as temporary:
        library = Path(temporary) / 'audio_conv_hwfma.so'
        command = ['cc', '-std=c11', '-O2', '-ffp-contract=off', '-fPIC',
                   '-shared', '-I', str(ROOT / 'include'),
                   str(ROOT / 'src/kernels/audio_conv1d_checked.c'),
                   '-lm', '-o', str(library)]
        subprocess.run(command, check=True)
        native = ctypes.CDLL(str(library))
        available = native.audio_conv1d_checked_hardware_fma_available
        available.restype = ctypes.c_int
        if not available():
            raise SystemExit('x86 FMA unavailable; hardware-FMA benchmark skipped')
        avx2_available = native.audio_conv1d_checked_avx2_fma_available
        avx2_available.restype = ctypes.c_int
        functions = {}
        for symbol in ('audio_conv1d_checked_channel_major_f32',
                       'audio_conv1d_checked_channel_major_f32_hwfma',
                       'audio_conv1d_checked_channel_major_f32_avx2_fma'):
            if symbol.endswith('avx2_fma') and not avx2_available():
                continue
            function = getattr(native, symbol)
            function.argtypes = SIGNATURE
            function.restype = ctypes.c_int
            functions[symbol] = function
        rng = random.Random(740)
        cases = []
        for input_channels, output_channels, frames, taps in (
                (512, 512, 36, 5), (128, 22, 12000, 7)):
            source = array('f', (rng.gauss(0, 0.05)
                                 for _ in range(input_channels * frames)))
            weight = array('f', (rng.gauss(0, 0.05) for _ in range(
                output_channels * input_channels * taps)))
            bias = array('f', (rng.gauss(0, 0.05)
                               for _ in range(output_channels)))
            output = array('f', [0.] * (output_channels * frames))
            scratch = array('f', [0.] * len(output))
            def ptr(array):
                return ctypes.cast((ctypes.c_float * len(array)).from_buffer(array), F)
            call_args = (ptr(source), len(source), frames, ptr(weight),
                         len(weight), ptr(bias), len(bias), ptr(output),
                         len(output), frames, ptr(scratch), len(scratch),
                         input_channels, output_channels, frames, taps, 1,
                         taps // 2, frames)
            samples = {symbol: [] for symbol in functions}
            outputs = {}
            for function in functions.values():
                if function(*call_args):
                    raise RuntimeError('warmup call failed')
            for iteration in range(args.repeats):
                order = list(functions.items())
                if iteration % 2:
                    order.reverse()
                for symbol, function in order:
                    start = time.perf_counter_ns()
                    status = function(*call_args)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    if status:
                        raise RuntimeError(f'{symbol} returned {status}')
                    samples[symbol].append(elapsed)
                    outputs[symbol] = output[:]
            scalar_symbol = 'audio_conv1d_checked_channel_major_f32'
            hardware_symbol = 'audio_conv1d_checked_channel_major_f32_hwfma'
            if outputs[scalar_symbol] != outputs[hardware_symbol]:
                raise RuntimeError('scalar/hardware-FMA outputs differ')
            vector_symbol = 'audio_conv1d_checked_channel_major_f32_avx2_fma'
            if vector_symbol in outputs and outputs[scalar_symbol] != outputs[vector_symbol]:
                raise RuntimeError('scalar/AVX2 outputs differ')
            scalar_ms = statistics.median(samples[scalar_symbol])
            hardware_ms = statistics.median(samples[hardware_symbol])
            cases.append({'input_channels': input_channels,
                          'output_channels': output_channels, 'frames': frames,
                          'taps': taps, 'scalar_median_ms': scalar_ms,
                          'hardware_fma_median_ms': hardware_ms,
                          'speedup': scalar_ms / hardware_ms,
                          'equal_elements': len(output),
                          'scalar_samples_ms': samples[scalar_symbol],
                          'hardware_fma_samples_ms': samples[hardware_symbol],
                          'avx2_fma_samples_ms': samples.get(vector_symbol),
                          'avx2_fma_median_ms': statistics.median(samples[vector_symbol])
                              if vector_symbol in samples else None})
        print(json.dumps({'host': platform.node(), 'cpu': platform.processor(),
                          'machine': platform.machine(), 'compiler': compiler,
                          'build_command': command[:-2] + ['-o', '<temporary .so>'],
                          'repeats': args.repeats, 'cases': cases}, indent=2))


if __name__ == '__main__':
    main()
