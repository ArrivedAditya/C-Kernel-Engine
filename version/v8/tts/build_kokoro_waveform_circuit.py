#!/usr/bin/env python3
"""Declare Kokoro's final convolution, spectral transform and inverse STFT."""

import argparse
import json
from pathlib import Path

from build_kokoro_generator_stage1_pool_circuit import (
    ROOT, PADDED_CAPACITY, build_circuit as pool_circuit,
)


OUTPUT = ROOT / 'version/v8/circuits/kokoro_waveform_bounded.json'
POST_CHANNELS = 22
BINS = 11
WAVEFORM_CAPACITY = 76800
ISTFT_SCRATCH = 2 * (WAVEFORM_CAPACITY + 20) + 20


def add_waveform(graph):
    for name, shape in (
        ('generator_conv_post', [POST_CHANNELS, PADDED_CAPACITY]),
        ('generator_magnitude', [PADDED_CAPACITY, BINS]),
        ('generator_phase', [PADDED_CAPACITY, BINS]),
        ('generator_waveform', [WAVEFORM_CAPACITY]),
    ):
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name

    conv = {
        'id': 'generator_conv_post', 'op': 'audio_conv1d_checked',
        'kernel': 'audio_conv1d_checked_channel_major_f32',
        'returns_status': True,
        'consumes_runtime_lengths': ['source_stft_frames'],
        'runtime_scalar_bindings': {
            'conv_input_frames': 'source_stft_frames',
            'conv_output_frames': 'source_stft_frames'},
        'weight_refs': {
            'weight': 'waveform_decoder.generator.conv_post.weight',
            'bias': 'waveform_decoder.generator.conv_post.bias'},
        'params': {
            'I': 128, 'O': POST_CHANNELS,
            'T': PADDED_CAPACITY, 'K': 7, 'U': PADDED_CAPACITY,
            'Q': POST_CHANNELS * PADDED_CAPACITY,
            'call_constants': {
                'conv_input_elements': 128 * PADDED_CAPACITY,
                'conv_input_stride': PADDED_CAPACITY,
                'conv_weight_elements': POST_CHANNELS * 128 * 7,
                'conv_bias_elements': POST_CHANNELS,
                'conv_output_elements': POST_CHANNELS * PADDED_CAPACITY,
                'conv_output_stride': PADDED_CAPACITY,
                'conv_scratch_elements': POST_CHANNELS * PADDED_CAPACITY,
                'conv_input_channels': 128,
                'conv_output_channels': POST_CHANNELS,
                'conv_kernel_size': 7, 'conv_stride': 1,
                'conv_padding': 3}},
        'graph_slots': {'inputs': {
            'input': 'generator_stage1_final_activation'},
            'outputs': {'output': 'generator_conv_post'}}}
    spectral = {
        'id': 'generator_spectral_head',
        'op': 'audio_spectral_head_mag_phase',
        'kernel': 'audio_spectral_head_mag_phase_f32',
        'returns_status': True,
        'consumes_runtime_lengths': ['source_stft_frames'],
        'runtime_scalar_bindings': {
            'spectral_frames': 'source_stft_frames'},
        'params': {
            'B': BINS, 'F': PADDED_CAPACITY, 'bins': BINS,
            'call_constants': {
                'spectral_input_elements': POST_CHANNELS * PADDED_CAPACITY,
                'spectral_input_stride': PADDED_CAPACITY,
                'spectral_magnitude_elements': BINS * PADDED_CAPACITY,
                'spectral_phase_elements': BINS * PADDED_CAPACITY}},
        'graph_slots': {'inputs': {'input': 'generator_conv_post'},
            'outputs': {'magnitude': 'generator_magnitude',
                        'phase': 'generator_phase'}}}
    inverse = {
        'id': 'generator_waveform', 'op': 'audio_istft_mag_phase',
        'kernel': 'audio_istft_mag_phase_f32',
        'returns_status': True,
        'consumes_runtime_lengths': ['source_stft_frames', 'source_samples'],
        'runtime_scalar_bindings': {'frames': 'source_stft_frames'},
        'params': {
            'F': PADDED_CAPACITY, 'N_fft': 20, 'H': 5,
            'Q': ISTFT_SCRATCH,
            'n_fft': 20, 'hop': 5,
            'call_constants': {
                'spectral_elements': BINS * PADDED_CAPACITY,
                'output_capacity': WAVEFORM_CAPACITY,
                'scratch_elements': ISTFT_SCRATCH}},
        'graph_slots': {'inputs': {
            'magnitude': 'generator_magnitude',
            'phase': 'generator_phase'},
            'outputs': {'output': 'generator_waveform'}}}

    for name, ops in (
        ('generator_spectral_projection', [conv, spectral]),
        ('generator_inverse_stft', [inverse]),
    ):
        graph['block_types'][name] = {
            'sequence': ['header', 'body', 'footer'],
            'header': [], 'body': {'type': 'dense', 'ops': ops}, 'footer': []}
        graph['sequence'].append(name)
    for op, tensors in (
        (conv, ('generator_conv_post',)),
        (spectral, ('generator_magnitude', 'generator_phase')),
        (inverse, ('generator_waveform',)),
    ):
        graph['semantic_checkpoints']['exports'][op['id']] = {
            'section': 'body', 'template_op_id': op['id'], 'op': op['op'],
            'checkpoints': [{
                'id': f'kokoro.generator.{tensor}',
                'producer': op['id'], 'tensor': tensor,
                'logical_layout': 'frame_bin' if tensor in (
                    'generator_magnitude', 'generator_phase') else
                    'sample_contiguous' if tensor == 'generator_waveform' else
                    'channel_major',
                'axis_names': ['frame', 'bin'] if tensor in (
                    'generator_magnitude', 'generator_phase') else
                    ['sample'] if tensor == 'generator_waveform' else
                    ['channel', 'frame'],
                'storage_dtype': 'fp32'} for tensor in tensors]}
    return graph


def build_circuit(tokens=36):
    # One start ID, at least one phoneme, and one end ID are required before
    # selecting voice_pack[tokens - 3]. The text encoder reserves 40 columns.
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 3 <= tokens <= 40:
        raise ValueError('waveform tokens must be an integer between 3 and 40')
    graph = pool_circuit(tokens)
    graph['name'] = 'kokoro_waveform_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_waveform_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_spectral_head=True,
        generated_waveform=True,
        native_rng=False,
        source_numerical_compatibility=False)
    return add_waveform(graph)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('waveform circuit differs from authoring')
    else:
        args.output.write_text(content)


if __name__ == '__main__':
    main()
