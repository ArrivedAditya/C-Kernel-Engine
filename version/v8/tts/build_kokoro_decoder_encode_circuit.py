#!/usr/bin/env python3
"""Declare Kokoro's first style-conditioned acoustic-decoder residual block."""

import argparse
import json
import math
from pathlib import Path

from build_kokoro_decoder_ingress_circuit import build_circuit as ingress_circuit
from build_kokoro_prosody_second_block_circuit import (
    activation, convolution, declared, norm)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'version/v8/circuits/kokoro_decoder_encode_bounded.json'
FRAMES = 128
STYLE = 128
INPUT_CHANNELS = 514
OUTPUT_CHANNELS = 1024
PREFIX = 'waveform_decoder.encode'


def style(name, prefix, channels):
    return declared(name, 'linear_rows_checked', 'linear_rows_checked_f32',
        {'input': 'external:decoder_style'}, name, [2 * channels],
        {'M': 1, 'K': STYLE, 'N': 2 * channels, 'call_constants': {
            'linear_input_elements': STYLE, 'linear_input_stride': STYLE,
            'linear_weight_elements': 2 * channels * STYLE,
            'linear_weight_stride': STYLE,
            'linear_bias_elements': 2 * channels,
            'linear_output_elements': 2 * channels,
            'linear_output_stride': 2 * channels, 'linear_rows': 1,
            'linear_input_channels': STYLE,
            'linear_output_channels': 2 * channels}},
        {'weight': f'{prefix}.fc.weight',
         'bias': f'{prefix}.fc.bias'})


def build_circuit():
    graph = ingress_circuit()
    graph['name'] = 'kokoro_decoder_encode_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_decoder_encode_bounded'
    graph['contract']['runtime_invariants'].update(
        generated_decoder_encode=True, complete_decoder=False,
        complete_waveform=False)
    graph['activation_buffers']['decoder_style'] = {'shape': [STYLE]}
    graph['activation_bindings']['decoder_style'] = 'decoder_style'
    ops = []
    add = ops.append
    add(style('decoder_encode_norm0_style', f'{PREFIX}.norm1',
              INPUT_CHANNELS))
    add(norm('decoder_encode_norm0', 'decoder_joined',
             'decoder_encode_norm0_style', f'{PREFIX}.norm1',
             INPUT_CHANNELS, FRAMES, 'expanded_frames'))
    add(activation('decoder_encode_act0', 'decoder_encode_norm0',
                   INPUT_CHANNELS, FRAMES, 'expanded_frames'))
    add(convolution('decoder_encode_conv0', 'decoder_encode_act0',
        f'{PREFIX}.conv1', INPUT_CHANNELS, OUTPUT_CHANNELS,
        FRAMES, 3, 'expanded_frames'))
    add(style('decoder_encode_norm1_style', f'{PREFIX}.norm2',
              OUTPUT_CHANNELS))
    add(norm('decoder_encode_norm1', 'decoder_encode_conv0',
             'decoder_encode_norm1_style', f'{PREFIX}.norm2',
             OUTPUT_CHANNELS, FRAMES, 'expanded_frames'))
    add(activation('decoder_encode_act1', 'decoder_encode_norm1',
                   OUTPUT_CHANNELS, FRAMES, 'expanded_frames'))
    add(convolution('decoder_encode_conv1', 'decoder_encode_act1',
        f'{PREFIX}.conv2', OUTPUT_CHANNELS, OUTPUT_CHANNELS,
        FRAMES, 3, 'expanded_frames'))
    add(convolution('decoder_encode_shortcut', 'decoder_joined',
        f'{PREFIX}.conv1x1', INPUT_CHANNELS, OUTPUT_CHANNELS,
        FRAMES, 1, 'expanded_frames', zero_bias=True))
    add(declared('decoder_encode_output',
        'audio_scaled_sum_strided_checked',
        'audio_scaled_sum_strided_f32_checked',
        {'left': 'decoder_encode_conv1',
         'right': 'decoder_encode_shortcut'},
        'decoder_encode_output', [OUTPUT_CHANNELS, FRAMES],
        {'R': OUTPUT_CHANNELS, 'C': FRAMES,
         'scale': 1.0 / math.sqrt(2.0), 'call_constants': {
            'sum_left_elements': OUTPUT_CHANNELS * FRAMES,
            'sum_left_stride': FRAMES,
            'sum_right_elements': OUTPUT_CHANNELS * FRAMES,
            'sum_right_stride': FRAMES,
            'sum_output_elements': OUTPUT_CHANNELS * FRAMES,
            'sum_output_stride': FRAMES,
            'sum_rows': OUTPUT_CHANNELS}},
        None, ('expanded_frames',),
        {'sum_columns': 'expanded_frames'}))
    for item, shape in ops:
        name = item['id']
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
        style_tensor = name.endswith('_style')
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'body', 'template_op_id': name,
            'op': item['op'],
            'checkpoints': [{'id': f'kokoro.decoder.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'feature_contiguous' if style_tensor else
                                  'channel_major',
                'axis_names': ['channel'] if style_tensor else
                              ['channel', 'frame'],
                'storage_dtype': 'fp32'}]}
    graph['sequence'].append('decoder_encode')
    graph['block_types']['decoder_encode'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [], 'body': {'type': 'dense',
                              'ops': [item for item, _ in ops]},
        'footer': []}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit('decoder encode circuit differs from authoring')
    else:
        args.output.write_text(content)


if __name__ == '__main__':
    main()
