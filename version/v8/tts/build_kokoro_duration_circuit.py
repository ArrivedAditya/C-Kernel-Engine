#!/usr/bin/env python3
"""Declare fixed 36-token Kokoro encoder → duration predictor graph."""
import argparse
import json
from pathlib import Path

from build_kokoro_encoder_circuit import build_circuit as encoder_circuit

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'version/v8/circuits/kokoro_duration_predictor_bounded.json'
TOKENS = 36
CHANNELS = 512
STYLE = 128
HIDDEN = 256
SCAN_WIDTH = CHANNELS + STYLE
BINS = 50


def concat_op(index, source):
    name = f'duration_concat{index}'
    output = f'scan{index}_input' if index < 3 else 'predictor_features'
    return {'id': name, 'op': 'feature_concat_broadcast_rows',
        'kernel': 'feature_concat_broadcast_rows_f32', 'returns_status': True,
        'params': {'T': TOKENS, 'C': CHANNELS, 'S': STYLE, 'N': SCAN_WIDTH,
                   'call_constants': {
            'concat_input_elements': TOKENS * CHANNELS,
            'concat_input_stride': CHANNELS,
            'concat_feature_elements': STYLE,
            'concat_output_elements': TOKENS * SCAN_WIDTH,
            'concat_output_stride': SCAN_WIDTH,
            'concat_rows': TOKENS, 'concat_input_channels': CHANNELS,
            'concat_feature_channels': STYLE,
            'concat_output_channels': SCAN_WIDTH}},
        'graph_slots': {'inputs': {'input': source,
                                  'feature': 'external:predictor_style'},
                        'outputs': {'output': output}}}


def scan_op(index, source, final=False):
    prefix = ('duration_prosody.head_scan' if final else
              f'duration_prosody.text_encoder.scan{index}')
    output = 'head_scan_output' if final else f'scan{index}_output'
    return {'id': 'duration_head_scan' if final else f'duration_scan{index}',
        'op': 'audio_lstm_bidirectional_scan',
        'kernel': 'audio_lstm_bidirectional_scan_f32', 'returns_status': True,
        'weight_refs': {kind: f'{prefix}.{kind}' for kind in
                        ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')},
        'params': {'T': TOKENS, 'I': SCAN_WIDTH, 'H': HIDDEN,
                   'input_size': SCAN_WIDTH, 'hidden_size': HIDDEN,
                   'call_constants': {
            'scan_input_elements': TOKENS * SCAN_WIDTH,
            'scan_weight_ih_elements': 2 * 4 * HIDDEN * SCAN_WIDTH,
            'scan_weight_hh_elements': 2 * 4 * HIDDEN * HIDDEN,
            'scan_bias_ih_elements': 2 * 4 * HIDDEN,
            'scan_bias_hh_elements': 2 * 4 * HIDDEN,
            'scan_output_elements': TOKENS * CHANNELS,
            'scan_hidden_state_elements': 2 * HIDDEN,
            'scan_cell_state_elements': 2 * HIDDEN,
            'scan_input_stride': SCAN_WIDTH,
            'scan_output_stride': CHANNELS}},
        'graph_slots': {'inputs': {'input': source},
                        'outputs': {'output': output}}}


def norm_op(index):
    prefix = f'duration_prosody.text_encoder.norm{index}'
    return {'id': f'duration_norm{index}', 'op': 'audio_adaptive_layer_norm',
        'kernel': 'audio_adaptive_layer_norm_f32', 'returns_status': True,
        'weight_refs': {kind: f'{prefix}.{kind}' for kind in
                        ('projection_weight', 'projection_bias')},
        'params': {'T': TOKENS, 'C': CHANNELS, 'S': STYLE,
                   'channels': CHANNELS, 'style_dim': STYLE,
                   'normalization_epsilon': 1e-5,
                   'call_constants': {
            'adaln_input_elements': TOKENS * CHANNELS,
            'adaln_style_elements': STYLE,
            'adaln_projection_weight_elements': 2 * CHANNELS * STYLE,
            'adaln_projection_bias_elements': 2 * CHANNELS,
            'adaln_output_elements': TOKENS * CHANNELS,
            'adaln_input_stride': CHANNELS,
            'adaln_output_stride': CHANNELS}},
        'graph_slots': {'inputs': {'input': f'scan{index}_output',
                                  'style': 'external:predictor_style'},
                        'outputs': {'output': f'norm{index}_output'}}}


def build_circuit():
    graph = encoder_circuit(TOKENS)
    graph['name'] = 'kokoro_duration_predictor_bounded'
    graph['native_entry'] = {'function': 'ck_kokoro_duration_predictor',
        'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                   {'c_type': 'size_t', 'name': 'arena_bytes'},
                   {'c_type': 'int32_t *', 'name': 'out_frames'}],
        'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
        'runtime_length_outputs': {'expanded_frames': 'out_frames'}}
    graph['contract']['runtime_invariants'].update(
        generated_duration_predictor=True, complete_waveform=False,
        all_tokens_valid=True, fixed_voice='af_heart',
        max_expanded_frames=128)
    graph['activation_buffers']['predictor_style'] = {'shape': [STYLE]}
    graph['activation_buffers']['duration_logits'] = {'shape': [TOKENS, BINS]}
    graph['activation_buffers']['runtime_values'] = {'shape': [TOKENS]}
    graph['activation_buffers']['runtime_valid_extent'] = {'shape': [1]}
    graph['activation_bindings'].update({name: name for name in
        ('predictor_style', 'duration_logits', 'runtime_values', 'runtime_valid_extent')})
    graph['runtime_lengths'] = {'expanded_frames': {'producer': 'duration_frames',
        'result': 'expanded_frames', 'capacity': 128, 'allow_zero': False}}
    graph['runtime_constants'].update({
        'logit_elements': TOKENS * BINS, 'phoneme_count': TOKENS,
        'logit_stride': BINS, 'duration_capacity': TOKENS,
        'max_duration': BINS, 'expanded_capacity': 128})
    header = [concat_op(0, 'phoneme_features')]
    body = []
    for i in range(3):
        graph['activation_buffers'][f'scan{i}_input'] = {'shape': [TOKENS, SCAN_WIDTH]}
        graph['activation_buffers'][f'scan{i}_output'] = {'shape': [TOKENS, CHANNELS]}
        graph['activation_buffers'][f'norm{i}_output'] = {'shape': [TOKENS, CHANNELS]}
        body.extend([scan_op(i, f'scan{i}_input'), norm_op(i),
                     concat_op(i+1, f'norm{i}_output')])
    graph['activation_buffers']['predictor_features'] = {'shape': [TOKENS, SCAN_WIDTH]}
    graph['activation_buffers']['head_scan_output'] = {'shape': [TOKENS, CHANNELS]}
    for name in graph['activation_buffers']:
        graph['activation_bindings'][name] = name
    projection = {'id': 'duration_projection', 'op': 'linear_rows_checked',
        'kernel': 'linear_rows_checked_f32', 'returns_status': True,
        'weight_refs': {'weight': 'duration_prosody.duration_head.weight',
                        'bias': 'duration_prosody.duration_head.bias'},
        'params': {'M': TOKENS, 'K': CHANNELS, 'N': BINS,
                   'call_constants': {
            'linear_input_elements': TOKENS * CHANNELS,
            'linear_input_stride': CHANNELS,
            'linear_weight_elements': BINS * CHANNELS,
            'linear_weight_stride': CHANNELS,
            'linear_bias_elements': BINS,
            'linear_output_elements': TOKENS * BINS,
            'linear_output_stride': BINS,
            'linear_rows': TOKENS, 'linear_input_channels': CHANNELS,
            'linear_output_channels': BINS}},
        'graph_slots': {'inputs': {'input': 'head_scan_output'},
                        'outputs': {'output': 'duration_logits'}}}
    frames = {'id': 'duration_frames', 'op': 'audio_duration_logits_to_frames',
        'kernel': 'audio_duration_logits_to_frames_f32', 'returns_status': True,
        'produces_runtime_lengths': {'expanded_frames': 'expanded_frames'},
        'params': {'T': TOKENS, 'B': BINS, 'bins': BINS},
        'graph_slots': {'inputs': {'logits': 'duration_logits'},
                        'outputs': {'durations': 'runtime_values',
                                    'expanded_frames': 'runtime_valid_extent'}}}
    footer = [scan_op(3, 'predictor_features', final=True), projection, frames]
    graph['sequence'].append('duration_predictor')
    graph['block_types']['duration_predictor'] = {'sequence': ['header', 'body', 'footer'],
        'header': header, 'body': {'type': 'dense', 'ops': body}, 'footer': footer}
    for op, family, contract in (
        ('feature_concat_broadcast_rows', 'feature_concat_broadcast_rows',
         'feature_concat_broadcast_rows_copy_fp32'),
        ('audio_lstm_bidirectional_scan', 'audio_lstm',
         'audio_lstm_bidirectional_scan_pytorch_ifgo_fp32'),
        ('audio_adaptive_layer_norm', 'audio_adaptive_layer_norm',
         'audio_adaptive_layer_norm_style_linear_fp32')):
        first = next(item for item in header + body + footer if item['op'] == op)
        graph['required_numerical_contracts'][op] = {
            'op': family, 'template_ops': [op], 'phases': {'prefill': {
                'contract_id': contract, 'validation': 'validated',
                'evidence': 'tests/test_v8_kokoro_generated_duration.py'}},
            'checkpoint': {'id': f'kokoro.duration.{first["id"]}',
                'producer': first['id'], 'logical_layout': 'token_major',
                'axis_names': ['token', 'channel']}}
    exports = graph['semantic_checkpoints']['exports']
    for section, ops in (('header', header), ('body', body), ('footer', footer)):
        for op in ops:
            if op['id'] == 'duration_frames':
                continue  # Runtime length and int32 durations have separate checks.
            tensor = op['graph_slots']['outputs']['output']
            exports[op['id']] = {
                'section': section, 'template_op_id': op['id'], 'op': op['op'],
                'checkpoints': [{'id': f'kokoro.duration.{tensor}',
                    'producer': op['id'], 'tensor': tensor,
                    'logical_layout': 'token_major',
                    'axis_names': ['token', 'channel'], 'storage_dtype': 'fp32'}]}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    contents = json.dumps(build_circuit(), indent=2) + '\n'
    if args.check:
        if args.output.read_text() != contents:
            raise SystemExit('duration circuit differs from authoring source')
    else:
        args.output.write_text(contents)


if __name__ == '__main__':
    main()
