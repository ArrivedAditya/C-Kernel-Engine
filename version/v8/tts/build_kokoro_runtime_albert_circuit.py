#!/usr/bin/env python3
"""Declare a capacity-36 first ALBERT layer with checked request token lengths."""
import json

from build_kokoro_albert_circuit import build_circuit as fixed_layer


def bind_runtime_tokens(graph, *, name, entry_function):
    """Bind one checked token extent through a declared encoder graph."""
    graph['name'] = name
    graph['required_numerical_contracts']['audio_scaled_residual_add'][
        'phases']['prefill']['contract_id'] = (
            'audio_scaled_residual_add_separate_fp32_runtime_extent')
    capacity = graph['activation_buffers']['word_ids']['shape'][0]
    if capacity < 2:
        raise ValueError('encoder capacity must hold at least two tokens')
    graph['contract']['runtime_invariants'].update(
        supported_tokens=f'1..{capacity}', all_tokens_valid=True,
        complete_waveform=False)
    block = graph['block_types']['phoneme_encoder']
    operations = block['header'] + block['body']['ops'] + block['footer']
    residual_ops = [op for op in operations
                    if op['op'] == 'audio_scaled_residual_add']
    widths = {graph['activation_buffers'][tensor]['shape'][1]
              for op in residual_ops
              for tensor in (*op['graph_slots']['inputs'].values(),
                             *op['graph_slots']['outputs'].values())}
    if not residual_ops or len(widths) != 1:
        raise ValueError('ALBERT residual geometry changed')
    width = widths.pop()
    graph['activation_buffers'].update({
        'request_length': {'shape': [1]},
        'valid_length': {'shape': [1]},
        'valid_residual_elements': {'shape': [1]},
    })
    graph['activation_bindings'].update({
        name: name for name in
        ('request_length', 'valid_length', 'valid_residual_elements')})
    entry = graph['native_entry']
    entry['function'] = entry_function
    entry['params'].append({'c_type': 'int32_t *', 'name': 'out_tokens'})
    entry['params'].append({'c_type': 'int32_t *', 'name': 'out_residual_elements'})
    entry['runtime_length_outputs'] = {
        'valid_tokens': 'out_tokens',
        'valid_residual_elements': 'out_residual_elements'}
    graph['runtime_constants'].update(
        value_elements=1, phoneme_count=1, max_duration=capacity,
        expanded_capacity=capacity)
    graph['runtime_lengths'] = {
        'valid_tokens': {'producer': 'check_tokens', 'result': 'valid_extent',
                         'capacity': capacity, 'allow_zero': False},
        'valid_residual_elements': {
            'producer': 'check_residual_elements', 'result': 'valid_extent',
            'capacity': capacity * width, 'allow_zero': False},
    }
    block['header'].insert(0, {
        'id': 'check_tokens', 'op': 'runtime_extent_sum',
        'kernel': 'runtime_extent_sum_i32', 'returns_status': True,
        'produces_runtime_lengths': {'valid_tokens': 'valid_extent'},
        'graph_slots': {'inputs': {'values': 'external:request_length'},
                        'outputs': {'valid_extent': 'valid_length'}},
    })
    block['header'].append({
        'id': 'check_residual_elements', 'op': 'runtime_extent_scale',
        'kernel': 'runtime_extent_scale_i32', 'returns_status': True,
        'consumes_runtime_lengths': ['valid_tokens'],
        'produces_runtime_lengths': {
            'valid_residual_elements': 'valid_extent'},
        'runtime_scalar_bindings': {'extent_scale_source': 'valid_tokens'},
        'params': {'call_constants': {
            'extent_scale_factor': width,
            'extent_scale_capacity': capacity * width}},
        'graph_slots': {'inputs': {}, 'outputs': {
            'valid_extent': 'valid_residual_elements'}},
    })
    for op in block['header'] + block['body']['ops'] + block['footer']:
        kind = op['op']
        if kind == 'embedding_three_table_layer_norm':
            op['consumes_runtime_lengths'] = ['valid_tokens']
            op['runtime_scalar_bindings'] = {'tokens': 'valid_tokens'}
            channels = graph['activation_buffers']['embedding_output']['shape'][1]
            op['params'] = {'T': capacity, 'C': channels, 'call_constants': {
                'id_elements': capacity,
                'output_elements': capacity * channels,
                'channels': channels, 'output_stride': channels}}
        elif kind == 'linear_rows_checked':
            op['consumes_runtime_lengths'] = ['valid_tokens']
            op['runtime_scalar_bindings'] = {'linear_rows': 'valid_tokens'}
            op['params']['call_constants'].pop('linear_rows')
        elif kind == 'attention_full_token_major_checked':
            op['consumes_runtime_lengths'] = ['valid_tokens']
            op['runtime_scalar_bindings'] = {'attention_tokens': 'valid_tokens'}
            op['params']['call_constants'].pop('attention_tokens')
        elif kind in ('layernorm', 'gelu'):
            key = 'norm_rows' if kind == 'layernorm' else 'gelu_rows'
            op['consumes_runtime_lengths'] = ['valid_tokens']
            op['runtime_scalar_bindings'] = {key: 'valid_tokens'}
            op['params']['call_constants'].pop(key)
        elif kind == 'audio_scaled_residual_add':
            op['kernel'] = 'audio_scaled_residual_add_runtime_f32'
            op['consumes_runtime_lengths'] = ['valid_residual_elements']
            op['runtime_scalar_bindings'] = {
                'residual_elements': 'valid_residual_elements'}
        elif kind not in ('runtime_extent_sum', 'runtime_extent_scale'):
            raise ValueError('undeclared runtime token semantics: ' + kind)
    return graph


def build_circuit():
    return bind_runtime_tokens(
        fixed_layer(36), name='kokoro_runtime_first_albert_layer',
        entry_function='ck_kokoro_runtime_first_albert_layer')


if __name__ == '__main__':
    print(json.dumps(build_circuit(), indent=2))
