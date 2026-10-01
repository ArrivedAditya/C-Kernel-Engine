#!/usr/bin/env python3
"""Declare Kokoro's checked A→2A F0/noise residual blocks."""
import math

from build_kokoro_prosody_first_block_circuit import build_circuit as first_circuit

STYLE = 128
INPUT_CHANNELS = 512
OUTPUT_CHANNELS = 256
INPUT_CAPACITY = 128
OUTPUT_CAPACITY = 256


def declared(name, op, kernel, inputs, output, shape, params,
             weights=None, lengths=(), bindings=None):
    item = {'id': name, 'op': op, 'kernel': kernel, 'returns_status': True,
            'params': params, 'graph_slots': {'inputs': inputs,
                                               'outputs': {'output': output}}}
    if weights:
        item['weight_refs'] = weights
    if lengths:
        item['consumes_runtime_lengths'] = list(lengths)
        item['runtime_scalar_bindings'] = bindings
    return item, shape


def style(name, prefix, channels):
    return declared(name, 'linear_rows_checked', 'linear_rows_checked_f32',
        {'input': 'external:predictor_style'}, name, [2 * channels],
        {'M': 1, 'K': STYLE, 'N': 2 * channels, 'call_constants': {
            'linear_input_elements': STYLE, 'linear_input_stride': STYLE,
            'linear_weight_elements': 2 * channels * STYLE,
            'linear_weight_stride': STYLE, 'linear_bias_elements': 2 * channels,
            'linear_output_elements': 2 * channels,
            'linear_output_stride': 2 * channels, 'linear_rows': 1,
            'linear_input_channels': STYLE,
            'linear_output_channels': 2 * channels}},
        {'weight': f'{prefix}.fc.weight', 'bias': f'{prefix}.fc.bias'})


def norm(name, source, style_name, prefix, channels, capacity, length):
    elements = channels * capacity
    return declared(name, 'audio_adain_instance_norm',
        'audio_adain_instance_norm_f32',
        {'input': source, 'style_affine': style_name}, name,
        [channels, capacity], {'C': channels, 'T': capacity,
            'normalization_epsilon': 1e-5, 'call_constants': {
                'adain_input_elements': elements, 'adain_input_stride': capacity,
                'adain_norm_weight_elements': channels,
                'adain_norm_bias_elements': channels,
                'adain_style_affine_elements': 2 * channels,
                'adain_output_elements': elements,
                'adain_output_stride': capacity,
                'adain_channels': channels}},
        {'norm_weight': f'{prefix}.norm.weight',
         'norm_bias': f'{prefix}.norm.bias'},
        (length,), {'adain_frames': length})


def activation(name, source, channels, capacity, length):
    return declared(name, 'leaky_relu_strided_checked',
        'leaky_relu_strided_f32_checked', {'input': source}, name,
        [channels, capacity], {'R': channels, 'C': capacity,
            'negative_slope': .2, 'call_constants': {
                'leaky_input_elements': channels * capacity,
                'leaky_input_stride': capacity,
                'leaky_output_elements': channels * capacity,
                'leaky_output_stride': capacity,
                'leaky_rows': channels}}, None, (length,),
        {'leaky_columns': length})


def convolution(name, source, prefix, input_channels, output_channels,
                capacity, kernel_size, length, zero_bias=False):
    return declared(name, 'audio_conv1d_checked',
        'audio_conv1d_checked_channel_major_f32',
        {'input': source}, name, [output_channels, capacity],
        {'I': input_channels, 'O': output_channels, 'T': capacity,
         'K': kernel_size, 'U': capacity, 'Q': output_channels * capacity,
         'call_constants': {
            'conv_input_elements': input_channels * capacity,
            'conv_input_stride': capacity,
            'conv_weight_elements': input_channels * output_channels * kernel_size,
            'conv_bias_elements': output_channels,
            'conv_output_elements': output_channels * capacity,
            'conv_output_stride': capacity,
            'conv_scratch_elements': output_channels * capacity,
            'conv_input_channels': input_channels,
            'conv_output_channels': output_channels,
            'conv_kernel_size': kernel_size,
            'conv_stride': 1, 'conv_padding': kernel_size // 2}},
        {'weight': f'{prefix}.weight',
         'bias': f'{prefix}.bias' if not zero_bias else
                 f'{prefix}.cke_zero_bias'},
        (length,), {'conv_input_frames': length,
                    'conv_output_frames': length})


def branch_ops(branch):
    stem = branch.lower()
    prefix = f'duration_prosody.{branch}.1'
    first = f'{stem}_block0_output'
    ops = []
    add = ops.append
    add(style(f'{stem}_block1_norm0_style', f'{prefix}.norm1', INPUT_CHANNELS))
    add(norm(f'{stem}_block1_norm0_output', first,
             f'{stem}_block1_norm0_style', f'{prefix}.norm1',
             INPUT_CHANNELS, INPUT_CAPACITY, 'expanded_frames'))
    add(activation(f'{stem}_block1_act0', f'{stem}_block1_norm0_output',
                   INPUT_CHANNELS, INPUT_CAPACITY, 'expanded_frames'))
    add(declared(f'{stem}_block1_pool',
        'audio_conv_transpose1d_depthwise_checked',
        'audio_conv_transpose1d_depthwise_channel_major_f32_checked',
        {'input': f'{stem}_block1_act0'}, f'{stem}_block1_pool',
        [INPUT_CHANNELS, OUTPUT_CAPACITY],
        {'C': INPUT_CHANNELS, 'T': INPUT_CAPACITY, 'K': 3,
         'U': OUTPUT_CAPACITY, 'Q': INPUT_CHANNELS * OUTPUT_CAPACITY,
         'call_constants': {
            'deconv_input_elements': INPUT_CHANNELS * INPUT_CAPACITY,
            'deconv_input_stride': INPUT_CAPACITY,
            'deconv_weight_elements': INPUT_CHANNELS * 3,
            'deconv_bias_elements': INPUT_CHANNELS,
            'deconv_output_elements': INPUT_CHANNELS * OUTPUT_CAPACITY,
            'deconv_output_stride': OUTPUT_CAPACITY,
            'deconv_scratch_elements': INPUT_CHANNELS * OUTPUT_CAPACITY,
            'deconv_channels': INPUT_CHANNELS, 'deconv_kernel_size': 3,
            'deconv_stride': 2, 'deconv_padding': 1,
            'deconv_output_padding': 1}},
        {'weight': f'{prefix}.pool.weight',
         'bias': f'{prefix}.pool.bias'},
        ('expanded_frames', 'upsampled_frames'),
        {'deconv_input_frames': 'expanded_frames',
         'deconv_output_frames': 'upsampled_frames'}))
    add(convolution(f'{stem}_block1_conv0', f'{stem}_block1_pool',
        f'{prefix}.conv1', INPUT_CHANNELS, OUTPUT_CHANNELS,
        OUTPUT_CAPACITY, 3, 'upsampled_frames'))
    add(style(f'{stem}_block1_norm1_style', f'{prefix}.norm2', OUTPUT_CHANNELS))
    add(norm(f'{stem}_block1_norm1_output', f'{stem}_block1_conv0',
             f'{stem}_block1_norm1_style', f'{prefix}.norm2',
             OUTPUT_CHANNELS, OUTPUT_CAPACITY, 'upsampled_frames'))
    add(activation(f'{stem}_block1_act1', f'{stem}_block1_norm1_output',
                   OUTPUT_CHANNELS, OUTPUT_CAPACITY, 'upsampled_frames'))
    add(convolution(f'{stem}_block1_conv1', f'{stem}_block1_act1',
        f'{prefix}.conv2', OUTPUT_CHANNELS, OUTPUT_CHANNELS,
        OUTPUT_CAPACITY, 3, 'upsampled_frames'))
    add(declared(f'{stem}_block1_shortcut_upsample',
        'audio_upsample_nearest_checked',
        'audio_upsample_nearest_channel_major_f32_checked',
        {'input': first}, f'{stem}_block1_shortcut_upsample',
        [INPUT_CHANNELS, OUTPUT_CAPACITY],
        {'C': INPUT_CHANNELS, 'T': INPUT_CAPACITY, 'U': OUTPUT_CAPACITY,
         'call_constants': {
            'nearest_input_elements': INPUT_CHANNELS * INPUT_CAPACITY,
            'nearest_input_stride': INPUT_CAPACITY,
            'nearest_output_elements': INPUT_CHANNELS * OUTPUT_CAPACITY,
            'nearest_output_stride': OUTPUT_CAPACITY,
            'nearest_channels': INPUT_CHANNELS, 'nearest_factor': 2}},
        None, ('expanded_frames', 'upsampled_frames'),
        {'nearest_input_frames': 'expanded_frames',
         'nearest_output_frames': 'upsampled_frames'}))
    add(convolution(f'{stem}_block1_shortcut_conv',
        f'{stem}_block1_shortcut_upsample', f'{prefix}.conv1x1',
        INPUT_CHANNELS, OUTPUT_CHANNELS, OUTPUT_CAPACITY, 1,
        'upsampled_frames', zero_bias=True))
    add(declared(f'{stem}_block1_output',
        'audio_scaled_sum_strided_checked',
        'audio_scaled_sum_strided_f32_checked',
        {'left': f'{stem}_block1_conv1',
         'right': f'{stem}_block1_shortcut_conv'},
        f'{stem}_block1_output', [OUTPUT_CHANNELS, OUTPUT_CAPACITY],
        {'R': OUTPUT_CHANNELS, 'C': OUTPUT_CAPACITY,
         'scale': 1.0 / math.sqrt(2.0), 'call_constants': {
            'sum_left_elements': OUTPUT_CHANNELS * OUTPUT_CAPACITY,
            'sum_left_stride': OUTPUT_CAPACITY,
            'sum_right_elements': OUTPUT_CHANNELS * OUTPUT_CAPACITY,
            'sum_right_stride': OUTPUT_CAPACITY,
            'sum_output_elements': OUTPUT_CHANNELS * OUTPUT_CAPACITY,
            'sum_output_stride': OUTPUT_CAPACITY,
            'sum_rows': OUTPUT_CHANNELS}},
        None, ('upsampled_frames',),
        {'sum_columns': 'upsampled_frames'}))
    return ops


def build_circuit():
    graph = first_circuit()
    graph['name'] = 'kokoro_prosody_second_block_bounded'
    graph['native_entry']['function'] = 'ck_kokoro_prosody_second_block_bounded'
    graph['native_entry']['params'].append(
        {'c_type': 'int32_t *', 'name': 'out_upsampled_frames'})
    graph['native_entry']['runtime_length_outputs']['upsampled_frames'] = \
        'out_upsampled_frames'
    graph['contract']['runtime_invariants'].update(
        generated_second_f0_noise_residual_blocks=True,
        complete_prosody=False, complete_waveform=False)
    graph['runtime_lengths']['upsampled_frames'] = {
        'producer': 'prosody_double_frames', 'result': 'valid_extent',
        'capacity': OUTPUT_CAPACITY, 'allow_zero': False}
    graph['activation_buffers']['runtime_upsampled_extent'] = {'shape': [1]}
    graph['activation_bindings']['runtime_upsampled_extent'] = \
        'runtime_upsampled_extent'
    extent = {'id': 'prosody_double_frames', 'op': 'runtime_extent_scale',
        'kernel': 'runtime_extent_scale_i32', 'returns_status': True,
        'consumes_runtime_lengths': ['expanded_frames'],
        'produces_runtime_lengths': {'upsampled_frames': 'valid_extent'},
        'runtime_scalar_bindings': {'extent_scale_source': 'expanded_frames'},
        'params': {'call_constants': {'extent_scale_factor': 2,
                                     'extent_scale_capacity': OUTPUT_CAPACITY}},
        'graph_slots': {'inputs': {},
                        'outputs': {'valid_extent': 'runtime_upsampled_extent'}}}
    ops = [extent]
    for branch in ('F0', 'N'):
        ops.extend(branch_ops(branch))
    for item, shape in ops[1:]:
        name = item['id']
        graph['activation_buffers'][name] = {'shape': shape}
        graph['activation_bindings'][name] = name
        style_tensor = name.endswith('_style')
        graph['semantic_checkpoints']['exports'][name] = {
            'section': 'body', 'template_op_id': name, 'op': item['op'],
            'checkpoints': [{'id': f'kokoro.prosody.{name}',
                'producer': name, 'tensor': name,
                'logical_layout': 'feature_contiguous' if style_tensor else
                                  'channel_major',
                'axis_names': ['channel'] if style_tensor else
                              ['channel', 'frame'],
                'storage_dtype': 'fp32'}]}
    graph['sequence'].append('prosody_second_block')
    graph['block_types']['prosody_second_block'] = {
        'sequence': ['header', 'body', 'footer'],
        'header': [extent],
        'body': {'type': 'dense', 'ops': [item for item, _ in ops[1:]]},
        'footer': []}
    return graph
