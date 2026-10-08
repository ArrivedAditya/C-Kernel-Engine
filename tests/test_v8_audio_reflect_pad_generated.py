"""Normal compiler proof: runtime extent -> left reflection -> consumer."""

import ctypes
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


class ReflectPadGeneratedTest(unittest.TestCase):
    def test_generated_reflection_and_failure_propagation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ops = [
                {'id': 'source', 'op': 'runtime_extent_sum',
                 'kernel': 'runtime_extent_sum_i32', 'returns_status': True,
                 'produces_runtime_lengths': {'source_frames': 'valid_extent'},
                 'params': {'T': 1},
                 'graph_slots': {'inputs': {'values': 'external:durations'},
                     'outputs': {'valid_extent': 'source_extent'}}},
                {'id': 'affine', 'op': 'runtime_extent_affine',
                 'kernel': 'runtime_extent_affine_i32', 'returns_status': True,
                 'consumes_runtime_lengths': ['source_frames'],
                 'produces_runtime_lengths': {'output_frames': 'valid_extent'},
                 'runtime_scalar_bindings': {'extent_affine_source': 'source_frames'},
                 'params': {'call_constants': {'extent_affine_factor': 1,
                     'extent_affine_offset': 1, 'extent_affine_capacity': 6}},
                 'graph_slots': {'inputs': {},
                     'outputs': {'valid_extent': 'output_extent'}}},
                {'id': 'pad', 'op': 'audio_reflect_pad1d_left_checked',
                 'kernel': 'audio_reflect_pad1d_left_channel_major_f32_checked',
                 'returns_status': True,
                 'consumes_runtime_lengths': ['source_frames', 'output_frames'],
                 'runtime_scalar_bindings': {
                     'reflect_pad_input_frames': 'source_frames',
                     'reflect_pad_output_frames': 'output_frames'},
                 'params': {'C': 2, 'T': 5, 'U': 6, 'call_constants': {
                     'reflect_pad_input_elements': 16,
                     'reflect_pad_input_stride': 8,
                     'reflect_pad_output_elements': 18,
                     'reflect_pad_output_stride': 9,
                     'reflect_pad_channels': 2,
                     'reflect_pad_left_padding': 1}},
                 'graph_slots': {'inputs': {'input': 'external:features'},
                     'outputs': {'output': 'padded'}}},
                {'id': 'consume', 'op': 'runtime_copy_valid',
                 'kernel': 'runtime_copy_valid_f32', 'returns_status': True,
                 'consumes_runtime_lengths': ['output_frames'],
                 'runtime_scalar_bindings': {'expanded_frames': 'output_frames'},
                 'params': {'C': 2, 'A_capacity': 9, 'call_constants': {
                     'expanded_elements': 18, 'channels': 2,
                     'expanded_stride': 9, 'output_elements': 18,
                     'output_stride': 9}},
                 'graph_slots': {'inputs': {'input': 'padded'},
                     'outputs': {'output': 'consumed'}}},
            ]
            buffers = {'durations': {'shape': [1]},
                'source_extent': {'shape': [1]},
                'output_extent': {'shape': [1]},
                'features': {'shape': [2, 8]},
                'padded': {'shape': [2, 9]},
                'consumed': {'shape': [2, 9]}}
            circuit = {'version': 3, 'name': 'synthetic_reflect_pad',
                'family': 'bounded_graph', 'checked_native_entry': True,
                'contract': {'runtime_invariants': {'inference_only': True}},
                'activation_buffers': buffers,
                'activation_bindings': {key: key for key in buffers},
                'runtime_constants': {'value_elements': 1, 'phoneme_count': 1,
                    'max_duration': 5, 'expanded_capacity': 5},
                'runtime_lengths': {
                    'source_frames': {'producer': 'source',
                        'result': 'valid_extent', 'capacity': 5,
                        'allow_zero': False},
                    'output_frames': {'producer': 'affine',
                        'result': 'valid_extent', 'capacity': 6,
                        'allow_zero': False}},
                'native_entry': {'function': 'ck_synthetic_reflect_pad',
                    'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                        {'c_type': 'size_t', 'name': 'arena_bytes'},
                        {'c_type': 'int32_t *', 'name': 'out_frames'}],
                    'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
                    'runtime_length_outputs': {'output_frames': 'out_frames'}},
                'sequence': ['component'], 'block_types': {'component': {
                    'sequence': ['header', 'body', 'footer'],
                    'header': ops[:2],
                    'body': {'type': 'dense', 'ops': [ops[2]]},
                    'footer': [ops[3]]}}}
            path = root / 'circuit.json'
            path.write_text(json.dumps(circuit))
            source = {'config': {'model': circuit['name'], 'arch': circuit['name'],
                'num_layers': 1, 'embed_dim': 2, 'num_heads': 1,
                'num_kv_heads': 1, 'head_dim': 2, 'intermediate_size': 2,
                'context_length': 9, 'max_seq_len': 9, 'vocab_size': 1,
                'activation_buffer_dtypes': {'durations': 'i32',
                    'source_extent': 'i32', 'output_extent': 'i32'}},
                'entries': [], 'quant_summary': {}, 'template': circuit}
            layout, calls, _, _, fn = compile_native_graph(root, source, path)
            self.assertEqual(calls['errors'], [])
            self.assertEqual([op['function'] for op in calls['operations'][-2:]],
                             ['audio_reflect_pad1d_left_channel_major_f32_checked',
                              'ck_runtime_copy_valid_f32'])
            fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32)]
            size = layout['memory']['arena']['total_size']
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(raw,
                (-ctypes.addressof(raw)) & 63)
            locations = {item['name']: item for item in
                layout['memory']['activations']['buffers']}
            def view(name, dtype, shape):
                return np.ndarray(shape, dtype, buffer=arena,
                    offset=locations[name]['abs_offset'])
            duration = view('durations', np.int32, (1,))
            features = view('features', np.float32, (2, 8))
            padded = view('padded', np.float32, (2, 9))
            consumed = view('consumed', np.float32, (2, 9))
            for valid in (2, 5, 3, 1, 0, 6, 2):
                duration[0] = valid
                features[:] = np.nan
                if valid > 0:
                    features[:, :min(valid, 5)] = np.arange(2 * min(valid, 5),
                        dtype=np.float32).reshape(2, min(valid, 5)) / 3
                padded[:] = consumed[:] = -71.
                returned = ctypes.c_int32(-999)
                status = fn(arena, len(arena), ctypes.byref(returned))
                if valid in (0, 1, 6):
                    self.assertNotEqual(status, 0)
                    self.assertEqual(returned.value, -999)
                    self.assertTrue(np.all(padded == -71.))
                    self.assertTrue(np.all(consumed == -71.))
                else:
                    self.assertEqual(status, 0)
                    self.assertEqual(returned.value, valid + 1)
                    expected = np.pad(features[:, :valid], ((0, 0), (1, 0)),
                                      mode='reflect')
                    np.testing.assert_array_equal(padded[:, :valid + 1], expected)
                    np.testing.assert_array_equal(consumed[:, :valid + 1], expected)
                    self.assertTrue(np.all(padded[:, valid + 1:] == -71.))
                    self.assertTrue(np.all(consumed[:, valid + 1:] == -71.))


if __name__ == '__main__':
    unittest.main()
