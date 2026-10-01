"""Model-neutral scaled-sum fixture through normal lowering and generated C."""
import ctypes
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


class CheckedScaledSumGeneratedTest(unittest.TestCase):
    def test_strides_and_producer_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            operation = {'id': 'join', 'op': 'audio_scaled_sum_strided_checked',
                'kernel': 'audio_scaled_sum_strided_f32_checked',
                'returns_status': True,
                'params': {'R': 2, 'C': 3, 'scale': 0.5,
                    'call_constants': {
                        'sum_left_elements': 10, 'sum_left_stride': 5,
                        'sum_right_elements': 12, 'sum_right_stride': 6,
                        'sum_output_elements': 14, 'sum_output_stride': 7,
                        'sum_rows': 2, 'sum_columns': 3}},
                'graph_slots': {'inputs': {'left': 'external:left',
                    'right': 'external:right'}, 'outputs': {'output': 'joined'}}}
            circuit = {'version': 3, 'name': 'synthetic_checked_scaled_sum',
                'family': 'bounded_graph', 'checked_native_entry': True,
                'contract': {'runtime_invariants': {'inference_only': True,
                    'production_kernel_heap_allocation': False}},
                'activation_buffers': {'left': {'shape': [2, 5]},
                    'right': {'shape': [2, 6]}, 'joined': {'shape': [2, 7]}},
                'activation_bindings': {name: name for name in
                    ('left', 'right', 'joined')},
                'native_entry': {'function': 'ck_synthetic_checked_scaled_sum',
                    'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                               {'c_type': 'size_t', 'name': 'arena_bytes'}],
                    'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'}},
                'sequence': ['component'], 'block_types': {'component': {
                    'sequence': ['header', 'body', 'footer'], 'header': [],
                    'body': {'type': 'dense', 'ops': [operation]}, 'footer': []}}}
            path = root / 'circuit.json'
            path.write_text(json.dumps(circuit))
            source = {'config': {'model': circuit['name'],
                'arch': circuit['name'], 'num_layers': 1, 'embed_dim': 2,
                'num_heads': 1, 'num_kv_heads': 1, 'head_dim': 2,
                'intermediate_size': 2, 'context_length': 3,
                'max_seq_len': 3, 'vocab_size': 1},
                'entries': [], 'quant_summary': {}, 'template': circuit}
            layout, calls, _library, _loaded, function = compile_native_graph(
                root, source, path)
            self.assertEqual(calls['errors'], [])
            self.assertEqual([item['function'] for item in calls['operations']],
                             ['audio_scaled_sum_strided_f32_checked'])
            size = layout['memory']['arena']['total_size']
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            buffers = {item['name']: item for item in
                       layout['memory']['activations']['buffers']}
            def view(name, shape):
                return np.ndarray(shape, np.float32, buffer=arena,
                                  offset=buffers[name]['abs_offset'])
            left, right, result = (view('left', (2, 5)),
                                   view('right', (2, 6)),
                                   view('joined', (2, 7)))
            left[:] = right[:] = np.nan
            left[:, :3] = [[1., 2., 3.], [4., 5., 6.]]
            right[:, :3] = [[7., 8., 9.], [10., 11., 12.]]
            result[:] = -77.
            self.assertEqual(function(arena, len(arena)), 0)
            np.testing.assert_array_equal(result[:, :3],
                (left[:, :3] + right[:, :3]) * np.float32(.5))
            self.assertTrue(np.all(result[:, 3:] == -77.))
            result[:] = -77.
            right[1, 2] = np.inf
            self.assertNotEqual(function(arena, len(arena)), 0)
            self.assertTrue(np.all(result == -77.))


if __name__ == '__main__':
    unittest.main()
