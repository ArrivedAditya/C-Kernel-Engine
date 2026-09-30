"""Model-neutral checked broadcast → broadcast through ordinary v8 codegen."""
import copy
import ctypes
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


def operation(name, source, feature, output, rows, channels, feature_channels,
              input_stride, output_stride):
    width = channels + feature_channels
    return {'id': name, 'op': 'feature_concat_broadcast_rows',
        'kernel': 'feature_concat_broadcast_rows_f32', 'returns_status': True,
        'params': {'T': rows, 'C': channels, 'S': feature_channels, 'N': width,
                   'call_constants': {
            'concat_input_elements': rows * input_stride,
            'concat_input_stride': input_stride,
            'concat_feature_elements': feature_channels,
            'concat_output_elements': rows * output_stride,
            'concat_output_stride': output_stride,
            'concat_rows': rows, 'concat_input_channels': channels,
            'concat_feature_channels': feature_channels,
            'concat_output_channels': width}},
        'graph_slots': {'inputs': {'input': source, 'feature': feature},
                        'outputs': {'output': output}}}


class BroadcastGeneratedTest(unittest.TestCase):
    def compile(self, mutate=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        first = operation('first', 'external:input', 'external:style_a',
                          'middle', 2, 5, 3, 7, 10)
        second = operation('second', 'middle', 'external:style_b',
                           'output', 2, 8, 2, 10, 12)
        template = {'version': 3, 'name': 'broadcast_twice', 'family': 'bounded_graph',
            'checked_native_entry': True,
            'contract': {'runtime_invariants': {'inference_only': True,
                                                'production_kernel_heap_allocation': False}},
            'activation_buffers': {name: {'shape': shape} for name, shape in (
                ('input', [2, 7]), ('style_a', [3]), ('style_b', [2]),
                ('middle', [2, 10]), ('output', [2, 12]))},
            'activation_bindings': {name: name for name in
                ('input', 'style_a', 'style_b', 'middle', 'output')},
            'native_entry': {'function': 'ck_broadcast_twice', 'params': [
                {'c_type': 'uint8_t *', 'name': 'arena'},
                {'c_type': 'size_t', 'name': 'arena_bytes'}],
                'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'}},
            'sequence': ['component'], 'block_types': {'component': {
                'sequence': ['header', 'body', 'footer'], 'header': [first],
                'body': {'type': 'dense', 'ops': [second]}, 'footer': []}}}
        if mutate:
            mutate(template)
        circuit = root / 'circuit.json'
        circuit.write_text(json.dumps(template))
        source = {'config': {'model': template['name'], 'arch': template['name'],
            'num_layers': 1, 'embed_dim': 8, 'num_heads': 1,
            'num_kv_heads': 1, 'head_dim': 8, 'intermediate_size': 8,
            'context_length': 2, 'max_seq_len': 2, 'vocab_size': 1},
            'entries': [], 'quant_summary': {}, 'template': template}
        layout, call_ir, library, loaded, fn = compile_native_graph(root, source, circuit)
        self.assertEqual(call_ir['errors'], [])
        size = layout['memory']['arena']['total_size']
        raw = (ctypes.c_uint8 * (size + 63))()
        arena = (ctypes.c_uint8 * size).from_buffer(raw, (-ctypes.addressof(raw)) & 63)
        buffers = {item['name']: item for item in layout['memory']['activations']['buffers']}
        def tensor(name, shape):
            return np.ndarray(shape, np.float32, buffer=arena,
                              offset=buffers[name]['abs_offset'])
        return arena, tensor, fn, loaded, call_ir, root

    def test_two_connected_calls_strides_repeat_and_failure(self):
        arena, tensor, fn, loaded, calls, root = self.compile()
        self.assertEqual([op['function'] for op in calls['operations']],
                         ['feature_concat_broadcast_rows_f32'] * 2)
        self.assertTrue((root / 'generated.c').exists())
        source = tensor('input', (2, 7)); a = tensor('style_a', (3,))
        b = tensor('style_b', (2,)); middle = tensor('middle', (2, 10))
        output = tensor('output', (2, 12))
        for offset in (0, 3):
            source[:] = np.nan
            source[:, :5] = np.arange(10, dtype=np.float32).reshape(2, 5) + offset
            a[:] = [0.25, -2, 4]; b[:] = [7, -3]
            middle[:] = -777; output[:] = -888
            self.assertEqual(fn(arena, len(arena)), 0)
            expected_middle = np.concatenate((source[:, :5],
                np.broadcast_to(a, (2, 3))), axis=1)
            expected = np.concatenate((expected_middle,
                np.broadcast_to(b, (2, 2))), axis=1)
            np.testing.assert_array_equal(middle[:, :8], expected_middle)
            np.testing.assert_array_equal(output[:, :10], expected)
            self.assertTrue((middle[:, 8:] == -777).all())
            self.assertTrue((output[:, 10:] == -888).all())
        middle[:] = -777; output[:] = -888; a[0] = np.nan
        self.assertEqual(fn(arena, len(arena)), -3)
        self.assertTrue((middle == -777).all())
        self.assertTrue((output == -888).all())
        self.assertEqual(fn(arena, len(arena) - 1), -2)
        self.assertTrue((output == -888).all())

    def test_contradictory_sizes_rejected_by_planner_or_provider(self):
        def oversized(template):
            op = template['block_types']['component']['body']['ops'][0]
            op['params']['call_constants']['concat_input_elements'] = 10_000
        with self.assertRaisesRegex(RuntimeError, 'HARD CALL CONSTANT FAULT'):
            self.compile(oversized)
        def false_width(template):
            op = template['block_types']['component']['body']['ops'][0]
            op['params']['N'] = 9
            op['params']['call_constants']['concat_output_channels'] = 9
        arena, tensor, fn, loaded, calls, root = self.compile(false_width)
        tensor('input', (2, 7))[:] = 1
        tensor('style_a', (3,))[:] = 2
        tensor('style_b', (2,))[:] = 3
        output = tensor('output', (2, 12)); output[:] = -888
        self.assertEqual(fn(arena, len(arena)), -2)
        self.assertTrue((output == -888).all())


if __name__ == '__main__':
    unittest.main()
