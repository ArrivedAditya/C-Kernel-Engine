"""Normal compilation of a model-neutral affine extent producer and consumer."""

import ctypes
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


class RuntimeExtentAffineGeneratedTest(unittest.TestCase):
    def test_checked_affine_length_controls_consumer(self):
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
                     'outputs': {'valid_extent': 'affine_extent'}}},
                {'id': 'copy', 'op': 'runtime_copy_valid',
                 'kernel': 'runtime_copy_valid_f32', 'returns_status': True,
                 'consumes_runtime_lengths': ['output_frames'],
                 'runtime_scalar_bindings': {'expanded_frames': 'output_frames'},
                 'params': {'C': 2, 'A_capacity': 7, 'call_constants': {
                     'expanded_elements': 14, 'channels': 2,
                     'expanded_stride': 7, 'output_elements': 14,
                     'output_stride': 7}},
                 'graph_slots': {'inputs': {'input': 'external:features'},
                     'outputs': {'output': 'copied'}}},
            ]
            buffers = {'durations': {'shape': [1]},
                'source_extent': {'shape': [1]},
                'affine_extent': {'shape': [1]},
                'features': {'shape': [2, 7]},
                'copied': {'shape': [2, 7]}}
            circuit = {'version': 3, 'name': 'synthetic_affine_extent',
                'family': 'bounded_graph', 'checked_native_entry': True,
                'contract': {'runtime_invariants': {'inference_only': True}},
                'activation_buffers': buffers,
                'activation_bindings': {key: key for key in buffers},
                'runtime_constants': {'value_elements': 1, 'phoneme_count': 1,
                    'max_duration': 6, 'expanded_capacity': 6,
                    'expanded_elements': 14, 'channels': 2,
                    'expanded_stride': 7, 'output_elements': 14,
                    'output_stride': 7},
                'runtime_lengths': {
                    'source_frames': {'producer': 'source',
                        'result': 'valid_extent', 'capacity': 6,
                        'allow_zero': True},
                    'output_frames': {'producer': 'affine',
                        'result': 'valid_extent', 'capacity': 6,
                        'allow_zero': False}},
                'native_entry': {'function': 'ck_synthetic_affine_extent',
                    'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                        {'c_type': 'size_t', 'name': 'arena_bytes'},
                        {'c_type': 'int32_t *', 'name': 'out_frames'}],
                    'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
                    'runtime_length_outputs': {'output_frames': 'out_frames'}},
                'sequence': ['component'], 'block_types': {'component': {
                    'sequence': ['header', 'body', 'footer'],
                    'header': [ops[0]],
                    'body': {'type': 'dense', 'ops': [ops[1]]},
                    'footer': [ops[2]]}}}
            path = root / 'circuit.json'
            path.write_text(json.dumps(circuit))
            source = {'config': {'model': circuit['name'], 'arch': circuit['name'],
                'num_layers': 1, 'embed_dim': 2, 'num_heads': 1,
                'num_kv_heads': 1, 'head_dim': 2, 'intermediate_size': 2,
                'context_length': 7, 'max_seq_len': 7, 'vocab_size': 1,
                'activation_buffer_dtypes': {'durations': 'i32',
                    'source_extent': 'i32', 'affine_extent': 'i32'}},
                'entries': [], 'quant_summary': {},
                'template': circuit}
            layout, calls, _, _, fn = compile_native_graph(root, source, path)
            self.assertEqual(calls['errors'], [])
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
            features = view('features', np.float32, (2, 7))
            copied = view('copied', np.float32, (2, 7))
            features[:] = np.arange(14, dtype=np.float32).reshape(2, 7)
            for value in (0, 5, 2, 6, 3):
                duration[0] = value
                copied[:] = -91.
                returned = ctypes.c_int32(-999)
                status = fn(arena, len(arena), ctypes.byref(returned))
                if value == 6:
                    self.assertNotEqual(status, 0)
                    self.assertEqual(returned.value, -999)
                    self.assertTrue(np.all(copied == -91.))
                else:
                    self.assertEqual(status, 0)
                    self.assertEqual(returned.value, value + 1)
                    np.testing.assert_array_equal(copied[:, :value + 1],
                        features[:, :value + 1])
                    self.assertTrue(np.all(copied[:, value + 1:] == -91.))
            # A map scalar may describe only planner-owned physical storage.
            bad = json.loads(json.dumps(circuit))
            bad['block_types']['component']['footer'][0]['params'][
                'call_constants']['output_elements'] = 15
            bad_path = root / 'bad_circuit.json'
            bad_path.write_text(json.dumps(bad))
            bad_source = dict(source, template=bad)
            with self.assertRaisesRegex(RuntimeError, 'HARD CALL CONSTANT FAULT'):
                compile_native_graph(root, bad_source, bad_path)


if __name__ == '__main__':
    unittest.main()
