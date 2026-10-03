"""Model-neutral runtime length → Snake through normal generated execution."""

import ctypes
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


class AudioSnakeGeneratedTest(unittest.TestCase):
    def test_runtime_length_weight_binding_failure_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alpha = np.array([1., .25, -2.], dtype=np.float32)
            payload = alpha.tobytes()
            entry = {'name': 'stage.alpha', 'dtype': 'fp32', 'shape': [3],
                     'file_offset': 0, 'size': len(payload),
                     'sha256': hashlib.sha256(payload).hexdigest()}
            ops = [
                {'id': 'extent', 'op': 'runtime_extent_sum',
                 'kernel': 'runtime_extent_sum_i32', 'returns_status': True,
                 'produces_runtime_lengths': {'frames': 'valid_extent'},
                 'params': {'T': 1},
                 'graph_slots': {'inputs': {'values': 'external:durations'},
                                 'outputs': {'valid_extent': 'frame_extent'}}},
                {'id': 'snake', 'op': 'audio_snake_strided_checked',
                 'kernel': 'audio_snake_strided_f32_checked',
                 'returns_status': True,
                 'consumes_runtime_lengths': ['frames'],
                 'runtime_scalar_bindings': {'snake_frames': 'frames'},
                 'weight_refs': {'alpha': 'stage.alpha'},
                 'params': {'C': 3, 'T': 7,
                            'call_constants': {
                                'snake_input_elements': 27,
                                'snake_input_stride': 9,
                                'snake_alpha_elements': 3,
                                'snake_output_elements': 30,
                                'snake_output_stride': 10,
                                'snake_channels': 3}},
                 'graph_slots': {'inputs': {'input': 'external:features'},
                                 'outputs': {'output': 'output'}}},
            ]
            buffers = {'durations': {'shape': [1]},
                       'frame_extent': {'shape': [1]},
                       'features': {'shape': [3, 9]},
                       'output': {'shape': [3, 10]}}
            circuit = {
                'version': 3, 'name': 'synthetic_channelwise_snake',
                'family': 'bounded_graph', 'checked_native_entry': True,
                'contract': {'runtime_invariants': {'inference_only': True}},
                'activation_buffers': buffers,
                'activation_bindings': {key: key for key in buffers},
                'runtime_constants': {'value_elements': 1, 'phoneme_count': 1,
                                      'max_duration': 7, 'expanded_capacity': 7},
                'runtime_lengths': {'frames': {
                    'producer': 'extent', 'result': 'valid_extent',
                    'capacity': 7, 'allow_zero': False}},
                'native_entry': {
                    'function': 'ck_synthetic_channelwise_snake',
                    'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                               {'c_type': 'size_t', 'name': 'arena_bytes'},
                               {'c_type': 'int32_t *', 'name': 'out_frames'}],
                    'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
                    'runtime_length_outputs': {'frames': 'out_frames'}},
                'sequence': ['component'], 'block_types': {'component': {
                    'sequence': ['header', 'body', 'footer'],
                    'header': ops[:1],
                    'body': {'type': 'dense', 'ops': ops[1:]},
                    'footer': []}}}
            path = root / 'circuit.json'
            path.write_text(json.dumps(circuit))
            source = {'config': {
                    'model': circuit['name'], 'arch': circuit['name'],
                    'num_layers': 1, 'embed_dim': 3, 'num_heads': 1,
                    'num_kv_heads': 1, 'head_dim': 3,
                    'intermediate_size': 3, 'context_length': 7,
                    'max_seq_len': 7, 'vocab_size': 1,
                    'activation_buffer_dtypes': {
                        'durations': 'i32', 'frame_extent': 'i32'}},
                'entries': [entry], 'quant_summary': {}, 'template': circuit}
            layout, calls, _library, _loaded, function = compile_native_graph(
                root, source, path)
            self.assertEqual(calls['errors'], [])
            self.assertEqual(calls['operations'][-1]['function'],
                             'audio_snake_strided_f32_checked')
            function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_int32)]
            size = layout['memory']['arena']['total_size']
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            weight = next(item for item in layout['memory']['weights']['entries']
                          if item['name'] == 'stage.alpha')
            arena[weight['abs_offset']:weight['abs_offset'] + len(payload)] = payload
            locations = {item['name']: item for item in
                         layout['memory']['activations']['buffers']}

            def view(name, dtype, shape):
                return np.ndarray(shape, dtype, buffer=arena,
                                  offset=locations[name]['abs_offset'])

            durations = view('durations', np.int32, (1,))
            features = view('features', np.float32, (3, 9))
            output = view('output', np.float32, (3, 10))
            first = None
            for valid in (7, 5, 7):
                durations[0] = valid
                features[:] = np.nan
                features[:, :valid] = np.arange(3 * valid,
                    dtype=np.float32).reshape(3, valid) / 5 - 2
                output[:] = -91.
                published = ctypes.c_int32(-999)
                self.assertEqual(function(arena, len(arena),
                                          ctypes.byref(published)), 0)
                self.assertEqual(published.value, valid)
                x = features[:, :valid]
                a = alpha[:, None]
                expected = x + (1 / a) * np.sin(a * x) ** 2
                np.testing.assert_allclose(output[:, :valid], expected,
                                           rtol=0, atol=2e-6)
                self.assertTrue(np.all(output[:, valid:] == -91.))
                if valid == 7:
                    if first is None: first = output.copy()
                    else: np.testing.assert_array_equal(output, first)
            for failure in ('duration', 'alpha'):
                durations[0] = 8 if failure == 'duration' else 7
                bound_alpha = np.ndarray((3,), np.float32, buffer=arena,
                                         offset=weight['abs_offset'])
                if failure == 'alpha': bound_alpha[-1] = 0
                output[:] = -91.
                published = ctypes.c_int32(-999)
                self.assertNotEqual(function(arena, len(arena),
                                             ctypes.byref(published)), 0)
                self.assertEqual(published.value, -999)
                self.assertTrue(np.all(output == -91.))
                bound_alpha[-1] = alpha[-1]
            durations[0] = 7
            self.assertEqual(function(arena, len(arena),
                                      ctypes.byref(published)), 0)


if __name__ == '__main__':
    unittest.main()
