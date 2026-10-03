"""Model-neutral checked extent -> activation -> dense transposed convolution."""

import ctypes
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


class DenseDeconvGeneratedTest(unittest.TestCase):
    def test_generated_provider_and_runtime_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = {
                'stage.weight': (np.arange(18, dtype=np.float32).reshape(2, 3, 3) - 8) / 16,
                'stage.bias': np.array([.1, -.2, .3], dtype=np.float32),
            }
            ops = [
                {'id': 'sum', 'op': 'runtime_extent_sum',
                 'kernel': 'runtime_extent_sum_i32', 'returns_status': True,
                 'produces_runtime_lengths': {'input_frames': 'valid_extent'},
                 'params': {'T': 1},
                 'graph_slots': {'inputs': {'values': 'external:durations'},
                                 'outputs': {'valid_extent': 'input_extent'}}},
                {'id': 'scale', 'op': 'runtime_extent_scale',
                 'kernel': 'runtime_extent_scale_i32', 'returns_status': True,
                 'consumes_runtime_lengths': ['input_frames'],
                 'produces_runtime_lengths': {'output_frames': 'valid_extent'},
                 'runtime_scalar_bindings': {'extent_scale_source': 'input_frames'},
                 'params': {'call_constants': {'extent_scale_factor': 2,
                                              'extent_scale_capacity': 10}},
                 'graph_slots': {'inputs': {},
                                 'outputs': {'valid_extent': 'output_extent'}}},
                {'id': 'activate', 'op': 'leaky_relu_strided_checked',
                 'kernel': 'leaky_relu_strided_f32_checked', 'returns_status': True,
                 'consumes_runtime_lengths': ['input_frames'],
                 'runtime_scalar_bindings': {'leaky_columns': 'input_frames'},
                 'params': {'R': 2, 'C': 5, 'negative_slope': .1,
                     'call_constants': {'leaky_input_elements': 14,
                         'leaky_input_stride': 7, 'leaky_output_elements': 14,
                         'leaky_output_stride': 7, 'leaky_rows': 2}},
                 'graph_slots': {'inputs': {'input': 'external:features'},
                                 'outputs': {'output': 'activated'}}},
                {'id': 'deconvolve', 'op': 'audio_conv_transpose1d_dense_checked',
                 'kernel': 'audio_conv_transpose1d_dense_channel_major_f32_checked',
                 'returns_status': True,
                 'consumes_runtime_lengths': ['input_frames', 'output_frames'],
                 'runtime_scalar_bindings': {
                     'dense_deconv_input_frames': 'input_frames',
                     'dense_deconv_output_frames': 'output_frames'},
                 'weight_refs': {'weight': 'stage.weight', 'bias': 'stage.bias'},
                 'params': {'I': 2, 'O': 3, 'T': 5, 'K': 3, 'U': 10, 'Q': 30,
                     'call_constants': {'dense_deconv_input_elements': 14,
                         'dense_deconv_input_stride': 7,
                         'dense_deconv_weight_elements': 18,
                         'dense_deconv_bias_elements': 3,
                         'dense_deconv_output_elements': 39,
                         'dense_deconv_output_stride': 13,
                         'dense_deconv_scratch_elements': 30,
                         'dense_deconv_input_channels': 2,
                         'dense_deconv_output_channels': 3,
                         'dense_deconv_kernel_size': 3,
                         'dense_deconv_stride': 2,
                         'dense_deconv_padding': 1,
                         'dense_deconv_output_padding': 1}},
                 'graph_slots': {'inputs': {'input': 'activated'},
                                 'outputs': {'output': 'output'}}},
            ]
            buffers = {'durations': {'shape': [1]},
                       'input_extent': {'shape': [1]},
                       'output_extent': {'shape': [1]},
                       'features': {'shape': [2, 7]},
                       'activated': {'shape': [2, 7]},
                       'output': {'shape': [3, 13]}}
            circuit = {'version': 3, 'name': 'synthetic_dense_deconvolution',
                'family': 'bounded_graph', 'checked_native_entry': True,
                'contract': {'runtime_invariants': {'inference_only': True}},
                'activation_buffers': buffers,
                'activation_bindings': {key: key for key in buffers},
                'runtime_constants': {'value_elements': 1, 'phoneme_count': 1,
                    'max_duration': 5, 'expanded_capacity': 5},
                'runtime_lengths': {
                    'input_frames': {'producer': 'sum', 'result': 'valid_extent',
                                     'capacity': 5, 'allow_zero': False},
                    'output_frames': {'producer': 'scale', 'result': 'valid_extent',
                                      'capacity': 10, 'allow_zero': False}},
                'native_entry': {'function': 'ck_synthetic_dense_deconvolution',
                    'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                               {'c_type': 'size_t', 'name': 'arena_bytes'},
                               {'c_type': 'int32_t *', 'name': 'out_frames'}],
                    'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
                    'runtime_length_outputs': {'output_frames': 'out_frames'}},
                'sequence': ['component'], 'block_types': {'component': {
                    'sequence': ['header', 'body', 'footer'],
                    'header': ops[:2], 'body': {'type': 'dense', 'ops': ops[2:3]},
                    'footer': ops[3:]}}}
            path = root / 'circuit.json'
            path.write_text(json.dumps(circuit))
            payload = b''
            entries = []
            for name, array in weights.items():
                data = array.tobytes()
                entries.append({'name': name, 'dtype': 'fp32',
                    'shape': list(array.shape), 'file_offset': len(payload),
                    'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
                payload += data
            source = {'config': {'model': circuit['name'], 'arch': circuit['name'],
                'num_layers': 1, 'embed_dim': 2, 'num_heads': 1,
                'num_kv_heads': 1, 'head_dim': 2, 'intermediate_size': 2,
                'context_length': 5, 'max_seq_len': 5, 'vocab_size': 1,
                'activation_buffer_dtypes': {'durations': 'i32',
                    'input_extent': 'i32', 'output_extent': 'i32'}},
                'entries': entries, 'quant_summary': {},
                'template': circuit}
            layout, calls, _library, _loaded, fn = compile_native_graph(
                root, source, path)
            self.assertEqual(calls['errors'], [])
            self.assertEqual(calls['operations'][-1]['function'],
                'audio_conv_transpose1d_dense_channel_major_f32_checked')
            fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32)]
            size = layout['memory']['arena']['total_size']
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            entries = {item['name']: item for item in entries}
            for item in layout['memory']['weights']['entries']:
                entry = entries[item['name']]
                start = entry['file_offset']
                arena[item['abs_offset']:item['abs_offset'] + item['size']] = \
                    payload[start:start + item['size']]
            locations = {item['name']: item for item in
                         layout['memory']['activations']['buffers']}
            def view(name, dtype, shape):
                return np.ndarray(shape, dtype, buffer=arena,
                                  offset=locations[name]['abs_offset'])
            duration = view('durations', np.int32, (1,))
            features = view('features', np.float32, (2, 7))
            activated = view('activated', np.float32, (2, 7))
            result = view('output', np.float32, (3, 13))
            for valid in (3, 5, 2):
                duration[0] = valid
                features[:] = np.nan
                features[:, :valid] = np.arange(2 * valid,
                    dtype=np.float32).reshape(2, valid) / 4 - 1
                activated[:] = result[:] = -77.
                returned = ctypes.c_int32(-999)
                self.assertEqual(fn(arena, len(arena), ctypes.byref(returned)), 0)
                self.assertEqual(returned.value, 2 * valid)
                expected = np.zeros((3, 2 * valid), np.float64)
                x = np.where(features[:, :valid] >= 0,
                             features[:, :valid], features[:, :valid] * .1)
                expected[:] = weights['stage.bias'][:, None]
                for ic in range(2):
                    for frame in range(valid):
                        for tap in range(3):
                            target = 2 * frame + tap - 1
                            if 0 <= target < 2 * valid:
                                expected[:, target] += (float(x[ic, frame]) *
                                    weights['stage.weight'][ic, :, tap])
                np.testing.assert_allclose(result[:, :2 * valid], expected,
                                           rtol=0, atol=2e-6)
                self.assertTrue(np.all(result[:, 2 * valid:] == -77.))
                self.assertTrue(np.all(activated[:, valid:] == -77.))
            duration[0] = 6
            activated[:] = result[:] = -77.
            returned = ctypes.c_int32(-999)
            self.assertNotEqual(fn(arena, len(arena), ctypes.byref(returned)), 0)
            self.assertEqual(returned.value, -999)
            self.assertTrue(np.all(activated == -77.))
            self.assertTrue(np.all(result == -77.))
            duration[0] = 3
            features[:, :3] = np.arange(6, dtype=np.float32).reshape(2, 3)
            weight_item = next(item for item in
                layout['memory']['weights']['entries']
                if item['name'] == 'stage.weight')
            bound_weight = np.ndarray((18,), np.float32, buffer=arena,
                                      offset=weight_item['abs_offset'])
            original = bound_weight[0].copy()
            bound_weight[0] = np.nan
            result[:] = -77.
            returned = ctypes.c_int32(-999)
            self.assertNotEqual(fn(arena, len(arena), ctypes.byref(returned)), 0)
            self.assertEqual(returned.value, -999)
            self.assertTrue(np.all(result == -77.))
            bound_weight[0] = original
            self.assertEqual(fn(arena, len(arena), ctypes.byref(returned)), 0)
            self.assertEqual(returned.value, 6)


if __name__ == '__main__':
    unittest.main()
