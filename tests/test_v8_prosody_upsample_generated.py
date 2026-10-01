"""Normal lowering and generated C carry a checked extent through two consumers."""
import ctypes
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


class GeneratedProsodyUpsampleTest(unittest.TestCase):
    def test_extent_and_physical_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ops = [
                {'id': 'sum', 'op': 'runtime_extent_sum',
                 'kernel': 'runtime_extent_sum_i32', 'returns_status': True,
                 'produces_runtime_lengths': {'source_frames': 'valid_extent'},
                 'params': {'T': 1},
                 'graph_slots': {'inputs': {'values': 'external:durations'},
                                 'outputs': {'valid_extent': 'source_extent'}}},
                {'id': 'scale', 'op': 'runtime_extent_scale',
                 'kernel': 'runtime_extent_scale_i32', 'returns_status': True,
                 'consumes_runtime_lengths': ['source_frames'],
                 'produces_runtime_lengths': {'output_frames': 'valid_extent'},
                 'runtime_scalar_bindings': {'extent_scale_source': 'source_frames'},
                 'params': {'call_constants': {'extent_scale_factor': 2,
                                              'extent_scale_capacity': 10}},
                 'graph_slots': {'inputs': {},
                                 'outputs': {'valid_extent': 'scaled_extent'}}},
                {'id': 'nearest', 'op': 'audio_upsample_nearest_checked',
                 'kernel': 'audio_upsample_nearest_channel_major_f32_checked',
                 'returns_status': True,
                 'consumes_runtime_lengths': ['source_frames', 'output_frames'],
                 'runtime_scalar_bindings': {'nearest_input_frames': 'source_frames',
                                             'nearest_output_frames': 'output_frames'},
                 'params': {'C': 2, 'T': 5, 'U': 10, 'call_constants': {
                     'nearest_input_elements': 14, 'nearest_input_stride': 7,
                     'nearest_output_elements': 26, 'nearest_output_stride': 13,
                     'nearest_channels': 2, 'nearest_factor': 2}},
                 'graph_slots': {'inputs': {'input': 'external:features'},
                                 'outputs': {'output': 'nearest_output'}}},
                {'id': 'transpose_conv',
                 'op': 'audio_conv_transpose1d_depthwise_checked',
                 'kernel': 'audio_conv_transpose1d_depthwise_channel_major_f32_checked',
                 'returns_status': True,
                 'consumes_runtime_lengths': ['source_frames', 'output_frames'],
                 'runtime_scalar_bindings': {'deconv_input_frames': 'source_frames',
                                             'deconv_output_frames': 'output_frames'},
                 'weight_refs': {'weight': 'deconv.weight', 'bias': 'deconv.bias'},
                 'params': {'C': 2, 'T': 5, 'K': 3, 'U': 10, 'Q': 20,
                     'call_constants': {'deconv_input_elements': 14,
                         'deconv_input_stride': 7, 'deconv_weight_elements': 6,
                         'deconv_bias_elements': 2, 'deconv_output_elements': 26,
                         'deconv_output_stride': 13, 'deconv_scratch_elements': 20,
                         'deconv_channels': 2, 'deconv_kernel_size': 3,
                         'deconv_stride': 2, 'deconv_padding': 1,
                         'deconv_output_padding': 1}},
                 'graph_slots': {'inputs': {'input': 'external:features'},
                                 'outputs': {'output': 'conv_output'}}},
            ]
            buffers = {'durations': {'shape': [1]},
                       'source_extent': {'shape': [1]},
                       'scaled_extent': {'shape': [1]},
                       'features': {'shape': [2, 7]},
                       'nearest_output': {'shape': [2, 13]},
                       'conv_output': {'shape': [2, 13]}}
            circuit = {'version': 3, 'name': 'synthetic_prosody_upsample',
                'family': 'bounded_graph', 'checked_native_entry': True,
                'contract': {'runtime_invariants': {'inference_only': True}},
                'activation_buffers': buffers,
                'activation_bindings': {key: key for key in buffers},
                'runtime_constants': {'value_elements': 1, 'phoneme_count': 1,
                    'max_duration': 5, 'expanded_capacity': 5},
                'runtime_lengths': {
                    'source_frames': {'producer': 'sum', 'result': 'valid_extent',
                                      'capacity': 5, 'allow_zero': False},
                    'output_frames': {'producer': 'scale', 'result': 'valid_extent',
                                      'capacity': 10, 'allow_zero': False}},
                'native_entry': {'function': 'ck_synthetic_prosody_upsample',
                    'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                               {'c_type': 'size_t', 'name': 'arena_bytes'},
                               {'c_type': 'int32_t *', 'name': 'out_frames'}],
                    'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'},
                    'runtime_length_outputs': {'output_frames': 'out_frames'}},
                'sequence': ['component'], 'block_types': {'component': {
                    'sequence': ['header', 'body', 'footer'], 'header': [ops[0]],
                    'body': {'type': 'dense', 'ops': ops[1:3]},
                    'footer': [ops[3]]}}}
            path = root / 'circuit.json'; path.write_text(json.dumps(circuit))
            # Synthetic weights use CKE's canonical bundle layout.
            from version.v8.tts import export_kokoro_bump as exporter
            weights = {'deconv.weight': np.ones((2, 1, 3), np.float32),
                       'deconv.bias': np.zeros((2,), np.float32)}
            bundle = exporter.write_bundle(root, weights,
                {key: {'source_name': key, 'transform': 'identity'} for key in weights},
                {'n_token': 1, 'hidden_dim': 2, 'plbert': {
                    'intermediate_size': 2, 'max_position_embeddings': 5,
                    'num_attention_heads': 1}},
                {'source': 'synthetic provider composition'})
            source = {'config': {'model': circuit['name'], 'arch': circuit['name'],
                'num_layers': 1, 'embed_dim': 2, 'num_heads': 1,
                'num_kv_heads': 1, 'head_dim': 2, 'intermediate_size': 2,
                'context_length': 5, 'max_seq_len': 5, 'vocab_size': 1,
                'activation_buffer_dtypes': {'durations': 'i32',
                    'source_extent': 'i32', 'scaled_extent': 'i32'}},
                'entries': bundle['entries'], 'quant_summary': {},
                'template': circuit}
            layout, calls, _library, _loaded, fn = compile_native_graph(root, source, path)
            self.assertEqual(calls['errors'], [])
            fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32)]
            entries = {item['name']: item for item in bundle['entries']}
            bump = (root / 'weights.bump').read_bytes()
            size = layout['memory']['arena']['total_size']
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(raw,
                (-ctypes.addressof(raw)) & 63)
            locations = {item['name']: item for item in
                layout['memory']['activations']['buffers']}
            def view(name, dtype, shape):
                return np.ndarray(shape, dtype, buffer=arena,
                                  offset=locations[name]['abs_offset'])
            for item in layout['memory']['weights']['entries']:
                entry = entries[item['name']]
                start = entry['file_offset']
                arena[item['abs_offset']:item['abs_offset'] + item['size']] = \
                    bump[start:start + item['size']]
            duration = view('durations', np.int32, (1,))
            features = view('features', np.float32, (2, 7))
            nearest = view('nearest_output', np.float32, (2, 13))
            transposed = view('conv_output', np.float32, (2, 13))
            features[:] = np.nan
            features[:, :5] = np.arange(10, dtype=np.float32).reshape(2, 5)
            for valid in (3, 5, 2):
                duration[0] = valid
                nearest[:] = transposed[:] = -77.
                returned = ctypes.c_int32(-999)
                self.assertEqual(fn(arena, len(arena), ctypes.byref(returned)), 0)
                self.assertEqual(returned.value, 2 * valid)
                np.testing.assert_array_equal(nearest[:, :2 * valid],
                    np.repeat(features[:, :valid], 2, axis=1))
                self.assertTrue(np.all(nearest[:, 2 * valid:] == -77.))
                self.assertTrue(np.isfinite(transposed[:, :2 * valid]).all())
                self.assertTrue(np.all(transposed[:, 2 * valid:] == -77.))
            duration[0] = 6
            nearest[:] = transposed[:] = -77.
            returned = ctypes.c_int32(-999)
            self.assertNotEqual(fn(arena, len(arena), ctypes.byref(returned)), 0)
            self.assertEqual(returned.value, -999)
            self.assertTrue(np.all(nearest == -77.))
            self.assertTrue(np.all(transposed == -77.))


if __name__ == '__main__':
    unittest.main()
