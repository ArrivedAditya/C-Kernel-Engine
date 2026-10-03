"""Pinned harmonic source through generated linear and tanh calls."""

import ctypes
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from export_kokoro_bump import write_bundle
from tests.v8_checked_graph_test_support import compile_native_graph


class GeneratedSourceProjectionTest(unittest.TestCase):
    def test_pinned_source_projection(self):
        fixture = ROOT / 'tests/fixtures/tts/kokoro_source_pinned.npz'
        meta = json.loads(fixture.with_suffix('.json').read_text())
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),
                         meta['fixture_sha256'])
        with np.load(fixture) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        frames, upsample, harmonics = 206, 300, 9
        samples = frames * upsample
        q = frames * harmonics + samples * harmonics
        weight_name = 'waveform_decoder.generator.m_source.l_linear.weight'
        bias_name = 'waveform_decoder.generator.m_source.l_linear.bias'
        tensors = {weight_name: arrays['linear_weight'],
                   bias_name: arrays['linear_bias']}
        origins = {name: {'source_name': name, 'transform': 'identity'}
                   for name in tensors}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = write_bundle(root, tensors, origins,
                {'n_token': 1, 'hidden_dim': harmonics,
                 'plbert': {'intermediate_size': harmonics,
                            'max_position_embeddings': samples,
                            'num_attention_heads': 1}},
                {'source': 'pinned Kokoro source projection fixture'})
            bump = (root / 'weights.bump').read_bytes()
            buffers = {'f0': {'shape': [frames]},
                'gaussian': {'shape': [samples, harmonics]},
                'harmonics': {'shape': [samples, harmonics]},
                'projected': {'shape': [samples, 1]},
                'source': {'shape': [samples, 1]}}
            ops = [
                {'id': 'harmonics', 'op': 'audio_harmonic_source_checked',
                 'kernel': 'audio_harmonic_source_checked_f32', 'returns_status': True,
                 'graph_slots': {'inputs': {'f0': 'external:f0',
                     'gaussian': 'external:gaussian'},
                     'outputs': {'output': 'harmonics'}},
                 'params': {'F': frames, 'S': samples, 'H': harmonics, 'Q': q,
                     'upsample': upsample, 'harmonics': harmonics,
                     'sample_rate': 24000., 'voiced_threshold': 10.,
                     'sine_amp': .1, 'noise_std': .003,
                     'call_constants': {'source_f0_capacity': frames,
                         'source_gaussian_capacity': samples * harmonics,
                         'source_output_capacity': samples * harmonics,
                         'source_scratch_capacity': q,
                         'source_frames': frames}}},
                {'id': 'linear', 'op': 'linear_rows_checked',
                 'kernel': 'linear_rows_checked_f32', 'returns_status': True,
                 'weight_refs': {'weight': weight_name, 'bias': bias_name},
                 'graph_slots': {'inputs': {'input': 'harmonics'},
                     'outputs': {'output': 'projected'}},
                 'params': {'M': samples, 'K': harmonics, 'N': 1,
                     'call_constants': {'linear_input_elements': samples * harmonics,
                         'linear_input_stride': harmonics,
                         'linear_weight_elements': harmonics,
                         'linear_weight_stride': harmonics,
                         'linear_bias_elements': 1,
                         'linear_output_elements': samples,
                         'linear_output_stride': 1,
                         'linear_rows': samples,
                         'linear_input_channels': harmonics,
                         'linear_output_channels': 1}}},
                {'id': 'tanh', 'op': 'tanh_strided_checked',
                 'kernel': 'tanh_strided_f32_checked', 'returns_status': True,
                 'graph_slots': {'inputs': {'input': 'projected'},
                     'outputs': {'output': 'source'}},
                 'params': {'R': samples, 'C': 1,
                     'call_constants': {'tanh_input_elements': samples,
                         'tanh_input_stride': 1, 'tanh_output_elements': samples,
                         'tanh_output_stride': 1, 'tanh_rows': samples,
                         'tanh_columns': 1}}}]
            circuit = {'version': 3, 'name': 'kokoro_source_projection_fixed',
                'family': 'bounded_graph', 'checked_native_entry': True,
                'contract': {'runtime_invariants': {'inference_only': True}},
                'activation_buffers': buffers,
                'activation_bindings': {name: name for name in buffers},
                'native_entry': {'function': 'ck_kokoro_source_projection_fixed',
                    'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                        {'c_type': 'size_t', 'name': 'arena_bytes'}],
                    'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'}},
                'sequence': ['component'], 'block_types': {'component': {
                    'sequence': ['header', 'body', 'footer'], 'header': [],
                    'body': {'type': 'dense', 'ops': ops}, 'footer': []}}}
            path = root / 'circuit.json'
            path.write_text(json.dumps(circuit))
            source = {'config': {'model': circuit['name'], 'arch': circuit['name'],
                'num_layers': 1, 'embed_dim': harmonics, 'num_heads': 1,
                'num_kv_heads': 1, 'head_dim': harmonics,
                'intermediate_size': harmonics,
                'context_length': samples, 'max_seq_len': samples,
                'vocab_size': 1},
                'entries': bundle['entries'], 'quant_summary': {},
                'template': circuit}
            layout, calls, _library, _loaded, fn = compile_native_graph(
                root, source, path)
            self.assertEqual(calls['errors'], [])
            self.assertEqual([op['function'] for op in calls['operations']],
                ['audio_harmonic_source_checked_f32', 'linear_rows_checked_f32',
                 'tanh_strided_f32_checked'])
            size = layout['memory']['arena']['total_size']
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            entries = {entry['name']: entry for entry in bundle['entries']}
            for item in layout['memory']['weights']['entries']:
                entry = entries[item['name']]
                payload = bump[entry['file_offset']:entry['file_offset'] + entry['size']]
                arena[item['abs_offset']:item['abs_offset'] + len(payload)] = payload
            locations = {item['name']: item for item in
                         layout['memory']['activations']['buffers']}
            def view(name, shape):
                return np.ndarray(shape, np.float32, buffer=arena,
                    offset=locations[name]['abs_offset'])
            view('f0', (frames,))[:] = arrays['f0'].reshape(-1)
            view('gaussian', (samples, harmonics))[:] = arrays['gaussian'].reshape(samples, harmonics)
            output = view('source', (samples,))
            output.fill(-91.)
            self.assertEqual(fn(arena, len(arena)), 0)
            self.assertTrue(np.isfinite(output).all())
            harmonic_actual = view('harmonics', (samples, harmonics))
            harmonic_reference = arrays['sine_waves'].reshape(samples, harmonics)
            harmonic_error = np.abs(harmonic_actual - harmonic_reference)
            harmonic_worst = float(np.max(harmonic_error))
            self.assertLessEqual(harmonic_worst, 5e-4)
            isolated = np.tanh(harmonic_reference @
                arrays['linear_weight'].reshape(harmonics, 1) +
                arrays['linear_bias'].reshape(1)).reshape(-1)
            expected = arrays['source'].reshape(-1)
            self.assertLessEqual(float(np.max(np.abs(isolated - expected))), 1e-6)
            error = np.abs(output - expected)
            worst = int(np.argmax(error))
            # tanh is 1-Lipschitz; this bounds amplification of the already
            # measured harmonic-source error through the pinned linear weight.
            bound = harmonic_worst * float(np.abs(arrays['linear_weight']).sum()) + 2e-6
            self.assertLessEqual(float(error[worst]), bound,
                (worst, float(output[worst]), float(expected[worst])))
            first = output.copy()
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, first)
            gaussian = view('gaussian', (samples, harmonics))
            original = float(gaussian[0, 0])
            gaussian[0, 0] = np.nan
            output.fill(-91.)
            self.assertNotEqual(fn(arena, len(arena)), 0)
            self.assertTrue(np.all(output == -91.))
            gaussian[0, 0] = original
            self.assertEqual(fn(arena, len(arena)), 0)
            np.testing.assert_array_equal(output, first)
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': 'kokoro.source-projection.generated-vs-pinned',
                'name': 'generated harmonics linear tanh versus direct full-model source',
                'provider': 'generated_kokoro_source_projection',
                'oracle': 'pinned-full-kmodel-pytorch28', 'status': 'pass',
                'max_diff': float(error[worst]), 'tolerance': bound,
                'worst_index': worst, 'configuration': '206 F0 frames; captured Gaussian'}))


if __name__ == '__main__':
    unittest.main()
