"""Complete generated Kokoro encoder with one bounded runtime token capacity."""
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
import export_kokoro_bump as exporter
from build_kokoro_runtime_encoder_circuit import build_circuit
from tests.v8_kokoro_fixture_support import load_first_layer_fixtures
from tests.v8_checked_graph_test_support import compile_native_graph


class KokoroRuntimeEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tensors = load_first_layer_fixtures(cls)
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.captures = {}
        base = ROOT / 'tests/fixtures/tts'
        for tokens in (36, 2, 8, 64):
            path = base / (('kokoro_encoder_pinned.npz' if tokens == 36 else
                            f'kokoro_encoder_{tokens}_pinned.npz'))
            meta = json.loads(path.with_suffix('.json').read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta['fixture_sha256']:
                raise RuntimeError(f'pinned {tokens}-token encoder fixture hash mismatch')
            if tokens != 36 and any(meta[key] != cls.captures[36][1][key]
                                    for key in ('dependencies', 'model_pin',
                                                'asset_hashes', 'capture_source_sha256')):
                raise RuntimeError('encoder fixtures have different oracle identities')
            cls.captures[tokens] = (dict(np.load(path)), meta)
            if tokens == 36:
                for kind in ('weight', 'bias'):
                    tensors[f'phoneme_projection.{kind}'] = cls.captures[36][0][
                        f'weight__phoneme_projection__{kind}']
        origins = {name: {'source_name': name, 'transform': 'identity'}
                   for name in tensors}
        bundle = exporter.write_bundle(root, tensors, origins, {
            'n_token': 178, 'hidden_dim': 512,
            'plbert': {'intermediate_size': 2048, 'max_position_embeddings': 512,
                       'num_attention_heads': 12}},
            {'source': 'pinned independent PyTorch encoder captures'})
        exporter.verify_bundle(root)
        cls.bump = (root / 'weights.bump').read_bytes()
        cls.entries = {entry['name']: entry for entry in bundle['entries']}
        graph = build_circuit()
        circuit = root / 'circuit.json'
        circuit.write_text(json.dumps(graph))
        source = {'config': {
            'model': graph['name'], 'arch': graph['name'], 'num_layers': 1,
            'embed_dim': 128, 'num_heads': 1, 'num_kv_heads': 1,
            'head_dim': 128, 'intermediate_size': 256,
            'context_length': 64, 'max_seq_len': 64, 'vocab_size': 178,
            'T': 64, 'C': 128, 'epsilon': 1e-12,
            'activation_buffer_dtypes': {
                'word_ids': 'i32', 'type_ids': 'i32',
                'request_length': 'i32', 'valid_length': 'i32',
                'valid_residual_elements': 'i32'}},
            'entries': bundle['entries'], 'quant_summary': {},
            'template': graph}
        cls.layout, cls.call_ir, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(root, source, circuit))
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32)]
        cls.activations = {item['name']: item for item in
                           cls.layout['memory']['activations']['buffers']}
        cls.weights = {item['name']: item for item in
                       cls.layout['memory']['weights']['entries']}
        cls.arena_size = cls.layout['memory']['arena']['total_size']

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def view(self, arena, name, shape, dtype=np.float32):
        return np.ndarray(shape, dtype=dtype, buffer=arena,
                          offset=self.activations[name]['abs_offset'])

    def arena(self):
        raw = (ctypes.c_uint8 * (self.arena_size + 63))()
        arena = (ctypes.c_uint8 * self.arena_size).from_buffer(
            raw, (-ctypes.addressof(raw)) & 63)
        for name, planned in self.weights.items():
            entry = self.entries[name]
            payload = self.bump[entry['file_offset']:
                                entry['file_offset'] + entry['size']]
            self.assertEqual(hashlib.sha256(payload).hexdigest(), entry['sha256'])
            arena[planned['abs_offset']:planned['abs_offset'] + len(payload)] = payload
        return arena

    def run_request(self, arena, tokens):
        data, _meta = self.captures[tokens]
        ids = self.view(arena, 'word_ids', (64,), np.int32)
        ids[:] = 178  # An invalid ID in reserved capacity must remain unread.
        ids[:tokens] = data['word_ids']
        self.view(arena, 'type_ids', (64,), np.int32)[:] = 0
        self.view(arena, 'request_length', (1,), np.int32)[0] = tokens
        output = self.view(arena, 'phoneme_features', (64, 512))
        output[:] = -777
        for layer in range(12):
            self.view(arena, f'l{layer:02d}_albert_layer_output',
                      (64, 768))[:] = -777
        token_result = ctypes.c_int32(-1)
        residual_result = ctypes.c_int32(-1)
        status = self.fn(arena, len(arena), ctypes.byref(token_result),
                         ctypes.byref(residual_result))
        self.assertEqual(status, 0)
        self.assertEqual(token_result.value, tokens)
        self.assertEqual(residual_result.value, tokens * 768)
        self.assertTrue(np.isfinite(output[:tokens]).all())
        self.assertTrue(np.all(output[tokens:] == -777))
        for layer in range(12):
            intermediate = self.view(
                arena, f'l{layer:02d}_albert_layer_output', (64, 768))
            self.assertTrue(np.all(intermediate[tokens:] == -777))
        return output[:tokens].copy()

    def test_one_entry_matches_four_pinned_lengths(self):
        self.assertEqual(self.call_ir['errors'], [])
        self.assertEqual(len(self.call_ir['operations']), 149)
        self.assertEqual(len(self.weights), 25)
        selected = [op['call_abi']['kernel_id'] for op in
                    self.call_ir['operations'] if
                    op['function'] == 'audio_scaled_residual_add_f32']
        self.assertEqual(selected, ['audio_scaled_residual_add_runtime_f32'] * 24)
        for op in self.call_ir['operations']:
            for checkpoint in op.get('semantic_checkpoints', []):
                self.assertNotEqual(checkpoint['resolved_contract_id'], 'unresolved')
        arena = self.arena()
        first_short = None
        for tokens in (2, 8, 36, 64, 2):
            with self.subTest(tokens=tokens):
                actual = self.run_request(arena, tokens)
                reference = self.captures[tokens][0]
                expected = reference['phoneme_features']
                error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                maximum = float(error.max())
                self.assertLessEqual(maximum, 3e-5)
                layer_maximum = 0.0
                for layer in range(12):
                    name = f'l{layer:02d}_albert_layer_output'
                    intermediate = self.view(arena, name, (64, 768))[:tokens]
                    self.assertTrue(np.isfinite(intermediate).all(), name)
                    deviation = np.abs(intermediate.astype(np.float64)
                                       - reference[name].astype(np.float64))
                    layer_maximum = max(layer_maximum, float(deviation.max()))
                    self.assertLessEqual(float(deviation.max()), 3e-5, name)
                if tokens == 2 and first_short is not None:
                    np.testing.assert_array_equal(actual, first_short)
                    continue
                if tokens == 2:
                    first_short = actual.copy()
                print('CKE_NUMERICAL_CASE ' + json.dumps({
                    'case_id': f'kokoro.runtime-encoder.{tokens}.pytorch28',
                    'name': 'One runtime-bound Kokoro phoneme encoder',
                    'provider': 'generated_kokoro_runtime_phoneme_encoder',
                    'dtype': 'fp32', 'direction': 'inference',
                    'oracle': 'pinned independent PyTorch 2.8 encoder capture',
                    'backend_version': self.captures[tokens][1]['dependencies']['torch'],
                    'status': 'pass', 'gate': 'composition', 'blocking': True,
                    'max_diff': maximum, 'tolerance': 3e-5,
                    'max_layer_diff': layer_maximum,
                    'metric': 'maximum absolute final-feature error',
                    'configuration': f'capacity=64, valid={tokens}, shared layers=12',
                    'reproduction_command': 'python3 -m unittest '
                        'tests.test_v8_kokoro_runtime_encoder',
                }, sort_keys=True))
        # One token is a legal encoder geometry but has no pinned model oracle.
        self.view(arena, 'word_ids', (64,), np.int32)[:] = 178
        self.view(arena, 'word_ids', (64,), np.int32)[0] = (
            self.captures[2][0]['word_ids'][0])
        self.view(arena, 'request_length', (1,), np.int32)[0] = 1
        singleton = self.view(arena, 'phoneme_features', (64, 512))
        singleton[:] = -777
        token_result = ctypes.c_int32(-1)
        residual_result = ctypes.c_int32(-1)
        self.assertEqual(self.fn(arena, len(arena),
                                 ctypes.byref(token_result),
                                 ctypes.byref(residual_result)), 0)
        self.assertEqual((token_result.value, residual_result.value), (1, 768))
        self.assertTrue(np.isfinite(singleton[0]).all())
        self.assertTrue(np.all(singleton[1:] == -777))
        # The failed request must not publish a length or alter the prior output.
        for bad in (-1, 0, 65, 2147483647):
            self.view(arena, 'request_length', (1,), np.int32)[0] = bad
            output = self.view(arena, 'phoneme_features', (64, 512))
            snapshot = output.copy()
            token_result = ctypes.c_int32(-1)
            residual_result = ctypes.c_int32(-1)
            self.assertNotEqual(self.fn(arena, len(arena),
                                        ctypes.byref(token_result),
                                        ctypes.byref(residual_result)), 0)
            self.assertEqual((token_result.value, residual_result.value), (-1, -1))
            np.testing.assert_array_equal(output, snapshot)
        self.view(arena, 'request_length', (1,), np.int32)[0] = 8
        self.view(arena, 'word_ids', (64,), np.int32)[0] = 178
        output = self.view(arena, 'phoneme_features', (64, 512))
        snapshot = output.copy()
        token_result = ctypes.c_int32(-1)
        residual_result = ctypes.c_int32(-1)
        self.assertNotEqual(self.fn(arena, len(arena),
                                    ctypes.byref(token_result),
                                    ctypes.byref(residual_result)), 0)
        self.assertEqual((token_result.value, residual_result.value), (-1, -1))
        np.testing.assert_array_equal(output, snapshot)
        recovered = self.run_request(arena, 8)
        np.testing.assert_allclose(recovered,
                                   self.captures[8][0]['phoneme_features'],
                                   atol=3e-5, rtol=0)


if __name__ == '__main__':
    unittest.main()
