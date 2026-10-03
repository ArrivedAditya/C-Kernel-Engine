"""Connected generated source STFT/convolution with explicit raw-phase diagnostic."""

import ctypes
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_duration_fixture_support import (
    populated_duration_arena, prepare_duration_fixture)
from tests.test_v8_kokoro_generated_prosody_complete import (
    complete_prosody_weights, verified_fixture)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_generated_source_stft_conv_circuit as author


class KokoroGeneratedSourceStftConvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        source_path = ROOT / 'tests/fixtures/tts/kokoro_source_pinned.npz'
        cls.source_meta = json.loads(source_path.with_suffix('.json').read_text())
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != cls.source_meta['fixture_sha256']:
            raise RuntimeError('pinned source fixture checksum mismatch')
        with np.load(source_path) as archive:
            cls.oracle = {key: archive[key].copy() for key in archive.files}
        weights, cls.prosody_oracle, cls.prosody_meta = complete_prosody_weights()
        if (cls.source_meta['model_pin'] != cls.prosody_meta['model_pin'] or
            cls.source_meta['code_pin'] != cls.prosody_meta['code_pin'] or
            cls.source_meta['asset_sha256'] != cls.prosody_meta['asset_sha256']):
            raise RuntimeError('source and prosody fixture identities differ')
        for key, name in (
            ('linear_weight', 'waveform_decoder.generator.m_source.l_linear.weight'),
            ('linear_bias', 'waveform_decoder.generator.m_source.l_linear.bias'),
            ('source_conv_weight', 'waveform_decoder.generator.noise_convs.0.weight'),
            ('source_conv_bias', 'waveform_decoder.generator.noise_convs.0.bias')):
            weights[name] = cls.oracle[key]
        fixture = prepare_duration_fixture(Path(cls.temp.name), author.OUTPUT, weights)
        for name, value in vars(fixture).items():
            setattr(cls, name, value)
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = \
            compile_native_graph(Path(cls.temp.name), cls.source, author.OUTPUT)
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + \
            [ctypes.POINTER(ctypes.c_int32)] * 5
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def view(self, arena, name, dtype=np.float32):
        item = self.buffers[name]
        return np.ndarray((item['size'] // np.dtype(dtype).itemsize,), dtype,
            buffer=arena, offset=item['abs_offset'])

    def arena(self):
        arena = populated_duration_arena(self.layout, self.entries, self.bump,
                                         self.encoder, self.duration)
        self.view(arena, 'source_gaussian').reshape(76800, 9)[:61800] = \
            self.oracle['gaussian'].reshape(61800, 9)
        self.view(arena, 'source_stft_window')[:] = self.oracle['window']
        taps = np.arange(20, dtype=np.float64)
        angles = -2 * np.pi * np.arange(11)[:, None] * taps / 20
        self.view(arena, 'source_stft_cos').reshape(11, 20)[:] = np.cos(angles)
        self.view(arena, 'source_stft_sin').reshape(11, 20)[:] = np.sin(angles)
        self.view(arena, 'source_conv0')[:] = -91.
        return arena

    def execute(self, arena):
        outputs = [ctypes.c_int32(-999) for _ in range(5)]
        status = self.fn(arena, len(arena), *(ctypes.byref(x) for x in outputs))
        return status, tuple(x.value for x in outputs)

    def test_generated_source_stft_conv_and_known_raw_phase_failure(self):
        self.assertEqual(json.loads(author.OUTPUT.read_text()), author.build_circuit())
        self.assertEqual(self.calls['errors'], [])
        self.assertEqual([op['function'] for op in self.calls['operations'][-4:]],
            ['ck_runtime_affine_i32_checked', 'ck_runtime_scale_i32_checked',
             'audio_stft_mag_phase_checked_f32',
             'audio_conv1d_checked_channel_major_f32'])
        arena = self.arena()
        status, lengths = self.execute(arena)
        self.assertEqual(status, 0)
        self.assertEqual(lengths, (103, 206, 61800, 12361, 2060))
        spectrum = self.view(arena, 'source_stft_channels').reshape(22, 15361)[:, :12361]
        expected = np.concatenate((self.oracle['magnitude'][0],
                                   self.oracle['phase'][0]), axis=0)
        self.assertTrue(np.isfinite(spectrum).all())
        mag_error = np.abs(spectrum[:11] - expected[:11])
        phase_error = np.abs(spectrum[11:] - expected[11:])
        conv = self.view(arena, 'source_conv0').reshape(256, 2560)[:, :2060]
        self.assertTrue(np.isfinite(conv).all())
        conv_error = np.abs(conv - self.oracle['first_source_conv'][0])
        worst = np.unravel_index(np.argmax(conv_error), conv_error.shape)
        mismatches = np.argwhere(phase_error > 1)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.generated-source-stft-conv.raw-phase-v1',
            'name': 'connected generated source through scalar STFT and first source convolution',
            'provider': 'audio_stft_mag_phase_checked_f32',
            'oracle': 'direct-pinned-full-model-pytorch28',
            'status': 'fail', 'gate': 'diagnostic', 'blocking': False,
            'reason': 'known raw-phase branch-cut incompatibility; source STFT convolution parity unresolved',
            'max_diff': float(conv_error[worst]), 'tolerance': 5e-5,
            'worst_index': list(map(int, worst)),
            'max_magnitude_error': float(mag_error.max()),
            'max_raw_phase_error': float(phase_error.max()),
            'raw_phase_branch_cut_mismatch_count': len(mismatches),
            'first_raw_phase_branch_cut_mismatches': mismatches[:8].tolist(),
            'configuration': '36 phonemes; 61,800 generated source samples; captured Gaussian',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        self.assertTrue(np.all(self.view(arena, 'source_conv0').reshape(256, 2560)[:, 2060:] == -91.))

    def test_rejection_and_recovery(self):
        arena = self.arena()
        lengths = [ctypes.c_int32(-999) for _ in range(5)]
        self.assertNotEqual(self.fn(arena, len(arena) - 1,
            *(ctypes.byref(x) for x in lengths)), 0)
        self.assertEqual([x.value for x in lengths], [-999] * 5)
        self.assertTrue(np.all(self.view(arena, 'source_conv0') == -91.))
        self.assertEqual(self.execute(arena), (0, (103, 206, 61800, 12361, 2060)))
        original = self.view(arena, 'source_conv0').reshape(256, 2560)[:, :2060].copy()
        self.view(arena, 'source_stft_window')[0] = np.nan
        self.view(arena, 'source_conv0')[:] = -91.
        status, lengths = self.execute(arena)
        self.assertNotEqual(status, 0)
        self.assertEqual(lengths[-1], -999)
        self.assertTrue(np.all(self.view(arena, 'source_conv0') == -91.))
        self.view(arena, 'source_stft_window')[0] = self.oracle['window'][0]
        self.assertEqual(self.execute(arena), (0, (103, 206, 61800, 12361, 2060)))
        alternate, meta = verified_fixture('generator_stage0_your')
        self.assertEqual(meta['model_pin'], self.source_meta['model_pin'])
        self.assertEqual(meta['asset_sha256'], self.source_meta['asset_sha256'])
        self.view(arena, 'word_ids', np.int32)[:36] = alternate['word_ids']
        self.view(arena, 'predictor_style')[:128] = alternate['predictor_style']
        self.view(arena, 'source_conv0')[:] = -91.
        self.assertEqual(self.execute(arena), (0, (98, 196, 58800, 11761, 1960)))
        other = self.view(arena, 'source_conv0').reshape(256, 2560)
        self.assertTrue(np.isfinite(other[:, :1960]).all())
        self.assertTrue(np.all(other[:, 1960:] == -91.))
        self.view(arena, 'word_ids', np.int32)[:36] = self.encoder['word_ids']
        self.view(arena, 'predictor_style')[:128] = self.duration['predictor_style']
        self.view(arena, 'source_conv0')[:] = -91.
        self.assertEqual(self.execute(arena), (0, (103, 206, 61800, 12361, 2060)))
        np.testing.assert_array_equal(
            self.view(arena, 'source_conv0').reshape(256, 2560)[:, :2060], original)


if __name__ == '__main__':
    unittest.main()
