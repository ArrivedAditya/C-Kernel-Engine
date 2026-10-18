"""One generated Kokoro entry produces bounded waveforms at runtime lengths."""

import ctypes
import hashlib
import json
from pathlib import Path
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_duration_fixture_support import prepare_duration_fixture
from tests.test_v8_kokoro_generator_stage0_join import join_weights_and_references
from tests.test_v8_kokoro_generator_stage0_pool import stage0_pool_weights
from tests.test_v8_kokoro_generated_decoder_complete import (
    checkpoint_metrics, require_matching_reference_identity)
from tests.test_v8_kokoro_generated_prosody_complete import verified_fixture
from tests import test_v8_kokoro_runtime_stage1_ingress as ingress_support
from tests import test_v8_kokoro_source_residual_first_pair as fixture_support
from tests import test_v8_kokoro_generated_albert_layer as xray_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from build_kokoro_runtime_waveform_circuit import OUTPUT, build_circuit


class KokoroRuntimeWaveformTest(unittest.TestCase):
    view = ingress_support.KokoroRuntimeStage1IngressTest.view
    arena = ingress_support.KokoroRuntimeStage1IngressTest.arena

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        graph = build_circuit()
        if OUTPUT.read_text() != json.dumps(graph, indent=2) + '\n':
            raise RuntimeError('committed runtime waveform differs')
        cls.circuit = cls.root / 'circuit.json'
        cls.circuit.write_text(json.dumps(graph))
        cls.refs = join_weights_and_references()
        cls.hello8, hello_meta = verified_fixture('runtime_decoder_hello_8')
        cls.hello13, thirteen_meta = verified_fixture(
            'runtime_waveform_hello_there_13')
        first, first_meta = fixture_support.load_fixture(
            'kokoro_generator_main_resblock0_pinned')
        tail, tail_meta = fixture_support.load_fixture(
            'kokoro_generator_stage0_pool_pinned')
        stage1, stage1_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_ingress_pinned')
        cls.source, source_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_source_conv_pinned')
        cls.join, join_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_join_pinned')
        cls.pool, pool_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_pool_pinned')
        cls.spectral, spectral_meta = fixture_support.load_fixture(
            'kokoro_spectral_head_pinned')
        cls.spectral_meta = spectral_meta
        require_matching_reference_identity(cls.refs.join_meta, hello_meta)
        require_matching_reference_identity(cls.refs.join_meta, thirteen_meta)
        for meta in (first_meta, tail_meta, stage1_meta, source_meta,
                     join_meta, pool_meta, spectral_meta):
            for field in ('model_pin', 'code_pin', 'asset_sha256'):
                if meta[field] != cls.refs.join_meta[field]:
                    raise RuntimeError(f'second source join {field} identity differs')
        weights = stage0_pool_weights(cls.refs, first, tail)
        weights['waveform_decoder.generator.ups.1.weight'] = stage1['weight']
        weights['waveform_decoder.generator.ups.1.bias'] = stage1['bias']
        weights['waveform_decoder.generator.noise_convs.1.weight'] = cls.source['weight']
        weights['waveform_decoder.generator.noise_convs.1.bias'] = cls.source['bias']
        weights.update({key: value for key, value in cls.join.items()
                        if key.startswith('waveform_decoder.')})
        weights.update({key: value for key, value in cls.pool.items()
                        if key.startswith('waveform_decoder.')})
        weights['waveform_decoder.generator.conv_post.weight'] = \
            cls.spectral['weight']
        weights['waveform_decoder.generator.conv_post.bias'] = \
            cls.spectral['bias']
        cls.fixture = prepare_duration_fixture(cls.root, cls.circuit, weights)
        cls.fixture.source['config']['activation_buffer_dtypes'].update({
            'request_length': 'i32', 'valid_length': 'i32',
            'valid_residual_elements': 'i32',
            'generator_stage1_deconv_extent_value': 'i32',
            'generator_stage1_pad_extent_value': 'i32'})
        cls.fixture.source['config']['context_length'] = 64
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(cls.root, cls.fixture.source, cls.circuit))
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + [
            ctypes.POINTER(ctypes.c_int32)] * 9
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}
        cls.weights = {item['name']: item for item in
                       cls.layout['memory']['weights']['entries']}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def bind_request(self, arena, count, ids, predictor_style, decoder_style,
                     gaussian):
        words = self.view(arena, 'word_ids', (64,), np.int32)
        words[:] = 178
        words[:count] = ids[:count]
        self.view(arena, 'request_length', (1,), np.int32)[0] = count
        self.view(arena, 'predictor_style', (128,))[:] = predictor_style
        self.view(arena, 'decoder_style', (128,))[:] = decoder_style
        noise = self.view(arena, 'source_gaussian', (76800, 9))
        noise[:] = 0
        noise[:gaussian.size // 9] = gaussian.reshape(-1, 9)

    def execute(self, arena, count, ids, predictor_style, decoder_style,
                gaussian):
        self.bind_request(arena, count, ids, predictor_style, decoder_style,
                          gaussian)
        for name in ('generator_stage1_source_conv',
                     'stage1_source_res_pair2_output',
                     'generator_stage1_join',
                     'stage1_main_resblock3_pair2_output',
                     'stage1_main_resblock4_pair2_output',
                     'stage1_main_resblock5_pair2_output',
                     'generator_stage1_residual_sum34',
                     'generator_stage1_residual_mean',
                     'generator_stage1_final_activation'):
            self.view(arena, name, (128, 15361))[:] = -91.
        self.view(arena, 'generator_conv_post', (22, 15361))[:] = -91.
        for name in ('generator_magnitude', 'generator_phase'):
            self.view(arena, name, (15361, 11))[:] = -91.
        self.view(arena, 'generator_waveform', (76800,))[:] = -91.
        lengths = [ctypes.c_int32(-1) for _ in range(9)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(value) for value in lengths))
        return status, [value.value for value in lengths]

    def test_standalone_13_id_diagnostic_wav(self):
        """The native host serializes the generated entry's valid samples."""
        root = self.root / 'standalone-runtime-waveform'
        root.mkdir()
        shutil.copy2(self.library, root / 'generated.so')
        arena = self.arena()
        reference = self.hello13
        self.bind_request(arena, 13, reference['word_ids'],
            reference['predictor_style'].ravel(),
            reference['decoder_style'].ravel(),
            reference['generator_harmonic_gaussian'])
        expected_lengths = [ctypes.c_int32(-1) for _ in range(9)]
        self.assertEqual(self.fn(arena, len(arena), *(
            ctypes.byref(value) for value in expected_lengths)), 0)
        self.assertEqual(expected_lengths[5].value, 36600)
        expected_pcm = np.rint(np.clip(self.view(arena,
            'generator_waveform', (76800,))[:36600], -1, 1) *
            np.float32(32767)).astype('<i2').tobytes()
        output = self.buffers['generator_waveform']
        package = root / 'prepared-captured-noise.cktts'
        external = {'word_ids', 'type_ids', 'request_length',
                    'predictor_style', 'decoder_style', 'source_gaussian',
                    'source_stft_window', 'source_stft_cos', 'source_stft_sin'}
        self.assertTrue(external <= self.buffers.keys())
        regions = sorted([*self.layout['memory']['weights']['entries'],
            *(self.buffers[name] for name in external)],
            key=lambda item: item['abs_offset'])
        segments = len(regions)
        with package.open('wb') as stream:
            stream.write(b'CKTTS001')
            stream.write(struct.pack('<QQQI', len(arena),
                output['abs_offset'], output['size'] // 4, segments))
            for item in regions:
                offset, length = item['abs_offset'], item['size']
                stream.write(struct.pack('<QQ', offset, length))
                stream.write(bytes(arena[offset:offset + length]))
        digest = hashlib.sha256((root / 'generated.so').read_bytes()).hexdigest()
        binary = root / 'native-kokoro-runtime-wav'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra',
            '-Werror', '-pedantic', '-DCK_TTS_RUNTIME_WAVEFORM',
            f'-DCK_TTS_EXPECTED_LIBRARY_SHA256="{digest}"',
            '-I', str(ROOT / 'include'), '-I', str(ROOT / 'src/kernels'),
            str(ROOT / 'version/v8/tts/native_kokoro_wav_host.c'),
            str(ROOT / 'src/kernels/audio_noise_counter.c'),
            '-L', str(root), '-l:generated.so', '-Wl,-rpath,$ORIGIN',
            '-lm', '-ldl', '-lcrypto', '-o', str(binary)],
            check=True, capture_output=True, text=True)
        wav = root / 'hello-there-13-diagnostic.wav'
        run = subprocess.run([str(binary), str(package), str(wav)],
                             cwd=root, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('artifact SHA-256 verified', run.stderr)
        with wave.open(str(wav), 'rb') as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(),
                audio.getframerate(), audio.getnframes()),
                (1, 2, 24000, 36600))
            pcm = audio.readframes(36600)
            self.assertTrue(any(pcm))
            self.assertEqual(pcm, expected_pcm)
        published = json.loads((ROOT / 'version/v8/tts/evidence/'
            'kokoro_runtime_waveform_20261018.json').read_text())
        committed_wav = ROOT / published['retained_wav']
        self.assertEqual(hashlib.sha256(committed_wav.read_bytes()).hexdigest(),
                         published['wav_sha256'])
        evidence = {'scope': '13-ID captured-Gaussian diagnostic WAV; '
                'connected numerical parity remains FAIL',
            'text': 'Hello there.', 'voice': 'af_heart',
            'voice_row': 10, 'samples': 36600, 'sample_rate': 24000,
            'library_sha256': digest,
            'package_sha256': hashlib.sha256(package.read_bytes()).hexdigest(),
            'wav_sha256': hashlib.sha256(wav.read_bytes()).hexdigest(),
            'model_pin': self.spectral_meta['model_pin'],
            'code_pin': self.spectral_meta['code_pin'],
            'segments': segments,
            'reproduction_command': './native-kokoro-runtime-wav '
                'prepared-captured-noise.cktts hello-there-13-diagnostic.wav'}
        (root / 'reproduction.json').write_text(json.dumps(evidence, indent=2))
        retained = os.environ.get('CKE_TTS_RETAIN_DIR')
        if retained:
            destination = Path(retained) / 'runtime-waveform-13'
            if destination.exists():
                raise RuntimeError('retained diagnostic output already exists')
            shutil.copytree(root, destination)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.runtime-waveform.standalone-13',
            'name': 'standalone native generated 13-ID diagnostic WAV',
            'provider': 'generated_kokoro_runtime_waveform',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'native WAV header and retained pinned full-model capture',
            'status': 'pass', 'gate': 'execution', 'blocking': True,
            'max_diff': 0, 'tolerance': 0,
            'configuration': '13 IDs; af_heart row 10; captured Gaussian',
            'wav_sha256': evidence['wav_sha256'],
            'reproduction_command': evidence['reproduction_command']}))

    def test_three_lengths_waveform_and_recovery(self):
        graph = build_circuit()
        self.assertEqual(graph['sequence'][-1],
                         'generator_inverse_stft')
        self.assertEqual(self.calls['errors'], [])
        self.assertEqual(self.calls['operations'][-1]['function'],
                         'audio_istft_mag_phase_f32')
        runtime = xray_support.xray_builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_runtime_waveform_bounded')
        self.assertEqual(runtime['sha256'],
                         hashlib.sha256(self.library.read_bytes()).hexdigest())
        arena = self.arena()
        first8 = None
        for count, reference, frames, stage0, samples, row, noise_name in (
                (8, self.hello8, 54, 1080, 32400, 5,
                 'generator_harmonic_gaussian'),
                (13, self.hello13, 61, 1220, 36600, 10,
                 'generator_harmonic_gaussian'),
                (36, self.refs.source, 103, 2060, 61800, 33, 'gaussian'),
                (8, self.hello8, 54, 1080, 32400, 5,
                 'generator_harmonic_gaussian')):
            ids = (reference['word_ids'] if count != 36 else
                   self.fixture.encoder['word_ids'])
            predictor = (reference['predictor_style'].ravel()
                         if count != 36 else
                         self.fixture.duration['predictor_style'])
            decoder = (reference['decoder_style'].ravel()
                       if count != 36 else
                       self.refs.main['decoder_style'].ravel())
            status, lengths = self.execute(arena, count, ids, predictor,
                                           decoder, reference[noise_name])
            valid = stage0 * 6 + 1
            self.assertEqual(status, 0)
            self.assertEqual(lengths, [count, count * 768, frames,
                frames * 2, stage0, samples, valid, valid - 1, valid])
            joined = self.view(arena, 'generator_stage1_join',
                               (128, 15361))
            blocks = [self.view(arena,
                f'stage1_main_resblock{index}_pair2_output',
                (128, 15361)) for index in (3, 4, 5)]
            mean = self.view(arena, 'generator_stage1_residual_mean',
                             (128, 15361))
            output = self.view(arena, 'generator_stage1_final_activation',
                               (128, 15361))
            conv = self.view(arena, 'generator_conv_post', (22, 15361))
            magnitude = self.view(arena, 'generator_magnitude', (15361, 11))
            phase = self.view(arena, 'generator_phase', (15361, 11))
            waveform = self.view(arena, 'generator_waveform', (76800,))
            self.assertTrue(np.isfinite(output[:, :valid]).all())
            self.assertTrue(np.isfinite(waveform[:samples]).all())
            self.assertTrue(np.all(waveform[samples:] == -91.))
            self.assertTrue(np.all(conv[:, valid:] == -91.))
            self.assertTrue(np.all(magnitude[valid:] == -91.))
            self.assertTrue(np.all(phase[valid:] == -91.))
            self.assertTrue(np.all(output[:, valid:] == -91.))
            self.assertTrue(np.all(mean[:, valid:] == -91.))
            for block in blocks:
                self.assertTrue(np.all(block[:, valid:] == -91.))
            np.testing.assert_allclose(mean[:, :valid],
                ((blocks[0][:, :valid] + blocks[1][:, :valid]) +
                 blocks[2][:, :valid]) * np.float32(1.0 / 3.0),
                rtol=0, atol=2e-6)
            np.testing.assert_allclose(output[:, :valid],
                np.where(mean[:, :valid] >= 0, mean[:, :valid],
                         mean[:, :valid] * np.float32(.01)),
                rtol=0, atol=2e-6)
            if count == 8 and first8 is not None:
                np.testing.assert_array_equal(waveform, first8)
                continue
            if count != 36:
                if count == 8:
                    first8 = waveform.copy()
                expected = reference[
                    'decoder_generator_stage1_final_activation'][0]
                reference_conv = reference['decoder_generator_conv_post'][0]
                reference_magnitude = reference['istft_magnitude'][0]
                reference_phase = reference['istft_phase'][0]
                reference_waveform = reference['waveform_f32']
            else:
                expected = self.pool['final_activation'][0]
                reference_conv = self.spectral['conv_post'][0]
                reference_magnitude = self.spectral['magnitude'][0]
                reference_phase = self.spectral['phase'][0]
                reference_waveform = self.spectral['waveform'].ravel()
            final = checkpoint_metrics(output[:, :valid], expected)
            joined_reference = (reference[
                'decoder_generator_stage1_join'][0] if count != 36
                else self.join['join'][0])
            join_metrics = checkpoint_metrics(joined[:, :valid],
                                              joined_reference)
            conv_metrics = checkpoint_metrics(conv[:, :valid], reference_conv)
            magnitude_metrics = checkpoint_metrics(
                magnitude[:valid].T, reference_magnitude)
            phase_metrics = checkpoint_metrics(phase[:valid].T, reference_phase)
            waveform_metrics = checkpoint_metrics(waveform[:samples],
                                                  reference_waveform)
            reference_rms = float(np.sqrt(np.mean(
                reference_waveform.astype(np.float64) ** 2)))
            reference_peak = float(np.max(np.abs(reference_waveform)))
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'kokoro.runtime-waveform.{count}',
                'name': 'variable-length connected waveform versus full model',
                'provider': 'generated_kokoro_runtime_waveform',
                'dtype': 'fp32', 'direction': 'inference',
                'oracle': 'pinned PyTorch 2.8 direct full-model captures',
                'status': 'fail' if waveform_metrics['max_abs'] > 2e-4 else 'pass',
                'gate': 'diagnostic', 'blocking': False,
                'reason': 'Connected source and join differences remain '
                          'under numerical investigation',
                'configuration': f'{count} IDs; af_heart row {row}; '
                    f'frames={valid}; captured Gaussian',
                'max_diff': waveform_metrics['max_abs'], 'tolerance': 2e-4,
                'rmse': waveform_metrics['rmse'],
                'worst_index': waveform_metrics['worst_index'],
                'actual': waveform_metrics['actual'],
                'reference': waveform_metrics['reference'],
                'checkpoint_metrics': {'join': join_metrics,
                    'final_activation': final, 'conv_post': conv_metrics,
                    'magnitude': magnitude_metrics, 'phase': phase_metrics,
                    'waveform': waveform_metrics},
                'reference_waveform_rms': reference_rms,
                'reference_waveform_peak': reference_peak,
                'waveform_rmse_over_reference_rms':
                    waveform_metrics['rmse'] / reference_rms,
                'reproduction_command': 'python3 -m unittest '
                    'tests.test_v8_kokoro_runtime_waveform',
            }, sort_keys=True))

        bad = self.hello8['generator_harmonic_gaussian'].copy()
        bad.reshape(-1)[0] = np.nan
        args = (8, self.hello8['word_ids'],
                self.hello8['predictor_style'].ravel(),
                self.hello8['decoder_style'].ravel())
        status, lengths = self.execute(arena, *args, bad)
        self.assertNotEqual(status, 0)
        self.assertEqual(lengths, [-1] * 9)
        output = self.view(arena, 'generator_stage1_final_activation',
                           (128, 15361))
        self.assertTrue(np.all(output == -91.))
        self.assertTrue(np.all(self.view(arena, 'generator_waveform',
                                     (76800,)) == -91.))
        self.view(arena, 'source_gaussian', (76800, 9))[0, 0] = \
            self.hello8['generator_harmonic_gaussian'].reshape(-1, 9)[0, 0]
        self.view(arena, 'request_length', (1,), np.int32)[0] = 65
        published = [ctypes.c_int32(-1) for _ in range(9)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(value) for value in published))
        self.assertNotEqual(status, 0)
        self.assertEqual([value.value for value in published], [-1] * 9)
        self.assertTrue(np.all(output == -91.))
        self.bind_request(arena, *args,
            self.hello8['generator_harmonic_gaussian'])
        published = [ctypes.c_int32(-1) for _ in range(9)]
        status = self.fn(arena, len(arena) - 1,
                         *(ctypes.byref(value) for value in published))
        self.assertNotEqual(status, 0)
        self.assertEqual([value.value for value in published], [-1] * 9)
        self.assertTrue(np.all(self.view(arena, 'generator_waveform',
                                     (76800,)) == -91.))
        status, lengths = self.execute(arena, *args,
            self.hello8['generator_harmonic_gaussian'])
        self.assertEqual((status, lengths[-1]), (0, 6481))
        np.testing.assert_array_equal(self.view(arena, 'generator_waveform',
                                               (76800,)), first8)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.runtime-waveform.three-length-safety',
            'name': 'one generated entry 8 to 13 to 36 to 8 IDs',
            'provider': 'generated_kokoro_runtime_waveform',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'fresh-eight-ID baseline and checked output extents',
            'status': 'pass', 'gate': 'execution', 'blocking': True,
            'max_diff': 0, 'tolerance': 0,
            'configuration': 'captured Gaussian; 32400,36600,61800,32400 '
                'valid samples; output padding; rejected input/arena recovery',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_kokoro_runtime_waveform',
        }, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
