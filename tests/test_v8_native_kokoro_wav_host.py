"""Cheap native host contract for seeded excitation packages."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_COUNT = 76800 * 9
GAUSSIAN_BYTES = GAUSSIAN_COUNT * 4
ARENA_BYTES = GAUSSIAN_BYTES + 128


class NativeKokoroWavHostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        stub = cls.root / 'stub.c'
        stub.write_text('''#include <stddef.h>
#include <stdint.h>
int ck_kokoro_waveform_bounded(uint8_t *arena, size_t capacity,
    int32_t *a, int32_t *b, int32_t *c, int32_t *d,
    int32_t *e, int32_t *f, int32_t *g) {
    if (!arena || capacity < ''' + str(ARENA_BYTES) + ''') return -1;
    const float *noise = (const float *)arena;
    float *audio = (float *)(arena + ''' + str(GAUSSIAN_BYTES) + ''');
    for (int i = 0; i < 20; ++i) audio[i] = .01f * noise[i];
    *a = 1; *b = 1; *c = 1; *d = 20; *e = 5; *f = 1; *g = 1;
    return 0;
}\n''')
        cls.binary = cls.root / 'wav-host'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            '-pedantic', '-I', str(ROOT / 'include'), '-I',
            str(ROOT / 'src/kernels'),
            str(ROOT / 'version/v8/tts/native_kokoro_wav_host.c'),
            str(ROOT / 'src/kernels/audio_noise_counter.c'), str(stub),
            '-lm', '-o', str(cls.binary)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def package(self, name, gaussian_offset=0, gaussian_count=GAUSSIAN_COUNT,
                waveform_offset=GAUSSIAN_BYTES):
        path = self.root / name
        path.write_bytes(struct.pack('<8sQQQQQI', b'CKTTS002', ARENA_BYTES,
            waveform_offset, 20, gaussian_offset, gaussian_count, 0))
        return path

    def test_seeded_replay_distinct_seed_and_bounds(self):
        package = self.package('native.cktts')
        hashes = []
        for index, seed in enumerate(('7', '7', '8')):
            wav = self.root / f'noise-{index}.wav'
            subprocess.run([str(self.binary), str(package), str(wav), seed],
                           check=True, capture_output=True)
            with wave.open(str(wav), 'rb') as stream:
                self.assertEqual((stream.getnchannels(),
                    stream.getsampwidth(), stream.getframerate(),
                    stream.getnframes()), (1, 2, 24000, 20))
                self.assertTrue(any(stream.readframes(20)))
            hashes.append(hashlib.sha256(wav.read_bytes()).hexdigest())
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[0], hashes[2])
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.native-wav-host.seeded-noise-v1',
            'name': 'native host seeded excitation package and WAV identity',
            'provider': 'audio_noise_counter_gaussian_f32',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'deterministic-replay-and-different-seed-control',
            'status': 'pass', 'gate': 'session', 'blocking': True,
            'max_diff': 0, 'tolerance': 0,
            'same_seed_wav_sha256': hashes[0],
            'different_seed_wav_sha256': hashes[2],
            'configuration': 'synthetic generated-entry stub; 20 PCM16 samples at 24 kHz',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_native_kokoro_wav_host'}))
        for seed in ('-1', '18446744073709551616', 'bad'):
            wav = self.root / 'rejected.wav'
            rejected = subprocess.run([str(self.binary), str(package),
                                       str(wav), seed], capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(wav.exists())
        bad = self.package('overrun.cktts', gaussian_offset=ARENA_BYTES - 4)
        wav = self.root / 'overrun.wav'
        rejected = subprocess.run([str(self.binary), str(bad), str(wav), '7'],
                                  capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(wav.exists())
        bad = self.package('wrong-size.cktts', gaussian_count=GAUSSIAN_COUNT - 1)
        rejected = subprocess.run([str(self.binary), str(bad), str(wav), '7'],
                                  capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(wav.exists())
        bad = self.package('overlap.cktts', waveform_offset=0)
        rejected = subprocess.run([str(self.binary), str(bad), str(wav), '7'],
                                  capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(wav.exists())


if __name__ == '__main__':
    unittest.main()
