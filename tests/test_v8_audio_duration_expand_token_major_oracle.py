"""Independent numerical and bounds controls for token-major duration expansion."""
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FP = ctypes.POINTER(ctypes.c_float)
IP = ctypes.POINTER(ctypes.c_int32)


class TokenMajorDurationExpandOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / 'expand.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            '-pedantic', '-shared', '-fPIC', '-I', str(ROOT / 'include'),
            str(ROOT / 'src/kernels/audio_duration_expand_token_major.c'),
            '-o', str(library)], check=True, capture_output=True, text=True)
        cls.fn = ctypes.CDLL(str(library)).audio_duration_expand_token_major_f32
        cls.fn.argtypes = [FP, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, IP, ctypes.c_size_t,
            FP, ctypes.c_size_t, ctypes.c_size_t]
        cls.fn.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def call(self, source, durations, output, channels, frames=None, **override):
        args = dict(input_elements=source.size, channels=channels,
            phoneme_count=len(durations), input_stride=source.shape[1],
            expanded_frames=int(np.sum(durations)) if frames is None else frames,
            output_elements=output.size, output_stride=output.shape[1])
        args.update(override)
        return self.fn(source.ctypes.data_as(FP), args['input_elements'],
            args['channels'], args['phoneme_count'], args['input_stride'],
            durations.ctypes.data_as(IP), args['expanded_frames'],
            output.ctypes.data_as(FP), args['output_elements'],
            args['output_stride'])

    def test_numpy_exact_copy_tails_strides_and_repeated_calls(self):
        for tokens, channels, durations in ((1,1,[1]),(3,5,[2,1,3]),
                                           (4,7,[1,2,1,2]),(36,640,[1]*36)):
            with self.subTest(shape=(tokens,channels)):
                source=np.full((tokens,channels+3),-77.,np.float32)
                source[:,:channels]=(np.arange(tokens*channels,dtype=np.float32)
                                     .reshape(tokens,channels)-17)/np.float32(13)
                counts=np.asarray(durations,np.int32)
                frames=int(counts.sum())
                output=np.full((channels,frames+5),-99.,np.float32)
                expected=np.repeat(source[:,:channels],counts,axis=0).T.copy()
                for _ in range(3):
                    self.assertEqual(self.call(source,counts,output,channels),0)
                    np.testing.assert_array_equal(output[:,:frames].view(np.uint32),
                                                  expected.view(np.uint32))
                    self.assertTrue(np.all(output[:,frames:]==-99.))
                    self.assertTrue(np.all(source[:,channels:]==-77.))
                if tokens>1:
                    counts[0]+=1; counts[-1]-=1
                    if counts[-1]>0:
                        self.assertEqual(self.call(source,counts,output,channels),0)
                        np.testing.assert_array_equal(output[:,:frames],
                            np.repeat(source[:,:channels],counts,axis=0).T)

    def test_pinned_kokoro_duration_feature_fixture(self):
        fixture=ROOT/'tests/fixtures/tts/duration_two_stream_reference.npz'
        meta=json.loads(fixture.with_suffix('.json').read_text())
        with np.load(fixture) as archive:
            channel_major=archive['duration_features'][:,:36]
            expected=archive['duration_expected']
        source=np.full((36,643),-77.,np.float32)
        source[:,:640]=channel_major.T
        durations=np.asarray(meta['durations'],np.int32)
        output=np.full((640,128),-99.,np.float32)
        self.assertEqual(self.call(source,durations,output,640,frames=103),0)
        np.testing.assert_array_equal(output[:,:103].view(np.uint32),
                                      expected.view(np.uint32))
        self.assertTrue(np.all(output[:,103:]==-99.))
        print('CKE_NUMERICAL_CASE '+json.dumps({
            'case_id':'audio.duration_expand.token_major.pinned-kokoro-v1',
            'name':'token-major checked duration expansion',
            'provider':'audio_duration_expand_token_major_f32',
            'dtype':'fp32','direction':'inference','oracle':'pinned-pytorch',
            'backend_version':'2.8.0+cpu','max_diff':0.0,'tolerance':0.0,
            'status':'pass','configuration':'tokens=36; channels=640; frames=103',
            'reproduction_command':'python3 -m unittest '+self.id()}))

    def test_rejection_preserves_all_output(self):
        source=np.arange(2*7,dtype=np.float32).reshape(2,7)
        durations=np.asarray([2,1],np.int32)
        output=np.full((5,6),-99.,np.float32)
        cases=(({},dict(input_elements=11),-2),
               ({},dict(output_elements=26),-2),
               ({},dict(input_stride=4),-1),
               ({},dict(output_stride=2),-1),
               ({},dict(phoneme_count=0),-1),
               ({},dict(input_stride=ctypes.c_size_t(-1).value),-3),
               ({'durations':[0,3]}, {},-1),
               ({'durations':[4,1]}, {},-2),
               ({'durations':[1,1]}, {},-1))
        for changed,override,status in cases:
            with self.subTest(changed=changed,override=override):
                values=np.asarray(changed.get('durations',[2,1]),np.int32)
                self.assertEqual(self.call(source,values,output,5,frames=3,**override),status)
                self.assertTrue(np.all(output==-99.))

    def test_live_pytorch_repeat_interleave(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f'live PyTorch oracle unavailable: {exc}')
        source=np.arange(3*7,dtype=np.float32).reshape(3,7)
        durations=np.asarray([2,1,3],np.int32)
        output=np.full((5,9),-99.,np.float32)
        expected=torch.repeat_interleave(torch.from_numpy(source[:,:5].copy()),
            torch.from_numpy(durations.astype(np.int64)),dim=0).T.numpy()
        self.assertEqual(self.call(source,durations,output,5),0)
        np.testing.assert_array_equal(output[:,:6],expected)


if __name__=='__main__':
    unittest.main()
