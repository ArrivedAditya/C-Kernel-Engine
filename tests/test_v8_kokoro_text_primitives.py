"""Independent pinned and adversarial checks for generic acoustic text providers."""
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/tts/kokoro_text_encoder_pinned.npz'
F = ctypes.POINTER(ctypes.c_float)
Z = ctypes.c_size_t
I = ctypes.c_int


class TextPrimitivesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / 'text_primitives.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            '-fPIC', '-shared', '-ffp-contract=off', '-I', str(ROOT / 'include'),
            str(ROOT / 'src/kernels/audio_conv1d_checked.c'),
            str(ROOT / 'src/kernels/strided_unary_checked.c'),
            '-lm', '-o', str(library)], check=True, capture_output=True, text=True)
        native = ctypes.CDLL(str(library))
        cls.conv = native.audio_conv1d_checked_channel_major_f32
        cls.conv.argtypes = [F,Z,Z,F,Z,F,Z,F,Z,Z,F,Z,Z,Z,Z,Z,Z,Z,Z]
        cls.conv.restype = I
        cls.transpose = native.transpose_strided_f32_checked
        cls.transpose.argtypes = [F,Z,Z,F,Z,Z,Z,Z]
        cls.transpose.restype = I
        cls.leaky = native.leaky_relu_strided_f32_checked
        cls.leaky.argtypes = [F,Z,Z,F,Z,Z,Z,Z,ctypes.c_float]
        cls.leaky.restype = I
        with np.load(FIXTURE) as archive:
            cls.reference = {name: archive[name].copy() for name in archive.files}

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    @staticmethod
    def ptr(array): return array.ctypes.data_as(F)

    def call_conv(self, source, weight, bias, target, scratch, dims):
        ic, oc, frames, kernel, stride, padding, out_frames = dims
        return self.conv(self.ptr(source), source.size, source.shape[1],
            self.ptr(weight), weight.size, self.ptr(bias), bias.size,
            self.ptr(target), target.size, target.shape[1],
            self.ptr(scratch), scratch.size, *dims)

    def test_pinned_effective_weight_convolution_and_repetition(self):
        ref = self.reference
        source = np.full((512,40), -17., np.float32)
        source[:,:36] = ref['embedding']
        target = np.full((512,40), -91., np.float32)
        scratch = np.empty((512,36), np.float32)
        dims = (512,512,36,5,1,2,36)
        for _ in range(2):
            self.assertEqual(self.call_conv(source,ref['conv0_weight'],
                ref['conv0_bias'],target,scratch,dims),0)
            expected = ref['conv0_output']
            actual = target[:,:36]
            error = np.abs(actual-expected)
            self.assertTrue(np.isfinite(actual).all())
            self.assertLessEqual(error.max(), 1.3e-5)
            self.assertTrue(np.all(target[:,36:]==-91.))
        worst = np.unravel_index(error.argmax(),error.shape)
        print('CKE_NUMERICAL_CASE '+json.dumps({'case_id':'audio.conv1d.text.effective-pinned-v1',
            'name':'checked Conv1D versus pinned effective-weight PyTorch output',
            'provider':'audio_conv1d_checked_channel_major_f32','dtype':'fp32',
            'direction':'inference','oracle':'pinned-pytorch','backend_version':'2.8.0+cpu',
            'status':'pass','max_diff':float(error.max()),'tolerance':1.3e-5,
            'configuration':'512x36; 512 outputs; kernel=5; pad=2; stride=1; physical_stride=40',
            'worst_index':list(map(int,worst)),
            'reproduction_command':'python3 -m unittest tests.test_v8_kokoro_text_primitives'}))

    def test_small_independent_convolution_and_rejection(self):
        rng = np.random.default_rng(42)
        source=np.full((3,9), -9., np.float32)
        source[:,:7]=rng.normal(size=(3,7)).astype(np.float32)
        weight=rng.normal(size=(4,3,3)).astype(np.float32)
        bias=rng.normal(size=4).astype(np.float32)
        target=np.full((4,10),-11.,np.float32)
        scratch=np.empty((4,7),np.float32)
        dims=(3,4,7,3,1,1,7)
        self.assertEqual(self.call_conv(source,weight,bias,target,scratch,dims),0)
        oracle=np.empty((4,7),np.float32)
        for oc in range(4):
            for t in range(7):
                terms=[float(source[ic,t+k-1])*float(weight[oc,ic,k])
                    for ic in range(3) for k in range(3) if 0<=t+k-1<7]
                oracle[oc,t]=float(bias[oc])+sum(terms)
        np.testing.assert_allclose(target[:,:7],oracle,rtol=2e-6,atol=2e-6)
        self.assertTrue(np.all(target[:,7:]==-11.))
        prior=target.copy()
        self.assertNotEqual(self.call_conv(source,weight,bias,target,
            scratch[:,:-1].copy(),dims),0)
        np.testing.assert_array_equal(target,prior)
        bad=source.copy();bad[1,4]=np.nan
        self.assertNotEqual(self.call_conv(bad,weight,bias,target,scratch,dims),0)
        np.testing.assert_array_equal(target,prior)
        self.assertNotEqual(self.call_conv(source,weight,bias,target,scratch,
            (3,4,7,3,1,2,7)),0)
        np.testing.assert_array_equal(target,prior)
        for bad_source,bad_weight,bad_bias,bad_target,bad_scratch,bad_dims in (
            (source[:,:6].copy(),weight,bias,target,scratch,dims),
            (source,weight[:,:,:2].copy(),bias,target,scratch,dims),
            (source,weight,bias[:3].copy(),target,scratch,dims),
            (source,weight,bias,np.full((4,6),-11.,np.float32),scratch,dims),
            (source,weight,bias,target,scratch[:3].copy(),dims),
            (source,weight,bias,target,scratch,(3,4,7,3,0,1,7)),
            (source,weight,bias,target,scratch,(3,4,7,3,1,1,8)),
            (source,weight,bias,target,scratch,(2**63,4,7,3,1,1,7))):
            checked=bad_target.copy()
            self.assertNotEqual(self.call_conv(bad_source,bad_weight,bad_bias,
                bad_target,bad_scratch,bad_dims),0)
            np.testing.assert_array_equal(bad_target,checked)
        bad_weight=weight.copy();bad_weight[2,1,0]=np.inf
        self.assertNotEqual(self.call_conv(source,bad_weight,bias,target,
            scratch,dims),0)
        np.testing.assert_array_equal(target,prior)

    def test_strided_transpose_and_leaky_relu(self):
        source=np.full((3,7), -99.,np.float32)
        source[:,:5]=np.array([[1,-2,3,-4,5],[6,-7,8,-9,10],[11,-12,13,-14,15]],np.float32)
        trans=np.full((5,4),-88.,np.float32)
        self.assertEqual(self.transpose(self.ptr(source),source.size,7,
            self.ptr(trans),trans.size,4,3,5),0)
        np.testing.assert_array_equal(trans[:,:3],source[:,:5].T)
        self.assertTrue(np.all(trans[:,3:]==-88.))
        result=np.full((5,4),-77.,np.float32)
        self.assertEqual(self.leaky(self.ptr(trans),trans.size,4,
            self.ptr(result),result.size,4,5,3,ctypes.c_float(0.2)),0)
        np.testing.assert_allclose(result[:,:3],np.where(trans[:,:3]>=0,
            trans[:,:3],trans[:,:3]*np.float32(.2)),rtol=0,atol=0)
        self.assertTrue(np.all(result[:,3:]==-77.))
        prior=result.copy(); bad=trans.copy();bad[2,2]=np.inf
        self.assertNotEqual(self.leaky(self.ptr(bad),bad.size,4,
            self.ptr(result),result.size,4,5,3,ctypes.c_float(.2)),0)
        np.testing.assert_array_equal(result,prior)
        self.assertNotEqual(self.transpose(self.ptr(source),18,7,
            self.ptr(trans),trans.size,4,3,5),0)


if __name__ == '__main__': unittest.main()
