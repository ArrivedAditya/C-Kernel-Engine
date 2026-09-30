"""Checked channel-major embedding against independent Kokoro/PyTorch capture."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FP = ctypes.POINTER(ctypes.c_float)
IP = ctypes.POINTER(ctypes.c_int32)
FIXTURE = ROOT / 'tests/fixtures/tts/kokoro_text_embedding_pinned.npz'


class AudioTextEmbeddingOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / 'text_embedding.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            '-pedantic', '-shared', '-fPIC', '-I', str(ROOT / 'include'),
            str(ROOT / 'src/kernels/audio_text_embedding.c'), '-lm',
            '-o', str(library)], check=True, capture_output=True, text=True)
        cls.fn = ctypes.CDLL(str(library)).audio_text_embedding_channel_major_f32
        cls.fn.argtypes = [IP, ctypes.c_size_t, FP, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            FP, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        cls.fn.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def call(self, ids, table, output, channels=None, **override):
        args = dict(id_elements=ids.size, table_elements=table.size,
            vocabulary=table.shape[0], channels=table.shape[1] if channels is None else channels,
            table_stride=table.shape[1], output_elements=output.size,
            tokens=ids.size, output_stride=output.shape[1])
        args.update(override)
        return self.fn(ids.ctypes.data_as(IP), args['id_elements'],
            table.ctypes.data_as(FP), args['table_elements'],
            args['vocabulary'], args['channels'], args['table_stride'],
            output.ctypes.data_as(FP), args['output_elements'],
            args['tokens'], args['output_stride'])

    def test_pinned_kokoro_capture_exact(self):
        meta = json.loads(FIXTURE.with_suffix('.json').read_text())
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                         meta['fixture_sha256'])
        self.assertEqual(meta['torch'], '2.8.0+cpu')
        with np.load(FIXTURE) as archive:
            ids, table, expected = (archive[name].copy() for name in
                                    ('ids', 'table', 'expected'))
        for name, array in (('ids',ids),('table',table),('expected',expected)):
            self.assertEqual(hashlib.sha256(array.tobytes()).hexdigest(),
                             meta['arrays_sha256'][name])
        output = np.full((512, 40), -91., np.float32)
        self.assertEqual(self.call(ids, table, output), 0)
        np.testing.assert_array_equal(output[:,:36].view(np.uint32),
                                      expected.view(np.uint32))
        self.assertTrue(np.all(output[:,36:] == -91.))
        print('CKE_NUMERICAL_CASE '+json.dumps({
            'case_id':'audio.text_embedding.pinned-kokoro-v1',
            'name':'checked channel-major text embedding',
            'provider':'audio_text_embedding_channel_major_f32',
            'dtype':'fp32','direction':'inference','oracle':'pinned-pytorch',
            'backend_version':meta['torch'],'max_diff':0.0,'tolerance':0.0,
            'status':'pass','configuration':'tokens=36; vocabulary=178; channels=512',
            'reproduction_command':'python3 -m unittest '+self.id()}))

    def test_small_tails_padded_strides_and_repeated_calls(self):
        table = np.arange(7*8, dtype=np.float32).reshape(7,8)
        ids = np.asarray([6,1,0], np.int32)
        output = np.full((5,7), -91., np.float32)
        for values in ([6,1,0], [0,0,6], [1,6,1]):
            ids[:] = values
            self.assertEqual(self.call(ids,table,output,channels=5),0)
            np.testing.assert_array_equal(output[:,:3].view(np.uint32),
                                          table[ids,:5].T.copy().view(np.uint32))
            self.assertTrue(np.all(output[:,3:] == -91.))
        self.assertEqual(self.call(ids,table,output,channels=5,tokens=1),0)
        np.testing.assert_array_equal(output[:,0],table[ids[0],:5])

    def test_rejection_preserves_output(self):
        ids = np.asarray([1,2,3],np.int32)
        table = np.arange(7*8,dtype=np.float32).reshape(7,8)
        output = np.full((5,7),-91.,np.float32)
        huge = ctypes.c_size_t(-1).value
        cases = [({'id_elements':2},-2),({'table_elements':42},-2),
            ({'output_elements':30},-2),({'tokens':0},-1),
            ({'channels':0},-1),({'table_stride':4},-1),
            ({'output_stride':2},-1),({'vocabulary':huge},-3),
            ({'output_stride':huge},-3)]
        for override, status in cases:
            with self.subTest(override=override):
                self.assertEqual(self.call(ids,table,output,**({'channels':5}|override)),status)
                self.assertTrue(np.all(output == -91.))
        for bad_id in (-1,7):
            ids[2]=bad_id
            self.assertEqual(self.call(ids,table,output,channels=5),-1)
            self.assertTrue(np.all(output == -91.))
        ids[2]=3
        for bad in (np.nan,np.inf,-np.inf):
            table[2,4]=bad
            self.assertEqual(self.call(ids,table,output,channels=5),-1)
            self.assertTrue(np.all(output == -91.))
        table[2,4]=1.
        table[6,4]=np.nan  # Unselected rows do not affect valid requests.
        self.assertEqual(self.call(ids,table,output,channels=5),0)

    def test_live_pytorch(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f'live PyTorch unavailable: {exc}')
        ids=np.asarray([2,0,1,2],np.int32)
        table=np.arange(4*6,dtype=np.float32).reshape(4,6)
        output=np.full((6,7),-91.,np.float32)
        expected=torch.nn.functional.embedding(torch.from_numpy(ids.astype(np.int64)),
                                               torch.from_numpy(table)).T.numpy()
        self.assertEqual(self.call(ids,table,output),0)
        np.testing.assert_array_equal(output[:,:4],expected)


if __name__=='__main__':
    unittest.main()
