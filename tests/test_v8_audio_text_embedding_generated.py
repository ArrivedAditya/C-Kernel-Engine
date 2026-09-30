"""Small model-neutral checked embedding through normal v8 lowering and codegen."""
import ctypes
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph


def circuit(tokens=3, vocabulary=7, channels=5):
    stride = tokens + 4
    return {
        'version': 3, 'name': 'synthetic_checked_embedding',
        'family': 'bounded_graph', 'checked_native_entry': True,
        'contract': {'runtime_invariants': {'inference_only': True,
            'production_kernel_heap_allocation': False}},
        'activation_buffers': {'ids': {'shape': [tokens]},
            'out': {'shape': [channels, stride]}},
        'activation_bindings': {'ids': 'ids', 'out': 'out'},
        'native_entry': {'function': 'ck_checked_embedding_fixture',
            'params': [{'c_type': 'uint8_t *', 'name': 'arena'},
                       {'c_type': 'size_t', 'name': 'arena_bytes'}],
            'arena': {'pointer': 'arena', 'bytes': 'arena_bytes'}},
        'runtime_constants': {}, 'sequence': ['embedding_block'],
        'block_types': {'embedding_block': {
            'sequence': ['header', 'body', 'footer'],
            'header': [{'id': 'lookup', 'op': 'embedding_lookup_checked',
                'kernel': 'audio_text_embedding_channel_major_f32',
                'returns_status': True, 'weight_refs': {'table': 'table'},
                'params': {'T': tokens, 'V': vocabulary, 'C': channels,
                    'T_capacity': stride, 'vocabulary': vocabulary,
                    'channels': channels, 'call_constants': {
                        'id_elements': tokens, 'table_elements': vocabulary*channels,
                        'vocabulary': vocabulary, 'channels': channels,
                        'table_stride': channels, 'output_elements': channels*stride,
                        'tokens': tokens, 'output_stride': stride}},
                'graph_slots': {'inputs': {'ids': 'external:ids'},
                    'outputs': {'output': 'out'}}}],
            'body': {'type': 'dense', 'ops': []}, 'footer': []}},
    }


class GeneratedEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        template = circuit()
        path = root/'circuit.json'
        path.write_text(json.dumps(template))
        cls.table = np.arange(7*5,dtype=np.float32).reshape(7,5)
        payload = cls.table.tobytes()
        source = {'config': {'model':template['name'], 'arch':template['name'],
            'num_layers':1,'embed_dim':5,'num_heads':1,'num_kv_heads':1,
            'head_dim':5,'intermediate_size':10,'context_length':3,
            'max_seq_len':3,'vocab_size':7, 'T':3,'C':5,
            'activation_buffer_dtypes': {'ids':'i32'}},
            'entries':[{'name':'table','dtype':'fp32','shape':[7,5],
                'file_offset':0,'size':len(payload),
                'sha256':hashlib.sha256(payload).hexdigest()}],
            'quant_summary':{},'template':template}
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = compile_native_graph(
            root,source,path)
        cls.activations = {item['name']: item for item in
            cls.layout['memory']['activations']['buffers']}
        cls.weights = {item['name']: item for item in
            cls.layout['memory']['weights']['entries']}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def arena(self):
        size=self.layout['memory']['arena']['total_size']
        raw=(ctypes.c_uint8*(size+63))()
        arena=(ctypes.c_uint8*size).from_buffer(raw,(-ctypes.addressof(raw))&63)
        item=self.weights['table']
        arena[item['abs_offset']:item['abs_offset']+item['size']]=self.table.tobytes()
        ids=np.ndarray((3,),np.int32,buffer=arena,
            offset=self.activations['ids']['abs_offset'])
        ids[:]=[6,1,0]
        output=np.ndarray((5,7),np.float32,buffer=arena,
            offset=self.activations['out']['abs_offset'])
        output[:]=-91.
        return arena,ids,output

    def test_normal_lowering_and_repeated_execution(self):
        self.assertEqual(self.calls['errors'],[])
        self.assertEqual([op['function'] for op in self.calls['operations']],
                         ['audio_text_embedding_channel_major_f32'])
        arena,ids,output=self.arena()
        for values in ([6,1,0],[0,6,1]):
            ids[:]=values
            self.assertEqual(self.fn(arena,len(arena)),0)
            np.testing.assert_array_equal(output[:,:3],self.table[ids].T)
            self.assertTrue(np.all(output[:,3:]==-91.))

    def test_rejected_producer_preserves_output(self):
        arena,ids,output=self.arena()
        ids[2]=7
        self.assertNotEqual(self.fn(arena,len(arena)),0)
        self.assertTrue(np.all(output==-91.))
        ids[2]=0
        self.assertNotEqual(self.fn(arena,len(arena)-1),0)
        self.assertTrue(np.all(output==-91.))


if __name__=='__main__': unittest.main()
