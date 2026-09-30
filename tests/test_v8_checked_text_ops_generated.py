"""Model-neutral Conv1D -> transpose -> LeakyReLU through normal v8 codegen."""
import ctypes
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph,build_ir_v8


class CheckedTextOpsGeneratedTest(unittest.TestCase):
    def test_checkpoint_contract_uses_selected_unambiguous_provider(self):
        template={'semantic_checkpoints':{
            'schema':'cke.semantic_checkpoint_contract','schema_version':1,
            'exports':{'fixture':{'section':'header','template_op_id':'copy',
                'op':'audio_duration_expand','checkpoints':[{
                    'id':'synthetic.output','producer':'copy','tensor':'out',
                    'logical_layout':'channel_major','axis_names':['channel','frame'],
                    'storage_dtype':'fp32'}]}}}}
        valid='audio_duration_expand_checked_strided_copy_fp32'
        other='audio_duration_expand_token_major_checked_strided_copy_fp32'
        def resolve(ids):
            arranged=[{'kernel':'selected','op':'audio_duration_expand',
                'template_op_id':'copy','section':'header','layer':-1}]
            registry={'kernels':[{'id':'selected','impl':{'function':'selected_copy'},
                'numerical_capabilities':[{'contract_id':contract,
                    'function':'selected_copy','status':'validated',
                    'explicit_selector':True,'phases':['prefill']}
                    for contract in ids]}]}
            build_ir_v8._attach_semantic_checkpoints(template,arranged,registry)
            return arranged[0]['semantic_checkpoints'][0]['resolved_contract_id']
        self.assertEqual(resolve([valid]),valid)
        self.assertEqual(resolve([valid,other]),'unresolved')
        self.assertEqual(resolve(['not-a-registered-contract']),'unresolved')

    def compile(self, bad_weight_elements=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        weight=np.arange(12,dtype=np.float32).reshape(2,2,3)/np.float32(10)
        bias=np.array([.25,-.5],dtype=np.float32)
        payload=weight.tobytes()+bias.tobytes()
        entries=[]
        for name,offset,data,shape in (
            ('fixture.weight',0,weight,[2,2,3]),
            ('fixture.bias',weight.nbytes,bias,[2])):
            entries.append({'name':name,'dtype':'fp32','shape':shape,
                'file_offset':offset,'size':data.nbytes,
                'sha256':hashlib.sha256(data.tobytes()).hexdigest()})
        conv={'id':'conv','op':'audio_conv1d_checked',
            'kernel':'audio_conv1d_checked_channel_major_f32','returns_status':True,
            'weight_refs':{'weight':'fixture.weight','bias':'fixture.bias'},
            'params':{'I':2,'O':2,'T':3,'K':3,'U':3,'Q':6,
                'call_constants':{'conv_input_elements':8,'conv_input_stride':4,
                    'conv_weight_elements':13 if bad_weight_elements else 12,
                    'conv_bias_elements':2,'conv_output_elements':8,
                    'conv_output_stride':4,'conv_scratch_elements':6,
                    'conv_input_channels':2,'conv_output_channels':2,
                    'conv_input_frames':3,'conv_kernel_size':3,
                    'conv_stride':1,'conv_padding':1,'conv_output_frames':3}},
            'graph_slots':{'inputs':{'input':'external:signal'},
                           'outputs':{'output':'convolved'}}}
        trans={'id':'transpose','op':'transpose_strided_checked',
            'kernel':'transpose_strided_f32_checked','returns_status':True,
            'params':{'R':2,'C':3,'call_constants':{
                'transpose_input_elements':8,'transpose_input_stride':4,
                'transpose_output_elements':6,'transpose_output_stride':2,
                'transpose_rows':2,'transpose_columns':3}},
            'graph_slots':{'inputs':{'input':'convolved'},
                           'outputs':{'output':'token_major'}}}
        act={'id':'activate','op':'leaky_relu_strided_checked',
            'kernel':'leaky_relu_strided_f32_checked','returns_status':True,
            'params':{'R':3,'C':2,'negative_slope':.2,'call_constants':{
                'leaky_input_elements':6,'leaky_input_stride':2,
                'leaky_output_elements':6,'leaky_output_stride':2,
                'leaky_rows':3,'leaky_columns':2}},
            'graph_slots':{'inputs':{'input':'token_major'},
                           'outputs':{'output':'activated'}}}
        circuit={'version':3,'name':'synthetic_checked_text_ops',
            'family':'bounded_graph','checked_native_entry':True,
            'contract':{'runtime_invariants':{'inference_only':True,
                'production_kernel_heap_allocation':False}},
            'activation_buffers':{name:{'shape':shape} for name,shape in (
                ('signal',[2,4]),('convolved',[2,4]),
                ('token_major',[3,2]),('activated',[3,2]))},
            'activation_bindings':{name:name for name in
                ('signal','convolved','token_major','activated')},
            'native_entry':{'function':'ck_synthetic_checked_text_ops',
                'params':[{'c_type':'uint8_t *','name':'arena'},
                          {'c_type':'size_t','name':'arena_bytes'}],
                'arena':{'pointer':'arena','bytes':'arena_bytes'}},
            'sequence':['component'], 'block_types':{'component':{
                'sequence':['header','body','footer'],'header':[conv],
                'body':{'type':'dense','ops':[trans]},'footer':[act]}}}
        path=root/'circuit.json';path.write_text(json.dumps(circuit))
        source={'config':{'model':circuit['name'],'arch':circuit['name'],
            'num_layers':1,'embed_dim':2,'num_heads':1,'num_kv_heads':1,
            'head_dim':2,'intermediate_size':2,'context_length':3,
            'max_seq_len':3,'vocab_size':1},'entries':entries,
            'quant_summary':{},'template':circuit}
        return compile_native_graph(root,source,path),weight,bias,payload

    def test_generated_composition_and_failure_propagation(self):
        (layout,calls,_library,loaded,fn),weight,bias,payload=self.compile()
        self.assertEqual(calls['errors'],[])
        self.assertEqual([op['function'] for op in calls['operations']],
            ['audio_conv1d_checked_channel_major_f32',
             'transpose_strided_f32_checked','leaky_relu_strided_f32_checked'])
        size=layout['memory']['arena']['total_size'];raw=(ctypes.c_uint8*(size+63))()
        arena=(ctypes.c_uint8*size).from_buffer(raw,(-ctypes.addressof(raw))&63)
        weights={e['name']:e for e in layout['memory']['weights']['entries']}
        for name,start,end in (('fixture.weight',0,weight.nbytes),
                               ('fixture.bias',weight.nbytes,len(payload))):
            entry=weights[name];arena[entry['abs_offset']:entry['abs_offset']+end-start]=payload[start:end]
        buffers={e['name']:e for e in layout['memory']['activations']['buffers']}
        def view(name,shape):return np.ndarray(shape,np.float32,buffer=arena,
            offset=buffers[name]['abs_offset'])
        signal=view('signal',(2,4));conv=view('convolved',(2,4));result=view('activated',(3,2))
        for values in ([[1,2,3],[4,5,6]], [[0,-1,2],[1,0,-2]]):
            signal[:]=-91.;signal[:,:3]=values;conv[:]=-77.;result[:]=-88.
            self.assertEqual(fn(arena,len(arena)),0)
            expected=np.empty((2,3),np.float32)
            for oc in range(2):
                for frame in range(3):
                    expected[oc,frame]=bias[oc]+sum(
                        float(signal[ic,frame+tap-1])*float(weight[oc,ic,tap])
                        for ic in range(2) for tap in range(3)
                        if 0<=frame+tap-1<3)
            oracle=np.where(expected.T>=0,expected.T,expected.T*np.float32(.2))
            np.testing.assert_allclose(result,oracle,rtol=0,atol=1e-6)
            self.assertTrue(np.all(conv[:,3:]==-77.))
        signal[0,1]=np.nan;conv[:]=-77.;result[:]=-88.
        self.assertNotEqual(fn(arena,len(arena)),0)
        self.assertTrue(np.all(conv==-77.));self.assertTrue(np.all(result==-88.))

    def test_wrong_logical_weight_geometry_rejected_before_codegen(self):
        with self.assertRaisesRegex(RuntimeError,'HARD CALL CONSTANT FAULT'):
            self.compile(bad_weight_elements=True)


if __name__=='__main__':unittest.main()
