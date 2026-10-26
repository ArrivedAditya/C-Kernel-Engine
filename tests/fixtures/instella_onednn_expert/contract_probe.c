#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ckernel_engine.h"
void gemm_nt_bf16_pytorch_onednn_3_12_brgemm_bf16_storage_workspace(const float *A,const void *B,const float *bias,float *C,int M,int N,int K,uint16_t *ib,size_t ibs,uint16_t *ob,size_t obs,uint16_t *bb,size_t bbs) {
    (void)bias;(void)ib;(void)ibs;(void)ob;(void)obs;(void)bb;(void)bbs;
    const uint16_t *w=(const uint16_t*)B;
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){float sum=0;for(int k=0;k<K;k++){uint32_t u=(uint32_t)w[n*K+k]<<16;float f;memcpy(&f,&u,4);sum+=A[m*K+k]*f;}C[m*N+n]=sum;}
}
void swiglu_forward_pytorch_bf16_storage(const float *input,float *output,int rows,int intermediate_dim){
 for(int r=0;r<rows;r++)for(int i=0;i<intermediate_dim;i++)output[r*intermediate_dim+i]=input[r*2*intermediate_dim+i]*input[r*2*intermediate_dim+intermediate_dim+i];
}
int main(void){
 enum {H=32,I=32,E=3,K=2,R=2,WORKSPACE=R*(12*H+16*I+8)+8*I};
 float x[R*H],routes[R*K]={0.375f,0.625f,-0.875f,1.125f};
 float guarded[2+R*H+2],*y=guarded+2;int ids[R*K]={2,0,1,2};
 uint16_t gate[E*I*H],up[E*I*H],down[E*H*I];
 for(int r=0;r<R;r++)for(int h=0;h<H;h++)x[r*H+h]=(float)(r+1);
 guarded[0]=guarded[1]=guarded[2+R*H]=guarded[3+R*H]=123456.0f;
 memset(gate,0,sizeof(gate));memset(up,0,sizeof(up));memset(down,0,sizeof(down));
 for(int e=0;e<E;e++)for(int i=0;i<I;i++) {
   gate[e*I*H+i*H+i]=0x3f80;up[e*I*H+i*H+i]=0x3f80;
   down[e*H*I+i*I+i]=(uint16_t)(0x3f80 + (e == 1 ? 0x80 : e == 2 ? 0xc0 : 0));
 }
 void *raw=malloc(WORKSPACE+64); uintptr_t p=((uintptr_t)raw+63)&~(uintptr_t)63;
 if(!raw)return 1;
 int rc=moe_swiglu_expert_forward_pytorch_onednn_3_12_bf16_storage_workspace(x,ids,routes,gate,up,down,y,R,H,I,E,K,(void*)p,WORKSPACE);
 if(rc) return 2;
 for(int j=0;j<R*H;j++) if(y[j] != (j<H ? 1.75f : 6.5f)) return 2;
 if(guarded[0]!=123456.0f||guarded[1]!=123456.0f||guarded[2+R*H]!=123456.0f||guarded[3+R*H]!=123456.0f)return 2;
 for (int j=0;j<R*H;j++) y[j]=12345.0f;
 rc=moe_swiglu_expert_forward_pytorch_onednn_3_12_bf16_storage_workspace(x,ids,routes,gate,up,down,y,R,H,I,E,K,(void*)p,WORKSPACE-1);
 if(rc!=-1) return 3;
 for(int j=0;j<R*H;j++)if(y[j]!=12345.0f)return 3;
 rc=moe_swiglu_expert_forward_pytorch_onednn_3_12_bf16_storage_workspace(x,ids,routes,gate,up,down,y,R,H,I,E,K,(void*)(p+1),WORKSPACE);
 if(rc!=-1)return 5;
 for(int j=0;j<R*H;j++)if(y[j]!=12345.0f)return 5;
 rc=moe_swiglu_expert_forward_pytorch_onednn_3_12_bf16_storage_workspace(x,ids,routes,gate,up,down,y,R,H-1,I,E,K,(void*)p,WORKSPACE);
 if(rc!=-1)return 6;
 for(int j=0;j<R*H;j++)if(y[j]!=12345.0f)return 6;
 ids[3]=E;
 rc=moe_swiglu_expert_forward_pytorch_onednn_3_12_bf16_storage_workspace(x,ids,routes,gate,up,down,y,R,H,I,E,K,(void*)p,WORKSPACE);
 if(rc!=-2)return 4;
 for(int j=0;j<R*H;j++)if(y[j]!=12345.0f)return 4;
 ids[3]=2;
 ids[1]=2;
 rc=moe_swiglu_expert_forward_pytorch_onednn_3_12_bf16_storage_workspace(x,ids,routes,gate,up,down,y,R,H,I,E,K,(void*)p,WORKSPACE);
 if(rc!=-2)return 8;
 for(int j=0;j<R*H;j++)if(y[j]!=12345.0f)return 8;
 ids[1]=0;
 routes[1]=NAN;
 rc=moe_swiglu_expert_forward_pytorch_onednn_3_12_bf16_storage_workspace(x,ids,routes,gate,up,down,y,R,H,I,E,K,(void*)p,WORKSPACE);
 if(rc!=-2)return 9;
 for(int j=0;j<R*H;j++)if(y[j]!=12345.0f)return 9;
 routes[1]=0.625f;
 rc=moe_swiglu_expert_forward_pytorch_onednn_3_12_bf16_storage_workspace(x,ids,routes,gate,up,down,y,R,H,I,E,K,(void*)p,WORKSPACE);
 if(rc)return 7;
 for(int j=0;j<R*H;j++)if(y[j] != (j<H ? 1.75f : 6.5f))return 7;
 puts("PASS: fractional/cancelling routes, all outputs and guards, rejection and recovery (stub arithmetic only)");
 free(raw);return 0;
}
