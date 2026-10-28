"""Controlled same-work MPS kernel probes; no production patches."""
import argparse
from collections import defaultdict
import json
import sys
import time
import numpy as np
import bench
from profile_microbatch import OUT, inputs

def measure(fn,n=1):
    import torch
    torch.mps.synchronize();t=time.perf_counter();value=None
    for _ in range(n):value=fn()
    torch.mps.synchronize();return (time.perf_counter()-t)/n,value

def owl():
    import torch
    torch.set_num_threads(1);torch.manual_seed(71)
    # Actual traced attention dimensions: 2 images x 12 heads x 3601 tokens.
    q=torch.randn((24,3601,64),device='mps',dtype=torch.float16)*.02
    k=torch.randn_like(q);v=torch.randn_like(q)
    def run(b,phases=False):
        parts=defaultdict(float);result=[]
        for i in range(0,24,b*12):
            qs,ks,vs=q[i:i+b*12],k[i:i+b*12],v[i:i+b*12]
            if phases:
                t,a=measure(lambda:torch.bmm(qs,ks.transpose(1,2)));parts['qk_bmm']+=t
                t,w=measure(lambda:torch.softmax(a,dim=-1));parts['softmax']+=t
                t,out=measure(lambda:torch.bmm(w,vs));parts['av_bmm']+=t
            else:
                a=torch.bmm(qs,ks.transpose(1,2));w=torch.softmax(a,dim=-1);out=torch.bmm(w,vs)
            result.append(out)
        return torch.cat(result),dict(parts)
    with torch.inference_mode():
        one=run(1)[0];two=run(2)[0];assert torch.equal(one,two)
        rows=[]
        for b in [1,2,2,1,2,1,1,2]*2:
            seconds,_=measure(lambda:run(b),3);_,parts=run(b,True)
            rows.append({'batch':b,'two_image_seconds':seconds,**parts})
    bench.dump(OUT/'owl-kernels.json',{'source':'synthetic values, actual traced attention shapes and dtype','shape':[24,3601,64],'outputs_exact':True,'rows':rows})
    print('owl kernels complete',flush=True)

def qwen():
    import torch
    import torch.nn.functional as F
    from action_search import ActionEngine
    from microbatch_qwen import prepare_inputs
    import transformers.models.qwen3_vl.modeling_qwen3_vl as module
    torch.set_num_threads(1)
    data=inputs('qwen')[0];ext=ActionEngine().model();net=ext._model
    batch=prepare_inputs(ext,data);prepared={k:v.to('mps') for k,v in batch.items()}
    patch=net.model.visual.patch_embed;pixels=prepared['pixel_values_videos'].to(patch.proj.weight.dtype)
    # Capture the real first-layer visual RoPE input, stop before the remainder.
    captured={};original=module.rotate_half
    class Captured(Exception):pass
    def capture(x):
        captured['x']=x.detach();raise Captured()
    module.rotate_half=capture
    try:
        with torch.inference_mode():net(**prepared)
    except Captured:pass
    finally:module.rotate_half=original
    assert captured
    x=captured['x'];n=len(pixels)//2;assert len(x)==len(pixels)==7168
    conv=lambda a:patch(a)
    # Kernel covers each entire input patch and stride == kernel: exactly the
    # same affine map algebraically. This is a local operator probe, not a
    # production change or a full retrieval-quality validation.
    linear=lambda a:F.linear(a,patch.proj.weight.reshape(patch.embed_dim,-1),patch.proj.bias)
    def apply(fn,a,b):return torch.cat([fn(a[i:i+b*n]) for i in range(0,len(a),b*n)])
    with torch.inference_mode():
        native=apply(conv,pixels,1);batched=apply(conv,pixels,2);alternative=apply(linear,pixels,1)
        assert torch.equal(native,batched)
        dif=(native.float()-alternative.float()).abs()
        agreement={'max_abs':float(dif.max()),'mean_abs':float(dif.mean()),'relative_l2':float(torch.linalg.vector_norm(native.float()-alternative.float())/torch.linalg.vector_norm(native.float()))}
        rows=[]
        for b in [1,2,2,1,2,1,1,2]:
            ct,_=measure(lambda:apply(conv,pixels,b))
            lt,_=measure(lambda:apply(linear,pixels,b),3)
            rt,_=measure(lambda:apply(original,x,b),10)
            nt,_=measure(lambda:apply(lambda y:-y[...,32:],x,b),20)
            rows.append({'batch':b,'conv3d_seconds':ct,'linear_probe_seconds':lt,'rotate_half_seconds':rt,'neg_slice_seconds':nt})
            print('qwen kernels',b,rows[-1],flush=True)
    bench.dump(OUT/'qwen-kernels.json',{'source':'actual sampled video tensors and loaded FP16 model weights','pixels_shape':list(pixels.shape),'rotary_shape':list(x.shape),'rotary_stride':list(x.stride()),'neg_input_stride':list(x[...,32:].stride()),'conv_batches_exact':True,'linear_probe_difference':agreement,'rows':rows,'limitation':'Operator probe only. Linear replacement is not deployed and has not passed full embedding/retrieval regression.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=['owl','qwen']);a=p.parse_args()
    owl() if a.model=='owl' else qwen()
