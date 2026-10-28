"""Isolated synchronized stage/module probes and real MPS profiling statistics.
Profiler runs are diagnostic only; their wall time is not throughput evidence.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import bench
OUT=Path(os.environ.get('MICROPROFILE_OUT',bench.OUT/'microbatch-profile-20261028')).resolve()

class Timers:
    def __init__(self):self.rows=defaultdict(lambda:{'seconds':0.,'calls':0})
    @contextmanager
    def measure(self,name):
        import torch
        torch.mps.synchronize();t=time.perf_counter()
        try:yield
        finally:
            torch.mps.synchronize();r=self.rows[name];r['seconds']+=time.perf_counter()-t;r['calls']+=1

def inputs(model):
    import torch,numpy as np,cv2
    from action_search import ActionEngine
    from types import SimpleNamespace
    d=json.loads((bench.OUT/'design.json').read_text());pairs=[]
    for mid in [1,40]:
        m=next(m for m in d['media'] if m['id']==mid);assert bench.sha(m['absolute_path'])==m['sha256']
        if model=='owl':
            cap=cv2.VideoCapture(m['absolute_path']);frames=[]
            for ts in [0.,.5]:
                cap.set(cv2.CAP_PROP_POS_MSEC,ts*1000);ok,f=cap.read();assert ok
                frames.append(torch.from_numpy(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)).permute(2,0,1))
            cap.release();pairs.append(torch.stack(frames))
        else:
            class Capture:
                def extract_video_segment_features(self,f):self.frames=f;return SimpleNamespace(vectors=np.zeros((1,2048),dtype='float32'))
            capture=Capture();engine=ActionEngine();engine.model=lambda:capture;frames=[]
            for ts in [0.,4.]:
                engine.encode_segment(m,{'ts':ts,'te':min(ts+8,m['duration'])});frames.append(capture.frames)
            pairs.append(frames)
    return pairs

def build(model):
    import torch
    if model=='owl':
        import frame_acceleration
        frame_acceleration.install_owl()
        import wise.feature.transformers_owlv2 as o
        ext=o.TransformersOWLv2FeatureExtractor(bench.OWL,device='mps',config=o.TransformersOWLv2FeatureExtractor.Config(device='mps',compile=False,warmup=False))
        net=ext.model.model
        @torch.inference_mode()
        def encode(frames,t):
            sizes=[(x.shape[2],x.shape[1]) for x in frames]
            with t.measure('preprocess'):pixels=ext.processor(images=frames,return_tensors='pt')['pixel_values']
            with t.measure('to_mps'):pixels=pixels.to('mps')
            with t.measure('model'):scores,embeds,boxes=o.get_object_features(net,pixels)
            with t.measure('float_and_sort'):scores,embeds,boxes=o.sort_by_objectness(scores.float(),embeds.float(),boxes.float())
            with t.measure('to_cpu'):scores,embeds,boxes=[x.cpu().numpy() for x in [scores,embeds,boxes]]
            with t.measure('construct_features'):features=ext.construct_features(embeds,scores,boxes,sizes)
            return [f.vectors for f in features]
        groups=[]
        for name,m in net.named_modules():
            if name.startswith('owlv2.vision_model') and name.endswith('.self_attn'):groups.append((name,m,'vision_attention'))
            elif name.startswith('owlv2.vision_model') and name.endswith('.mlp'):groups.append((name,m,'vision_mlp'))
        info={'dtype':str(next(net.parameters()).dtype),'attention_class':type(net.owlv2.vision_model.encoder.layers[0].self_attn).__name__}
    else:
        from action_search import ActionEngine
        from microbatch_qwen import prepare_inputs
        ext=ActionEngine().model();net=ext._model
        @torch.inference_mode()
        def encode(frames,t):
            with t.measure('preprocess'):cpu=prepare_inputs(ext,frames)
            with t.measure('to_mps'):prepared={k:v.to(ext.DEVICE) for k,v in cpu.items()}
            with t.measure('model'):outputs=net(**prepared)
            with t.measure('pool_normalize'):x=ext._last_token_pool(outputs.last_hidden_state,prepared['attention_mask']).float();x=x/x.norm(dim=-1,keepdim=True)
            with t.measure('to_cpu'):value=x.cpu().numpy()
            return [v[None] for v in value]
        groups=[('model.visual',net.model.visual,'visual_total'),('model.language_model',net.model.language_model,'language_total')]
        for name,m in net.named_modules():
            if name.startswith('model.visual.blocks.') and name.endswith('.attn'):groups.append((name,m,'visual_attention'))
            elif name.startswith('model.visual.blocks.') and name.endswith('.mlp'):groups.append((name,m,'visual_mlp'))
            elif name.startswith('model.language_model.layers.') and name.endswith('.self_attn'):groups.append((name,m,'language_attention'))
            elif name.startswith('model.language_model.layers.') and name.endswith('.mlp'):groups.append((name,m,'language_mlp'))
        info={'dtype':str(next(net.parameters()).dtype),'vision_attention':net.model.visual.config._attn_implementation,'text_attention':net.model.language_model.config._attn_implementation}
    return ext,net,encode,groups,info

def worker(model,mode,batch):
    import torch,numpy as np
    from resource_metrics import ResourceSampler
    torch.set_num_threads(1);torch.mps.set_per_process_memory_fraction(.55)
    data=inputs(model);ext,net,encode,groups,info=build(model)
    result={'model':model,'mode':mode,'runtime':info,'stages':[],'modules':[]}
    print('runtime',info,flush=True)
    def run_pair(pair,b,t):
        out=[]
        for i in range(0,2,b):out+=encode(pair[i:i+b],t)
        return out
    if mode=='mps':
        # Only one fixed spatial shape and four logical input items total.
        # Initialization/copy and first-use compilation remain in stats; do not
        # sum graph timings into an end-to-end time or treat this as throughput.
        for _ in range(2):run_pair(data[0],batch,Timers())
        torch.mps.synchronize();result['batch']=batch;result['logical_items']=4
        bench.dump(OUT/f'{model}-mps-b{batch}.json',result);return
    # Warm both batch shapes before the interleaved, no-module-hook measurements.
    reference={}
    for b in [1,2]:
        for i,pair in enumerate(data):
            value=run_pair(pair,b,Timers())
            if b==1:reference[i]=value
            else:assert all(np.array_equal(x,y) for x,y in zip(reference[i],value))
    with ResourceSampler() as resources:
        for turn,b in enumerate([1,2,2,1,2,1,1,2]):
            t=Timers();start=time.perf_counter()
            for i,pair in enumerate(data):
                value=run_pair(pair,b,t)
                assert all(np.array_equal(x,y) for x,y in zip(reference[i],value))
            result['stages'].append({'turn':turn,'batch':b,'items':4,'wall_seconds':time.perf_counter()-start,'timers':dict(t.rows)})
            bench.dump(OUT/f'{model}-stages.json',result);print(model,'stage',turn,b,result['stages'][-1]['wall_seconds'],flush=True)
    result['resources']=resources.result
    # Separately instrument modules. Nested categories are inclusive and MUST NOT
    # be summed (e.g. visual_total already includes visual_attention and visual_mlp).
    active={'timers':None};originals=[]
    for name,module,label in groups:
        original=module.forward
        def wrapped(*a,_orig=original,_label=label,**kw):
            with active['timers'].measure(_label):return _orig(*a,**kw)
        module.forward=wrapped;originals.append((module,original))
    try:
        for turn,b in enumerate([1,2,2,1]):
            t=Timers();active['timers']=t
            for i,pair in enumerate(data):
                value=run_pair(pair,b,t)
                assert all(np.array_equal(x,y) for x,y in zip(reference[i],value))
            result['modules'].append({'turn':turn,'batch':b,'items':4,'timers':dict(t.rows)})
            bench.dump(OUT/f'{model}-stages.json',result);print(model,'modules',turn,b,flush=True)
    finally:
        for module,original in originals:module.forward=original
    # CPU dispatch trace: operator shapes/call counts, explicitly NOT GPU durations.
    result['cpu_dispatch']={}
    for b in [1,2]:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as prof:
            run_pair(data[0],b,Timers())
        rows=[{'op':e.key,'count':e.count,'input_shapes':e.input_shapes,'self_cpu_dispatch_us':e.self_cpu_time_total} for e in prof.key_averages(group_by_input_shape=True)]
        result['cpu_dispatch'][str(b)]=rows
        prof.export_chrome_trace(str(OUT/f'{model}-cpu-dispatch-b{b}.json'))
    result['status']='completed';bench.dump(OUT/f'{model}-stages.json',result)

def suite():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'production-before.json').exists(),'Use fresh MICROPROFILE_OUT'
    before=bench.fingerprint();bench.dump(OUT/'production-before.json',before)
    for model in ['owl','qwen']:
        for mode,b in [('stages',1),('mps',1),('mps',2)]:
            env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
            if mode=='mps':
                env['PYTORCH_MPS_LOG_PROFILE_INFO']='392'
                # Commit/wait after profiled operations to avoid shared-buffer
                # attribution where different ops receive the same GPU time.
                env['PYTORCH_MPS_TRACE_SIGNPOSTS']='4'
            name=f'{model}-{mode}'+(f'-b{b}' if mode=='mps' else '')
            with (OUT/f'{name}.log').open('w') as log:
                r=subprocess.run([sys.executable,'-u',__file__,'worker','--model',model,'--mode',mode,'--batch',str(b)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            print(name,'exit',r.returncode,flush=True)
            if r.returncode:raise RuntimeError(name)
    after=bench.fingerprint();bench.dump(OUT/'production-after.json',after);assert before==after

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['suite','worker']);p.add_argument('--model',choices=['owl','qwen']);p.add_argument('--mode',choices=['stages','mps']);p.add_argument('--batch',type=int,default=1);a=p.parse_args()
    suite() if a.command=='suite' else worker(a.model,a.mode,a.batch)
