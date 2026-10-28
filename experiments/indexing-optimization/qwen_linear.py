"""Fresh Conv3d vs equivalent linear patch projection; isolated, batch=1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType
import bench
OUT=Path(os.environ.get('QWEN_LINEAR_OUT',bench.OUT/'qwen-linear-20261028')).resolve()

def install_linear(extractor):
    import torch
    import torch.nn.functional as F
    patch=extractor._model.model.visual.patch_embed
    conv=patch.proj
    kernel=(patch.temporal_patch_size,patch.patch_size,patch.patch_size)
    assert isinstance(conv,torch.nn.Conv3d)
    assert conv.kernel_size==conv.stride==kernel
    assert conv.padding==(0,0,0) and conv.dilation==(1,1,1) and conv.groups==1
    assert conv.in_channels==patch.in_channels and conv.out_channels==patch.embed_dim
    assert conv.weight.device.type=='mps' and conv.weight.dtype==torch.float16
    width=conv.in_channels*kernel[0]*kernel[1]*kernel[2]
    original=patch.forward
    def forward(self,hidden_states):
        assert hidden_states.ndim==2 and hidden_states.shape[1]==width
        return F.linear(hidden_states.to(dtype=self.proj.weight.dtype),self.proj.weight.reshape(self.embed_dim,width),self.proj.bias)
    patch.forward=MethodType(forward,patch)
    return original,{'class':type(patch).__name__,'kernel':list(kernel),'input_width':width,'output_width':patch.embed_dim,'dtype':str(conv.weight.dtype),'device':str(conv.weight.device)}

def worker(variant,part,repeat):
    # Production now scopes Linear in encode_segment. Disable that scope here
    # so the explicit Conv/Linear experimental variants remain independent.
    from contextlib import nullcontext
    from unittest.mock import patch
    with patch('action_search.linear_video_projection', lambda extractor: nullcontext()):
        return _worker(variant,part,repeat)

def _worker(variant,part,repeat):
    import numpy as np
    import torch
    import psutil
    from action_search import ActionEngine
    from resource_metrics import ResourceSampler
    design=json.loads((OUT/'design.json').read_text());media={m['id']:m for m in design['media']}
    selected=set(design['sample_media_ids'])
    segments=[s for s in design['segments'] if (s['media_id'] in selected)==(part=='sample')]
    report={'variant':variant,'part':part,'repeat':repeat,'segments':segments,'settings':design['settings'],'batch':1,'status':'running','measurements':[],'input_hashes':[]}
    path=OUT/f'{variant}-{part}-r{repeat}.json'
    assert not path.exists()
    engine=ActionEngine();t=time.perf_counter();ext=engine.model();report['load_seconds']=time.perf_counter()-t
    if variant=='linear':_,report['patch']=install_linear(ext)
    else:report['patch']={'class':type(ext._model.model.visual.patch_embed).__name__,'original_forward':True}
    p=ext._model.model.visual.patch_embed.proj
    report['projection_weight_sha256']=hashlib.sha256(p.weight.detach().cpu().numpy().tobytes()+p.bias.detach().cpu().numpy().tobytes()).hexdigest()
    assert all(v.dtype==torch.float16 for v in ext._model.parameters() if v.is_floating_point())
    first=segments[0];t=time.perf_counter();engine.encode_segment(media[first['media_id']],first);torch.mps.synchronize();report['warmup_seconds']=time.perf_counter()-t
    original=ext.extract_video_segment_features;captured={}
    def measured(frames):
        t=time.perf_counter();raw_hash=hashlib.sha256(frames.numpy().tobytes()).hexdigest();hash_seconds=time.perf_counter()-t
        torch.mps.synchronize();t=time.perf_counter();value=original(frames);torch.mps.synchronize()
        captured.update(encode_seconds=time.perf_counter()-t,hash_seconds=hash_seconds,raw_hash=raw_hash,raw_shape=list(frames.shape))
        return value
    ext.extract_video_segment_features=measured
    values=[]
    with ResourceSampler() as resources:
        for i,s in enumerate(segments):
            assert psutil.virtual_memory().available>2*1024**3,'Available RAM below 2GiB'
            t=time.perf_counter();v=engine.encode_segment(media[s['media_id']],s);elapsed=time.perf_counter()-t-captured['hash_seconds']
            assert np.isfinite(v).all() and abs(float(np.linalg.norm(v))-1)<1e-4
            values.append(v);report['input_hashes'].append(captured['raw_hash'])
            report['measurements'].append({'segment':s,'encode_seconds':captured['encode_seconds'],'decode_and_encode_seconds':elapsed,'decode_and_wrapper_seconds':elapsed-captured['encode_seconds'],'raw_shape':captured['raw_shape']})
            bench.dump(path,report)
            print(variant,part,repeat,f'{i+1}/{len(segments)}',round(captured['encode_seconds'],3),flush=True)
    report['resources']=resources.result
    for key in ['encode_seconds','decode_and_encode_seconds','decode_and_wrapper_seconds']:report[key]=sum(r[key] for r in report['measurements'])
    np.save(path.with_suffix('.npy'),np.stack(values));report['status']='completed';bench.dump(path,report)

def suite():
    from action_search import read_snapshot,SETTINGS
    import numpy as np
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'design.json').exists(),'Use fresh QWEN_LINEAR_OUT'
    before=bench.fingerprint();bench.dump(OUT/'production-before.json',before)
    meta,vectors,_=read_snapshot(Path(before['project']),complete=True)
    media=meta['media'];
    for m in media:
        path=Path(m['location'])/m['path']
        if not path.is_absolute():path=bench.ROOT/path
        m['absolute_path']=str(path.resolve());m['sha256']=bench.sha(path)
    source=json.loads((bench.OUT/'design.json').read_text())
    design={'project':before['project'],'media':media,'segments':meta['segments'],'sample_media_ids':source['sample_media_ids'],'settings':SETTINGS,'queries_source':'experiments/wise-upload-original/design.json','query_vectors_source':'experiments/wise-upload-original/current-full/query-vectors.npy'}
    bench.dump(OUT/'design.json',design);np.save(OUT/'production-reference.npy',vectors)
    env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
    runs=[('conv','sample',1),('linear','sample',1),('linear','sample',2),('conv','sample',2),('conv','rest',1),('linear','rest',1)]
    for variant,part,rep in runs:
        name=f'{variant}-{part}-r{rep}'
        with (OUT/(name+'.log')).open('w') as log:
            r=subprocess.run([sys.executable,'-u',__file__,'worker','--variant',variant,'--part',part,'--repeat',str(rep)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        print(name,'exit',r.returncode,flush=True)
        if r.returncode:raise RuntimeError(name)
    for m in media:assert bench.sha(m['absolute_path'])==m['sha256']
    after=bench.fingerprint();bench.dump(OUT/'production-after.json',after);assert before==after

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['suite','worker']);p.add_argument('--variant',choices=['conv','linear']);p.add_argument('--part',choices=['sample','rest']);p.add_argument('--repeat',type=int,default=1);a=p.parse_args()
    suite() if a.command=='suite' else worker(a.variant,a.part,a.repeat)
