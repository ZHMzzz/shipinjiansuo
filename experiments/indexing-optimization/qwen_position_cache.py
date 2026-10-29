"""Isolated bounded position-output cache benchmark; production Linear stays on."""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType
import bench

OUT = Path(os.environ.get('QWEN_POSITION_OUT', bench.OUT/'qwen-position-20261028')).resolve()

class PositionCache:
    """Inference-only, per-model LRU. No approximate position computation."""
    def __init__(self, visual, limit=64*1024**2):
        self.visual, self.limit = visual, limit
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.misses = self.evictions = 0
        self.originals = {}
        self.grids = set()
        for name in ('fast_pos_embed_interpolate', 'rot_pos_emb'):
            original = getattr(visual, name)
            self.originals[name] = original
            def wrapper(module, grid, name=name, original=original):
                import torch
                if module.training or torch.is_grad_enabled():
                    return original(grid)
                shape = tuple(tuple(int(v) for v in row) for row in grid.detach().cpu().tolist())
                self.grids.add(shape)
                weight = module.pos_embed.weight
                freq = module.rotary_pos_emb.inv_freq
                key = (name, shape, str(weight.device), str(weight.dtype),
                       weight.data_ptr(), weight._version, freq.data_ptr(), freq._version,
                       str(freq.device), str(freq.dtype), module.spatial_merge_size)
                if key in self.entries:
                    self.hits += 1
                    self.entries.move_to_end(key)
                    return self.entries[key]
                self.misses += 1
                value = original(grid).detach()
                size = value.numel()*value.element_size()
                if size <= self.limit:
                    while self.entries and self.bytes + size > self.limit:
                        _, old = self.entries.popitem(last=False)
                        self.bytes -= old.numel()*old.element_size()
                        self.evictions += 1
                    self.entries[key] = value
                    self.bytes += size
                    self.peak_bytes = max(self.peak_bytes, self.bytes)
                return value
            setattr(visual, name, MethodType(wrapper, visual))

    def clear(self):
        self.entries.clear(); self.bytes = 0
        self.hits = self.misses = self.evictions = self.peak_bytes = 0
        self.grids.clear()

    def close(self):
        for name, original in self.originals.items():
            setattr(self.visual, name, original)
        self.clear()

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                    resident_bytes=self.bytes, peak_bytes=self.peak_bytes,
                    limit_bytes=self.limit, unique_grids=len(self.grids), entries=len(self.entries))


def worker(variant, part, repeat):
    import numpy as np
    import torch
    import psutil
    from action_search import ActionEngine
    from resource_metrics import ResourceSampler
    design=json.loads((OUT/'design.json').read_text())
    media={m['id']:m for m in design['media']}; selected=set(design['sample_media_ids'])
    segments=[s for s in design['segments'] if (s['media_id'] in selected)==(part=='sample')]
    path=OUT/f'{variant}-{part}-r{repeat}.json'; assert not path.exists()
    engine=ActionEngine(); start=time.perf_counter(); ext=engine.model()
    report=dict(variant=variant, part=part, repeat=repeat, segments=segments,
                load_seconds=time.perf_counter()-start, status='running', measurements=[], input_hashes=[])
    # Warm model only; then start caching from empty, including misses in timing.
    first=segments[0]; engine.encode_segment(media[first['media_id']], first)
    cache=PositionCache(ext._model.model.visual) if variant=='cached' else None
    original=ext.extract_video_segment_features; captured={}
    def measured(frames):
        t=time.perf_counter(); h=hashlib.sha256(frames.numpy().tobytes()).hexdigest(); hash_seconds=time.perf_counter()-t
        torch.mps.synchronize(); t=time.perf_counter(); result=original(frames); torch.mps.synchronize()
        captured.update(encode_seconds=time.perf_counter()-t, hash_seconds=hash_seconds, hash=h)
        return result
    ext.extract_video_segment_features=measured
    values=[]
    try:
        with ResourceSampler() as resources:
            for i,s in enumerate(segments):
                assert psutil.virtual_memory().available>2*1024**3
                t=time.perf_counter(); v=engine.encode_segment(media[s['media_id']],s)
                elapsed=time.perf_counter()-t-captured['hash_seconds']
                assert np.isfinite(v).all() and abs(float(np.linalg.norm(v))-1)<1e-4
                values.append(v);report['input_hashes'].append(captured['hash'])
                report['measurements'].append(dict(segment=s,encode_seconds=captured['encode_seconds'],decode_and_encode_seconds=elapsed))
                bench.dump(path, report)
                print(variant,part,repeat,f'{i+1}/{len(segments)}',round(captured['encode_seconds'],3),flush=True)
        report['resources']=resources.result
        report['cache']=cache.stats() if cache else None
        for key in ('encode_seconds','decode_and_encode_seconds'):
            report[key]=sum(x[key] for x in report['measurements'])
        report['status']='completed';np.save(path.with_suffix('.npy'),np.stack(values));bench.dump(path,report)
    finally:
        if cache:cache.close()


def profile():
    """Separate synchronized probes; never count these as throughput timings."""
    import torch
    from action_search import ActionEngine
    from statistics import median
    design=json.loads((OUT/'design.json').read_text());media={m['id']:m for m in design['media']}
    engine=ActionEngine();ext=engine.model();visual=ext._model.model.visual
    seg=design['segments'][0];engine.encode_segment(media[seg['media_id']],seg)
    rows=[]
    for variant in ('baseline','cached'):
        cache=PositionCache(visual) if variant=='cached' else None
        saved={};timers={}
        for name in ('fast_pos_embed_interpolate','rot_pos_emb'):
            original=getattr(visual,name);saved[name]=original;timers[name]=[]
            def timed(module,*args,name=name,original=original,**kwargs):
                torch.mps.synchronize();t=time.perf_counter();v=original(*args,**kwargs);torch.mps.synchronize()
                timers[name].append(time.perf_counter()-t);return v
            setattr(visual,name,MethodType(timed,visual))
        durations=[]
        for _ in range(5):
            t=time.perf_counter();engine.encode_segment(media[seg['media_id']],seg);durations.append(time.perf_counter()-t)
        for name,original in saved.items():setattr(visual,name,original)
        rows.append(dict(variant=variant,seconds=timers,wall=durations,
                         warm_median={k:median(v[1:]) for k,v in timers.items()},cache=cache.stats() if cache else None))
        if cache:cache.close()
    bench.dump(OUT/'profile.json',{'method':'5 repetitions, first includes cache miss; synchronized hooks perturb scheduling; no throughput inference','rows':rows})
    print(json.dumps(rows,ensure_ascii=False),flush=True)


def paired():
    """Supplementary hot-cache ABBA in one model instance, two fixed clips."""
    import numpy as np
    import torch
    from action_search import ActionEngine
    from statistics import mean
    design=json.loads((OUT/'design.json').read_text());media={m['id']:m for m in design['media']}
    segments=[design['segments'][0],design['segments'][-1]]
    engine=ActionEngine();ext=engine.model();visual=ext._model.model.visual
    expected=[engine.encode_segment(media[s['media_id']],s) for s in segments]
    cache=PositionCache(visual)
    for s in segments:engine.encode_segment(media[s['media_id']],s)
    cached_methods={n:getattr(visual,n) for n in cache.originals}
    original_extract=ext.extract_video_segment_features;elapsed=[]
    def measured(frames):
        torch.mps.synchronize();t=time.perf_counter();v=original_extract(frames);torch.mps.synchronize()
        elapsed.append(time.perf_counter()-t);return v
    ext.extract_video_segment_features=measured;rows=[]
    try:
        for turn,variant in enumerate(['baseline','cached','cached','baseline']*4):
            for n,method in (cache.originals if variant=='baseline' else cached_methods).items():setattr(visual,n,method)
            elapsed.clear();t=time.perf_counter()
            for s,v in zip(segments,expected):
                np.testing.assert_array_equal(engine.encode_segment(media[s['media_id']],s),v)
            rows.append(dict(turn=turn,variant=variant,encode_seconds=sum(elapsed),decode_and_encode_seconds=time.perf_counter()-t))
        means={v:mean(r['encode_seconds'] for r in rows if r['variant']==v) for v in ('baseline','cached')}
        result={'method':'One resident model, two fixed clips per turn, 16 ABBA turns, cache prefilled; no synchronized inner-function hooks',
                'rows':rows,'mean_encode_seconds':means,'time_reduction_percent':100*(1-means['cached']/means['baseline']),
                'all_vectors_exact':True,'cache':cache.stats()}
        bench.dump(OUT/'paired.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:cache.close()


def suite():
    from action_search import read_snapshot, SETTINGS
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'design.json').exists()
    before=bench.fingerprint();bench.dump(OUT/'production-before.json',before)
    meta,_,_=read_snapshot(Path(before['project']),complete=True)
    for m in meta['media']:
        p=Path(m['location'])/m['path'];p=p if p.is_absolute() else bench.ROOT/p
        m['absolute_path']=str(p.resolve());m['sha256']=bench.sha(p)
    source=json.loads((bench.OUT/'qwen-linear-20261028/design.json').read_text())
    design={'project':before['project'],'media':meta['media'],'segments':meta['segments'],
            'sample_media_ids':source['sample_media_ids'],'settings':SETTINGS,
            'queries_source':source['queries_source'],'query_vectors_source':source['query_vectors_source'],
            'baseline':'Production same-weight Linear, batch 1; no position cache',
            'candidate':'Same baseline plus per-instance 64MiB LRU of positional interpolation and rotary frequency outputs'}
    bench.dump(OUT/'design.json',design)
    env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
    commands=[('profile',[sys.executable,'-u',__file__,'profile'])]
    for v,p,r in [('baseline','sample',1),('cached','sample',1),('cached','sample',2),('baseline','sample',2),('baseline','rest',1),('cached','rest',1)]:
        name=f'{v}-{p}-r{r}';commands.append((name,[sys.executable,'-u',__file__,'worker','--variant',v,'--part',p,'--repeat',str(r)]))
    for name,cmd in commands:
        with (OUT/f'{name}.log').open('w') as log:
            ret=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        print(name,'exit',ret.returncode,flush=True)
        if ret.returncode:raise RuntimeError(name)
    for m in design['media']:assert bench.sha(m['absolute_path'])==m['sha256']
    after=bench.fingerprint();bench.dump(OUT/'production-after.json',after);assert before==after

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['suite','profile','paired','worker']);p.add_argument('--variant',choices=['baseline','cached']);p.add_argument('--part',choices=['sample','rest']);p.add_argument('--repeat',type=int,default=1);a=p.parse_args()
    {'suite':suite,'profile':profile,'paired':paired,'worker':lambda:worker(a.variant,a.part,a.repeat)}[a.command]()
