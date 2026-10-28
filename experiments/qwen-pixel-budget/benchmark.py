"""Single-video pixel-budget ablation; no production writes except stop/start."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
LOW=1048576
HIGH=7864320
QUERIES=[
    ('红色安全帽写字',True),
    ('蓝色工服的人员在门岗旁，一人坐着写字、一人站着',True),
    ('站在门岗旁的蓝色工服人员头上没有戴安全帽',True),
    ('红色安全帽骑自行车',False),
    ('户外慢跑',False),
    ('有人在游泳池里游泳',False),
]


def dump(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def audit(design):
    changed=[p for p,h in design['protected'].items() if not Path(p).exists() or sha(p)!=h]
    result=dict(changed_files=changed,pointer_unchanged=os.readlink(ROOT/'projects/demo')==design['pointer'])
    dump(HERE/'production-after.json',result)
    assert not changed and result['pointer_unchanged'],result


def worker(name,budget):
    import numpy as np
    import torch
    from scripts.action_search import ActionEngine
    from resource_metrics import ResourceSampler
    d=json.loads((HERE/'design.json').read_text())
    engine=ActionEngine();report=dict(name=name,budget=budget,rounds=[],segments=[],status='running')
    with ResourceSampler() as resources:
        t=time.perf_counter();extractor=engine.model();extractor.max_total_pixels=budget
        assert str(next(extractor._model.parameters()).dtype)=='torch.float16'
        report['dtype']=str(next(extractor._model.parameters()).dtype)
        report['load_seconds']=time.perf_counter()-t
        inputs={};stats={}
        def hook(model,args,kw):
            inputs['tokens']=int(kw['input_ids'].shape[-1])
            if kw.get('video_grid_thw') is not None:inputs['grid']=kw['video_grid_thw'].cpu().tolist()
        extractor._model.register_forward_pre_hook(hook,with_kwargs=True)
        original=extractor.extract_video_segment_features
        def capture(frames):
            t=time.perf_counter()
            fingerprint=hashlib.sha256(frames.cpu().numpy().tobytes()).hexdigest()
            hash_time=time.perf_counter()-t
            inputs.clear();t=time.perf_counter();value=original(frames);torch.mps.synchronize()
            stats.update(encode_seconds=time.perf_counter()-t,hash_seconds=hash_time,raw_hash=fingerprint,raw_shape=list(frames.shape),model_input=dict(inputs))
            return value
        extractor.extract_video_segment_features=capture
        query_vectors=extractor.extract_text_features([q[0] for q in QUERIES])
        torch.mps.synchronize();np.save(HERE/(name+'-queries.npy'),query_vectors)
        # Warm the visual shape using the first real segment, outside measured rounds.
        t=time.perf_counter();engine.encode_segment(d['media'],d['segments'][0]);report['warmup_seconds']=time.perf_counter()-t
        vectors=[]
        for round_id in range(2):
            wall=time.perf_counter();hash_time=0
            for seg in d['segments']:
                t=time.perf_counter();vector=engine.encode_segment(d['media'],seg);seconds=time.perf_counter()-t
                assert np.isfinite(vector).all()
                vectors.append(vector);hash_time+=stats['hash_seconds']
                report['segments'].append(dict(round=round_id,**seg,**stats,wall_seconds_excluding_hash=seconds-stats['hash_seconds']))
            report['rounds'].append(dict(round=round_id,wall_seconds_excluding_hash=time.perf_counter()-wall-hash_time))
            dump(HERE/(name+'.json'),report);print(name,'round',round_id,report['rounds'][-1],flush=True)
        np.save(HERE/(name+'-vectors.npy'),np.stack(vectors))
    report['resources']=resources.result;report['status']='completed';dump(HERE/(name+'.json'),report)


def suite():
    import httpx
    import numpy as np
    import psutil
    from scripts.action_search import read_snapshot
    assert not (HERE/'design.json').exists(),'Refuse to overwrite frozen experiment'
    project=(ROOT/'projects/demo').resolve();meta,matrix,_=read_snapshot(project,complete=True)
    media=next(m for m in meta['media'] if m['id']==27)
    indices=[i for i,s in enumerate(meta['segments']) if s['media_id']==27]
    protected=[p for p in project.rglob('*') if p.is_file() and p.suffix!='.log']
    protected+=list((ROOT/'scripts').glob('*.py'))+[p for p in (ROOT/'ui').rglob('*') if p.is_file()]
    protected += [ROOT/'wise_config.yaml',ROOT/'data/people.sqlite',ROOT/'wise/src/wise/feature/qwen3vl.py']
    source=Path(media['location'])/media['path']
    if not source.is_absolute():source=ROOT/source
    protected.append(source)
    design=dict(date='2026-10-28',project=str(project),pointer=os.readlink(ROOT/'projects/demo'),media=media,
        segments=[meta['segments'][i] for i in indices],settings=meta['settings'],queries=QUERIES,
        protected={str(p):sha(p) for p in protected},order=['low-1','high-1','high-2','low-2'])
    dump(HERE/'design.json',design);np.save(HERE/'production-vectors.npy',matrix[indices])
    jobs=httpx.get('http://127.0.0.1:9670/mvp-api/jobs',timeout=20).json()['jobs']
    assert not [j for j in jobs if j['status'] not in ('completed','failed')]
    status=dict(status='running',started=time.time());dump(HERE/'suite.json',status)
    subprocess.run([sys.executable,str(ROOT/'scripts/manage.py'),'stop'],check=True)
    try:
        for name in design['order']:
            budget=LOW if name.startswith('low') else HIGH
            with (HERE/(name+'.log')).open('w') as log:
                proc=subprocess.Popen([sys.executable,str(Path(__file__)),'worker','--name',name,'--budget',str(budget)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'HF_HUB_OFFLINE':'1'})
                start=time.monotonic();low=0
                while proc.poll() is None:
                    low=low+1 if psutil.virtual_memory().available<2*1024**3 else 0
                    if low>=6 or time.monotonic()-start>600:
                        proc.terminate()
                        try:proc.wait(15)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait()
                        raise RuntimeError('Memory/time guard: '+name)
                    time.sleep(.5)
                assert proc.returncode==0,name+' failed: inspect log'
        status['status']='completed'
    except BaseException as exc:
        status['status']='failed';status['error']=repr(exc);raise
    finally:
        subprocess.run([sys.executable,str(ROOT/'scripts/manage.py'),'start'],check=True)
        status['service_restored']=False
        for _ in range(120):
            try:
                response=httpx.get('http://127.0.0.1:9670/mvp-api/actions/status',timeout=1)
                if response.status_code==200 and response.json()['model_loaded']:
                    status['service_restored']=True;status['restored_status']=response.json();break
            except httpx.HTTPError:pass
            time.sleep(1)
        status['finished']=time.time();dump(HERE/'suite.json',status);audit(design)
        assert status['service_restored']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['suite','worker']);parser.add_argument('--name');parser.add_argument('--budget',type=int)
    args=parser.parse_args();suite() if args.mode=='suite' else worker(args.name,args.budget)
