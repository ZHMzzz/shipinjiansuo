"""Read-only production inputs; all outputs live beside this file.

suite is authorized to stop/restart only this project's recorded local service.
Reference decoder/encoder are unmodified exports of the local WISE upstream HEAD.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
MODEL = 'hf/Qwen/Qwen3-VL-Embedding/2B'
REPEAT_MEDIA = [1,18,19,21,27,35,37,40]


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def load(name):
    return json.loads((HERE/name).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference(name):
    path = HERE/'reference'/name
    spec = importlib.util.spec_from_file_location('_wise_original_'+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare():
    import numpy as np
    import httpx
    from scripts.action_search import read_snapshot
    if (HERE/'design.json').exists():
        raise RuntimeError('Design already frozen; use existing experiment or a new directory')
    project = (ROOT/'projects/demo').resolve()
    meta, matrix, _ = read_snapshot(project, complete=True)
    old = json.loads((ROOT/'experiments/qwen8b-full-library/dataset.json').read_text())
    sources = []
    for media in meta['media']:
        path = (ROOT/Path(media['location'])/media['path']).resolve()
        sources.append({**media, 'absolute_path':str(path), 'sha256':sha(path)})
    known = {m['id']:m for m in sources}
    for source in old['sources']:
        assert known[source['id']]['sha256'] == source['sha256'], source['id']
    protected = [p for p in project.rglob('*') if p.is_file() and p.suffix != '.log']
    protected += list((ROOT/'scripts').glob('*.py')) + list((ROOT/'ui').glob('*'))
    protected += [ROOT/'wise_config.yaml', ROOT/'data/people.sqlite']
    protected = [p for p in protected if p.is_file()]
    hashes = {str(p):sha(p) for p in protected}
    ref = HERE/'reference'; ref.mkdir(exist_ok=True)
    commit = subprocess.check_output(['git','-C',str(ROOT/'wise'),'rev-parse','HEAD'],text=True).strip()
    refs = {}
    for source in ['feature/qwen3vl.py','feature/hf_models.py','dataloader/video_segment_dataset.py']:
        data = subprocess.check_output(['git','-C',str(ROOT/'wise'),'show',f'{commit}:src/wise/{source}'])
        target = ref/Path(source).name;target.write_bytes(data);refs[source]=sha(target)
    assert sha(ref/'hf_models.py') == sha(ROOT/'wise/src/wise/feature/hf_models.py')
    queries = old['queries'] + [dict(id=f'D{i+1:02}',text=text,group='用户诊断',answer_status='ungraded')
        for i,text in enumerate(['骑自行车','红色安全帽骑自行车','红色安全帽写字','户外慢跑','未佩戴安全帽'])]
    np.save(HERE/'production-vectors.npy',matrix)
    shutil.copy2(ROOT/'experiments/qwen8b-full-library/labels-freeze.json',HERE/'labels-freeze.json')
    with httpx.Client(base_url='http://127.0.0.1:9670', timeout=20) as client:
        jobs = client.get('/mvp-api/jobs').json()['jobs']
        assert not [j for j in jobs if j['status'] not in ('completed','failed')], 'Active uploads'
        api = client.get('/mvp-api/actions/status').json()
    design = dict(project=str(project), pointer=os.readlink(ROOT/'projects/demo'),
        settings=meta['settings'], sources=sources, production_segments=meta['segments'],queries=queries,
        labeled_media_ids=sorted(s['id'] for s in old['sources']),
        labeled_dataset_sha256=sha(ROOT/'experiments/qwen8b-full-library/dataset.json'),
        protected_files=hashes, upstream_commit=commit, reference_sha256=refs,
        model=MODEL, repeats=REPEAT_MEDIA, before_api=api,
        date='2026-10-28', timezone='America/Los_Angeles',
        variants={'current':dict(dtype='float16',pixels=1048576,sampling='production OpenCV'),
                  'original':dict(dtype='float32',pixels=7864320,sampling='unmodified WISE VideoSegmentDataset')})
    dump(HERE/'design.json',design)
    print('Frozen',len(sources),'videos',len(meta['segments']),'current segments',len(queries),'queries',flush=True)


def audit():
    design = load('design.json')
    changed = [p for p,h in design['protected_files'].items() if not Path(p).exists() or sha(p)!=h]
    changed_sources = [s['id'] for s in design['sources'] if sha(s['absolute_path'])!=s['sha256']]
    report = dict(pointer_unchanged=os.readlink(ROOT/'projects/demo')==design['pointer'],
        changed_files=changed,changed_source_videos=changed_sources,checked_files=len(design['protected_files']))
    dump(HERE/'production-after.json',report)
    assert report['pointer_unchanged'] and not changed and not changed_sources, report
    return report


def worker(args):
    import numpy as np
    import torch
    from resource_metrics import ResourceSampler
    from scripts.action_search import ActionEngine, windows
    from wise.data_models import MediaChunkType, ModalityType
    from wise.feature.store import FaissStore
    from wise.index.feature_search_index import FeatureSearchIndex
    design = load('design.json');folder=HERE/args.name;folder.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.mps.set_per_process_memory_fraction(.7)
    report = dict(status='running',variant=args.variant,name=args.name,segments=[],queries=[],repeat_segments=[],
        videos=[],pid=os.getpid(),sampler=design['variants'][args.variant],timing_scope='Qwen phase; excludes OWLv2/face/transcode/upload')
    vectors=[];query_vectors=[];captured={};engine=None
    def checkpoint(): dump(folder/'result.json',report)
    def phase(value):
        report['phase']=value;checkpoint();print(args.name,value,flush=True)
    with ResourceSampler() as resources:
        phase('load_model');start=time.perf_counter()
        if args.variant=='current':
            engine=ActionEngine();extractor=engine.model()
        else:
            Extractor=reference('qwen3vl.py').Qwen3VLEmbeddingFeatureExtractor
            extractor=Extractor(MODEL,device='mps',warmup=False,
                config=Extractor.Config(device='mps',warmup=False,compile=False))
            _=extractor._model
        torch.mps.synchronize();report['model_load_seconds']=time.perf_counter()-start
        report['weight_dtype']=str(next(extractor._model.parameters()).dtype)
        def hook(module,inputs,kwargs):
            captured['tokens']=int(kwargs['input_ids'].shape[-1])
            if kwargs.get('video_grid_thw') is not None:
                captured['video_grid_thw']=kwargs['video_grid_thw'].detach().cpu().tolist()
        extractor._model.register_forward_pre_hook(hook,with_kwargs=True)
        start=time.perf_counter();extractor.extract_text_features(['视频中的场景和动作']);torch.mps.synchronize()
        report['text_warmup_seconds']=time.perf_counter()-start
        original_encode=extractor.extract_video_segment_features
        stats={}
        def encode(frames):
            captured.clear();t=time.perf_counter()
            value=original_encode(frames);torch.mps.synchronize()
            stats.update(encode_seconds=time.perf_counter()-t,frame_shape=list(frames.shape),input=dict(captured))
            return value
        extractor.extract_video_segment_features=encode
        dataset_cls=reference('video_segment_dataset.py').VideoSegmentDataset
        def encode_video(media, repeated=False):
            rows=[];wall=time.perf_counter();mid=media['id']
            if args.variant=='current':
                for segment in windows(media['duration']):
                    rec=dict(media_id=mid,ts=segment[0],te=segment[1])
                    t=time.perf_counter();vec=engine.encode_segment(media,rec)
                    rows.append((rec,vec,time.perf_counter()-t,dict(stats)))
            else:
                dataset=dataset_cls({str(mid):media['absolute_path']},segment_duration=8.,segment_overlap=4.,
                    num_frames_per_segment=16,preprocessing_function_map={MODEL:extractor.preprocess_video_segment},thumbnails=True)
                iterator=iter(dataset)
                while True:
                    t=time.perf_counter()
                    try: _,chunks=next(iterator)
                    except StopIteration: break
                    chunk=chunks[MediaChunkType.VIDEO][MODEL]
                    vec=extractor.extract_video_segment_features(chunk.tensor).vectors[0]
                    rows.append((dict(media_id=mid,ts=float(chunk.pts),te=float(chunk.end_pts)),vec,
                                 time.perf_counter()-t,dict(stats)))
            elapsed=time.perf_counter()-wall
            assert rows, ('No segments',mid)
            for rec,vec,elapsed_segment,stat in rows:
                assert np.isfinite(vec).all() and abs(np.linalg.norm(vec)-1)<1e-4
                row={**rec,**stat,'wall_seconds':elapsed_segment}
                if repeated: report['repeat_segments'].append(row)
                else: report['segments'].append(row);vectors.append(vec)
            report['videos'].append(dict(media_id=mid,wall_seconds=elapsed,segments=len(rows),repeat=repeated))
            checkpoint();print('VIDEO',args.name,mid,'repeat' if repeated else 'main',len(rows),round(elapsed,3),flush=True)
        phase('video_indexing')
        media=design['sources'] if not args.check else [m for m in design['sources'] if m['id'] in REPEAT_MEDIA]
        for item in media: encode_video(item)
        if not args.check:
            phase('video_repeat')
            for item in media:
                if item['id'] in REPEAT_MEDIA: encode_video(item,True)
        np.save(folder/'vectors.npy',np.stack(vectors))
        phase('native_index_write');start=time.perf_counter()
        features=folder/'features';features.mkdir()
        store=FaissStore(ModalityType.VIDEO,str(features));store.enable_write()
        store.add(list(range(1,len(vectors)+1)),np.stack(vectors));store.save_current_shard()
        report['feature_store_seconds']=time.perf_counter()-start
        index=FeatureSearchIndex(ModalityType.VIDEO,MODEL,{'features_dir':features,'index_dir':folder/'index'})
        start=time.perf_counter();index.create_index('IndexFlatIP');report['faiss_build_seconds']=time.perf_counter()-start
        phase('text_queries')
        for round_id in range(3):
            for query in design['queries']:
                t=time.perf_counter();vector=extractor.extract_text_features([query['text']]);torch.mps.synchronize()
                seconds=time.perf_counter()-t
                assert np.isfinite(vector).all()
                report['queries'].append(dict(id=query['id'],round=round_id,seconds=seconds,tokens=captured['tokens']))
                query_vectors.append(vector[0])
            checkpoint();print('QUERY_ROUND',args.name,round_id,flush=True)
        np.save(folder/'query-vectors.npy',np.stack(query_vectors))
        report['status']='completed';checkpoint()
    report['resources']=resources.result;report['phase']='complete';checkpoint()


def guarded_worker(name,variant,check=False):
    import psutil
    command=[sys.executable,str(Path(__file__)),'worker','--name',name,'--variant',variant]
    if check:command.append('--check')
    samples=[];start=time.monotonic();low=0;reason=None
    with (HERE/(name+'.log')).open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false'})
        while proc.poll() is None:
            available=psutil.virtual_memory().available
            low=low+1 if available<2*1024**3 else 0
            samples.append(dict(seconds=time.monotonic()-start,available_bytes=available,swap_bytes=psutil.swap_memory().used))
            if low>=6 or time.monotonic()-start>3000:
                reason='memory_guard' if low>=6 else 'time_guard'
                proc.terminate()
                try:proc.wait(15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                break
            time.sleep(.5)
        code=proc.wait()
    dump(HERE/(name+'-resources.json'),dict(exit_code=code,stop_reason=reason,wall_seconds=time.monotonic()-start,samples=samples))
    if code:raise RuntimeError(f'{name} failed ({reason or code}); see log')


def suite():
    import httpx
    audit()
    with httpx.Client(base_url='http://127.0.0.1:9670',timeout=20) as client:
        jobs=client.get('/mvp-api/jobs').json()['jobs']
        assert not [j for j in jobs if j['status'] not in ('completed','failed')]
    subprocess.run([sys.executable,str(ROOT/'scripts/manage.py'),'stop'],check=True)
    status={'status':'running','started_epoch':time.time()};dump(HERE/'suite.json',status)
    try:
        guarded_worker('current-full','current')
        guarded_worker('original-full','original')
        guarded_worker('current-check','current',True)
        status['status']='completed'
    except BaseException as exc:
        status['status']='failed';status['error']=repr(exc)
        raise
    finally:
        subprocess.run([sys.executable,str(ROOT/'scripts/manage.py'),'start'],check=True)
        ready=False
        for _ in range(120):
            try:
                response=httpx.get('http://127.0.0.1:9670/mvp-api/actions/status',timeout=1)
                if response.status_code==200 and response.json()['model_loaded']:
                    ready=True;status['restored_status']=response.json();break
            except httpx.HTTPError:pass
            time.sleep(1)
        status['service_restored']=ready;status['finished_epoch']=time.time();dump(HERE/'suite.json',status)
        audit()
        assert ready,'Local service not ready after restore; inspect logs/server.log'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','suite','worker','audit'])
    p.add_argument('--name');p.add_argument('--variant',choices=['current','original']);p.add_argument('--check',action='store_true')
    args=p.parse_args()
    {'prepare':prepare,'suite':suite,'worker':lambda:worker(args),'audit':audit}[args.command]()
