"""Isolated WISE feature/index benchmark; production is read-only.

Run with scripts/env.sh sourced. Variants are process-local monkeypatches.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
OWL = 'transformers/owlv2/google/owlv2-base-patch16-ensemble'
FACE = 'deepinsight/insightface/buffalo_l/_unknown'
MEDIA_IDS = [1, 2, 3, 5, 9, 13, 14, 19, 21, 40, 41]


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(4 * 1024**2):
            h.update(b)
    return h.hexdigest()


def fingerprint():
    p = (ROOT / 'projects/demo').resolve()
    files = {str(f.relative_to(p)): sha(f) for f in sorted(p.rglob('*'))
             if f.is_file() and f.suffix not in ('.log',)}
    code = [ROOT/'wise_config.yaml', ROOT/'requirements-macos.lock.txt']
    code += list((ROOT/'scripts').glob('*.py')) + list((ROOT/'wise/src').rglob('*.py'))
    return {'project': str(p), 'project_files': files,
            'people_sha256': sha(ROOT/'data/people.sqlite'),
            'code': {str(f.relative_to(ROOT)): sha(f) for f in sorted(code)}}


def prepare():
    if (OUT/'design.json').exists():
        raise RuntimeError('Design already frozen; do not overwrite')
    p = (ROOT/'projects/demo').resolve()
    c = sqlite3.connect('file:'+str(p/'metadata/internal.db')+'?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    media = [dict(r) for r in c.execute('SELECT m.id,m.path,m.duration,s.location FROM media m JOIN source_collections s ON s.id=m.source_collection_id ORDER BY m.id')]
    inputs = OUT/'inputs'; inputs.mkdir(exist_ok=True)
    for m in media:
        f = Path(m['location'])/m['path']
        if not f.is_absolute(): f = ROOT/f
        m['absolute_path'] = str(f.resolve())
        if m['id'] in MEDIA_IDS:
            m['sha256'] = sha(f)
            (inputs/m['path']).symlink_to(f.resolve())
    design = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'media': media, 'sample_media_ids': MEDIA_IDS,
              'total_sample_duration': sum(m['duration'] for m in media if m['id'] in MEDIA_IDS),
              'settings': {'fps': 2, 'owl_threshold': .02, 'owl_size': 960,
                           'face_det_size': [640,640], 'face_det_threshold': .5, 'face_gate': .45},
              'versions': {n: importlib.metadata.version(n) for n in ['torch','transformers','onnxruntime','insightface','numpy']},
              'hardware': subprocess.run(['sysctl','-n','machdep.cpu.brand_string','hw.memsize'],capture_output=True,text=True).stdout,
              'server_running': True, 'baseline_scope': 'WISE extraction and index build on fixed sample; no Qwen/transcode/publication',
              'production_before': fingerprint()}
    dump(OUT/'design.json',design)
    print(json.dumps({'videos':len(MEDIA_IDS),'duration':design['total_sample_duration'],'versions':design['versions']},ensure_ascii=False))


def child(args):
    import logging
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',level=logging.INFO)
    import torch
    from resource_metrics import ResourceSampler
    from owl_variants import install_variant
    install_variant(args.owl, processor_device=args.processor_device)
    face_controller = None
    if args.face != 'cpu':
        from face_variants import install_variant as install_face
        # Graph offload is profiled in coreml_probe; exclude profiling overhead
        # from all feature/index timing runs.
        face_controller = install_face(args.face, cache_dir=OUT/'cache/face')
    import wise._main.extract_features as extract
    from wise.feature.transformers_owlv2 import TransformersOWLv2FeatureExtractor, TransformersOWLv2Model
    from wise.feature.insightface import InsightFaceFeatureExtractor
    stats = {'owl':args.owl,'face':args.face,'processor_device':args.processor_device,'timers':{},'initializing':False}

    def timed(cls, name, label, gpu):
        original = getattr(cls, name)
        depth = 0
        def wrapper(self,*a,**kw):
            nonlocal depth
            outer = depth == 0
            if outer:
                if gpu: torch.mps.synchronize()
                started = time.perf_counter()
            depth += 1
            try:
                return original(self,*a,**kw)
            finally:
                depth -= 1
                if outer:
                    if gpu: torch.mps.synchronize()
                    key = ('warmup_' if stats['initializing'] else '') + label
                    row=stats['timers'].setdefault(key, {'seconds':0.0,'calls':0})
                    row['seconds'] += time.perf_counter()-started; row['calls'] += 1
        setattr(cls,name,wrapper)
    timed(TransformersOWLv2FeatureExtractor,'extract_image_features','owl_features',True)
    timed(TransformersOWLv2Model,'get_image_features','owl_forward',True)
    timed(InsightFaceFeatureExtractor,'extract_image_features','face_features',False)
    init = extract.initialise_feature_extractors
    def init_wrapper(*a,**kw):
        stats['initializing'] = True; started=time.perf_counter()
        try: return init(*a,**kw)
        finally:
            torch.mps.synchronize();stats['initialization_seconds']=time.perf_counter()-started;stats['initializing']=False
    extract.initialise_feature_extractors = init_wrapper
    started=time.perf_counter()
    with ResourceSampler() as resources:
        try:
            cmd=['extract-features',args.inputs,'--project-dir',args.project,'--num-workers','0','--yes',
                 '--video-feature-id',OWL,'--video-feature-id',FACE]
            extract.main(cmd)
            stats['status']='completed'
        except BaseException as e:
            stats['status']='failed';stats['error']=repr(e)
            raise
        finally:
            stats['extract_main_seconds']=time.perf_counter()-started
            dump(args.metrics,stats)
    stats['resources']=resources.result
    if face_controller is not None:
        stats['face_backend']=face_controller.report()
    dump(args.metrics,stats)


def run(args):
    design=json.loads((OUT/'design.json').read_text())
    for media in design['media']:
        if media['id'] in design['sample_media_ids']:
            if sha(media['absolute_path']) != media['sha256']:
                raise RuntimeError('Frozen video changed: '+media['path'])
    run_dir=OUT/'runs'/args.name
    if run_dir.exists():raise RuntimeError('Run directory exists; never reuse partial indexes')
    run_dir.mkdir(parents=True)
    project=run_dir/'project'
    env={**os.environ,'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false',
         'FEATURE_EXTRACTOR_CONFIG':json.dumps({'default':{'device':'mps','compile':False,'warmup':True},FACE:{'device':'cpu','compile':False,'warmup':True}})}
    row={'name':args.name,'owl':args.owl,'face':args.face,'processor_device':args.processor_device,
         'project':str(project),'status':'running','started_epoch':time.time()}
    dump(run_dir/'timing.json',row)
    command=[sys.executable,'-u',str(__file__),'child','--inputs',args.inputs,'--project',str(project),
             '--metrics',str(run_dir/'metrics.json'),'--owl',args.owl,'--face',args.face,'--processor-device',args.processor_device]
    for phase,cmd in [('extract',command),('index',[sys.executable,'-u','-m','wise','create-index','--project-dir',str(project),'--overwrite'])]:
        started=time.perf_counter()
        with (run_dir/(phase+'.log')).open('w') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
        row[phase+'_wall_seconds']=time.perf_counter()-started;row[phase+'_exit_code']=result.returncode
        dump(run_dir/'timing.json',row)
        print(args.name,phase,round(row[phase+'_wall_seconds'],3),result.returncode,flush=True)
        if result.returncode:
            row['status']='failed';dump(run_dir/'timing.json',row);return
    c=sqlite3.connect(project/'metadata/internal.db')
    row['vector_counts']=dict(c.execute('SELECT feature_extractor_id,count(*) FROM vectors GROUP BY 1'))
    row['wise_total_seconds']=row['extract_wall_seconds']+row['index_wall_seconds']
    row['status']='completed';row['finished_epoch']=time.time();dump(run_dir/'timing.json',row)


if __name__ == '__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    s.add_parser('prepare')
    for name in ['run','child']:
        q=s.add_parser(name)
        q.add_argument('--owl',default='baseline');q.add_argument('--face',default='cpu')
        q.add_argument('--processor-device',default='cpu');q.add_argument('--inputs',default=str(OUT/'inputs'))
        if name=='run':q.add_argument('--name',required=True)
        else:
            q.add_argument('--project',required=True);q.add_argument('--metrics',required=True)
    args=p.parse_args()
    {'prepare':prepare,'run':lambda:run(args),'child':lambda:child(args)}[args.command]()
