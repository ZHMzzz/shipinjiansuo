"""Isolated OWLv2 1/2-frame benchmark; production adapters, no production edits."""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import bench

OUT=Path(os.environ.get('MICROBATCH_OUT', bench.OUT/'microbatch-20261028')).resolve()

def child(args):
    import frame_acceleration
    import owl_variants
    def install(*a,**kw):
        frame_acceleration.enable()
        import wise.feature.transformers_owlv2 as owl
        cls=owl.TransformersOWLv2FeatureExtractor
        source=textwrap.dedent(inspect.getsource(cls.extract_image_features))
        assert source.count('len(images) > 1:')==1 and source.count('images.split(1)')==1
        if args.batch==2:
            source=source.replace('len(images) > 1:', 'len(images) > 2:').replace('images.split(1)','images.split(2)')
            ns={};exec(compile(source,'<isolated-owl-batch2>','exec'),vars(owl),ns)
            cls.extract_image_features=ns['extract_image_features']
        original=owl.TransformersOWLv2Model.get_image_features
        batches={}
        def forward(self,*a,**kw):
            images=kw.get('images',a[0] if a else None)
            size=len(images);batches[str(size)]=batches.get(str(size),0)+1
            bench.dump(Path(args.metrics).with_name('observed-batches.json'),batches)
            assert size<=args.batch
            return original(self,*a,**kw)
        owl.TransformersOWLv2Model.get_image_features=forward
    owl_variants.install_variant=install
    # CoreML is installed by the production adapter, so skip the old experimental adapter.
    args.owl='production_fast_fp16';args.face='cpu';args.processor_device='cpu'
    bench.child(args)

def suite():
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.glob('owl-b*')):raise FileExistsError('Use a fresh MICROBATCH_OUT; existing trials are immutable')
    before=bench.fingerprint();bench.dump(OUT/'production-before.json',before)
    design=json.loads((bench.OUT/'design.json').read_text())
    for m in design['media']:
        if m['id'] in design['sample_media_ids']: assert bench.sha(m['absolute_path'])==m['sha256']
    env={**os.environ,'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false',
         'VIDEO_FRAME_ACCELERATION':'1','FEATURE_EXTRACTOR_CONFIG':json.dumps({'default':{'device':'mps','compile':False,'warmup':True},bench.FACE:{'device':'cpu','compile':False,'warmup':True}})}
    for batch,rep in [(1,1),(2,1),(2,2),(1,2)]:
        run=OUT/f'owl-b{batch}-r{rep}';run.mkdir()
        project=run/'project';row={'batch':batch,'repeat':rep,'status':'running','project':str(project)}
        cmds=[('extract',[sys.executable,'-u',__file__,'child','--batch',str(batch),'--inputs',str(bench.OUT/'inputs'),'--project',str(project),'--metrics',str(run/'metrics.json')]),
              ('index',[sys.executable,'-m','wise','create-index','--project-dir',str(project),'--overwrite'])]
        for phase,cmd in cmds:
            t=time.perf_counter()
            with (run/(phase+'.log')).open('w') as log:
                result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            row[phase+'_wall_seconds']=time.perf_counter()-t;row[phase+'_exit_code']=result.returncode
            bench.dump(run/'timing.json',row)
            print(run.name,phase,row[phase+'_wall_seconds'],result.returncode,flush=True)
            if result.returncode: raise RuntimeError(f'Failed: {run}/{phase}.log')
        row['status']='completed';row['wise_total_seconds']=row['extract_wall_seconds']+row['index_wall_seconds'];bench.dump(run/'timing.json',row)
    after=bench.fingerprint();bench.dump(OUT/'production-after-owl.json',after)
    assert before==after,'Production changed during experiment'
    from evaluate import compare
    result=compare(OUT/'owl-b1-r1/project',OUT/'owl-b2-r1/project',bench.OUT/'queries.json',bench.OUT/'query-baseline.npz',OUT/'owl-quality.json',Path(before['project']),None)
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['suite','child']);p.add_argument('--batch',type=int,choices=[1,2]);p.add_argument('--inputs');p.add_argument('--project');p.add_argument('--metrics');a=p.parse_args()
    suite() if a.mode=='suite' else child(a)
