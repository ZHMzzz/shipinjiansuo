"""Isolated serial worker experiment; only model objects survive between jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import bench

OUT=bench.OUT/'frame-worker-reuse-20261028'


def worker():
    import logging
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(levelname)s %(message)s')
    from frame_acceleration import enable
    enable()
    import torch
    import psutil
    from resource_metrics import ResourceSampler
    import wise._main.extract_features as extract
    cache={};counts={'created':0,'reused':0};original=extract.FeatureExtractorFactory
    reuse=os.environ['EXPERIMENT_REUSE']=='1'
    def factory(feature_id,config):
        key=(feature_id,json.dumps(config,sort_keys=True))
        if reuse and key in cache:
            counts['reused']+=1
            return cache[key]
        value=original(feature_id,config);counts['created']+=1
        if reuse:cache[key]=value
        return value
    extract.FeatureExtractorFactory=factory
    pipe=os.fdopen(int(os.environ['EXPERIMENT_RESULT_FD']),'w',buffering=1)
    def send(value):pipe.write(json.dumps(value)+'\n')
    def memory():
        process=psutil.Process()
        return {'rss_bytes':process.memory_info().rss,'mps_allocated_bytes':torch.mps.current_allocated_memory(),
                'mps_driver_bytes':torch.mps.driver_allocated_memory(),'open_fds':process.num_fds()}
    send({'event':'ready','pid':os.getpid(),'memory':memory()})
    for line in sys.stdin:
        command=json.loads(line)
        if command['op']=='stop':break
        if command['op']=='stats':
            send({'event':'stats','memory':memory(),'models':len(cache),'counts':counts});continue
        try:
            start=time.perf_counter()
            with ResourceSampler() as resources:
                extract.main(['extract-features',*command['args']])
                torch.mps.synchronize()
            import gc
            gc.collect()
            send({'event':'completed','pid':os.getpid(),'seconds':time.perf_counter()-start,
                  'resources':resources.result,'idle_memory':memory(),'models':len(cache),'counts':dict(counts)})
        except BaseException as error:
            send({'event':'failed','error':repr(error),'pid':os.getpid()})
            # Never reuse possibly contaminated task state after failure.
            break
    pipe.close()


class Process:
    def __init__(self,name,reuse):
        read_fd,write_fd=os.pipe()
        self.log=(OUT/f'{name}.log').open('w')
        started=time.perf_counter()
        self.process=subprocess.Popen([sys.executable,'-u',__file__,'worker'],cwd=bench.ROOT,
            env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                 'EXPERIMENT_REUSE':str(int(reuse)),'EXPERIMENT_RESULT_FD':str(write_fd)},
            stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,text=True,pass_fds=(write_fd,))
        os.close(write_fd);self.pipe=os.fdopen(read_fd)
        self.ready=self.receive();assert self.ready['event']=='ready',self.ready
        self.startup_seconds=time.perf_counter()-started

    def receive(self):
        import select
        if not select.select([self.pipe],[],[],600)[0]:raise TimeoutError('Worker timed out')
        line=self.pipe.readline()
        if not line:raise RuntimeError('Worker exited before reporting; inspect log')
        return json.loads(line)

    def request(self,op,**kwargs):
        start=time.perf_counter();self.process.stdin.write(json.dumps({'op':op,**kwargs})+'\n');self.process.stdin.flush()
        result=self.receive();result['request_wall_seconds']=time.perf_counter()-start;return result

    def close(self):
        if self.process.poll() is None:
            try:self.process.stdin.write('{"op":"stop"}\n');self.process.stdin.flush()
            except BrokenPipeError:pass
            try:self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.terminate();self.process.wait(timeout=10)
        self.pipe.close();self.process.stdin.close();self.log.close()


def main():
    import hashlib
    import shutil
    import sqlite3
    import tempfile
    import faiss
    from action_search import media_catalog
    from native_qwen import suspend_native_store
    faiss.omp_set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    before=bench.fingerprint();bench.dump(OUT/'production-before.json',before)
    base=Path(before['project']);catalog={m['id']:m for m in media_catalog(base)}
    sources={}
    for mid in (40,41,43):
        m=catalog[mid];source=Path(m['location'])/m['path'];source=source if source.is_absolute() else bench.ROOT/source
        sources[mid]={'path':str(source),'sha256':bench.sha(source),'duration':m['duration']}
    report={'status':'running','sources':sources,'trials':[],
            'scope':'Frame extraction only, new-process versus serial cached-model worker; same full-library clones and prepared videos; no Qwen, index rebuild, upload transfer or publication.'}
    bench.dump(OUT/'result.json',report)
    work_root=OUT/'work';work_root.mkdir();persistent=None
    signatures={};process_count=0

    def run_job(variant,mid,repeat,process,charge_startup=False):
        source=Path(sources[mid]['path']);work=Path(tempfile.mkdtemp(prefix='worker-',dir=work_root))
        inputs=work/'inputs';inputs.mkdir();(inputs/source.name).symlink_to(source)
        project=work/'project';subprocess.run(['/bin/cp','-cR',str(base),str(project)],check=True)
        try:
            with suspend_native_store(project):
                result=process.request('extract',args=[str(inputs),'--project-dir',str(project),'--num-workers','0','--yes'])
            assert result['event']=='completed',result
            result.update(variant=variant,media_id=mid,repeat=repeat,duration=sources[mid]['duration'],
                          startup_charged_seconds=process.startup_seconds if charge_startup else 0.)
            # Baseline exit is charged separately by the caller. Reused worker remains alive.
            result['elapsed_seconds']=result['request_wall_seconds']+result['startup_charged_seconds']
            with sqlite3.connect(project/'metadata/internal.db') as db:
                new_id=db.execute('select max(id) from media').fetchone()[0]
                counts=dict(db.execute('select feature_extractor_id,count(*) from vectors where media_id=? group by 1',(new_id,)))
                assert counts.get(bench.OWL,0)>0 and not set(counts)-{bench.OWL,bench.FACE}
                signatures_now={}
                for model,table in [(bench.OWL,'vector_metadata_owlv2'),(bench.FACE,'vector_metadata_insightface')]:
                    ids=db.execute('select id,timestamp from vectors where media_id=? and feature_extractor_id=? order by id',(new_id,model)).fetchall()
                    metadata=db.execute(f'SELECT t.* FROM {table} t JOIN vectors v ON t.vector_id=v.id WHERE v.media_id=? ORDER BY t.vector_id',(new_id,)).fetchall()
                    matrices=[]
                    if counts.get(model):
                        shard=sorted((project/'store'/model/'features').glob('video-*.faiss'))[-1]
                        index=faiss.read_index(str(shard));assert index.ntotal==counts[model]
                        matrix=index.index.reconstruct_n(0,index.ntotal)
                        vector_hash=hashlib.sha256(matrix.tobytes()).hexdigest()
                    else:vector_hash=hashlib.sha256(b'').hexdigest()
                    signatures_now[model]={'vectors':vector_hash,'ids_times_metadata':hashlib.sha256(json.dumps([ids,metadata]).encode()).hexdigest()}
            if mid not in signatures:signatures[mid]=signatures_now
            assert signatures_now==signatures[mid],f'Outputs changed: {variant} {mid}'
            result.update(vector_counts=counts,outputs_exact=True,output_signatures=signatures_now)
            report['trials'].append(result);bench.dump(OUT/'result.json',report)
            print(variant,mid,repeat,round(result['elapsed_seconds'],3),result['counts'],flush=True)
            return result
        finally:
            assert work.parent==work_root and work.name.startswith('worker-');shutil.rmtree(work)

    try:
        # Alternate ordering; repeated source 40 also measures first vs later work.
        for repeat in (1,2):
            for i,mid in enumerate((40,41,43)):
                order=('fresh','reused') if (i+repeat)%2 else ('reused','fresh')
                for variant in order:
                    if variant=='fresh':
                        process_count+=1;proc=Process(f'fresh-{process_count}',False)
                        try:row=run_job(variant,mid,repeat,proc,True)
                        finally:
                            t=time.perf_counter();proc.close();exit_seconds=time.perf_counter()-t
                        row['exit_seconds']=exit_seconds;row['elapsed_seconds']+=exit_seconds
                    else:
                        first=persistent is None
                        if first:
                            persistent=Process('reused',True);report['persistent_startup_seconds']=persistent.startup_seconds
                        row=run_job(variant,mid,repeat,persistent,first)
                    bench.dump(OUT/'result.json',report)
        report['idle_after_six_jobs']=persistent.request('stats')
        # A malformed task fails in isolation; fail-stop then recreate, no state reuse.
        failure=persistent.request('extract',args=[str(OUT/'missing-input'),'--project-dir',str(OUT/'invalid-project'),'--yes'])
        assert failure['event']=='failed';old_pid=persistent.process.pid;persistent.close();persistent=None
        replacement=Process('recovered',True)
        try:
            row=run_job('recovered',43,1,replacement,True)
            report['failure_recovery']={'invalid_task':failure,'old_pid':old_pid,'new_pid':replacement.process.pid,
                                       'next_task_completed':True,'next_task_outputs_exact':row['outputs_exact']}
            assert old_pid!=replacement.process.pid
        finally:replacement.close()
        for source in sources.values():assert bench.sha(source['path'])==source['sha256']
        report['status']='completed'
    except BaseException as error:
        report['status']='failed';report['error']=repr(error);raise
    finally:
        if persistent:persistent.close()
        report['production_unchanged']=bench.fingerprint()==before
        bench.dump(OUT/'result.json',report)
        assert report['production_unchanged']

if __name__=='__main__':
    worker() if len(sys.argv)>1 and sys.argv[1]=='worker' else main()
