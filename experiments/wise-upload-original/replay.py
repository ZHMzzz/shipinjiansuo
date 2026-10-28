"""Upload-stage replay on disposable copies. Never publishes or touches live DBs.

Original variant is an offline WISE-only Qwen writer, not production activation.
Cold model load is reported separately, as the live server preloads Qwen.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'scripts')]
from benchmark import MODEL, audit, dump, load, reference, sha


def child(variant):
    import numpy as np
    import torch
    from scripts.action_search import ActionEngine, windows, read_snapshot
    from scripts.native_qwen import sync_native_qwen
    from resource_metrics import ResourceSampler
    from wise.data_models import MediaChunkType, ModalityType
    from wise.feature.store import FaissStore
    from wise.index.feature_search_index import FeatureSearchIndex
    from wise.wise_project import WiseProject
    from wise.config import APIConfig
    from wise.api.services.project import LocalWiseProjectService
    from wise.api.services.search import LocalSearchService
    design=load('design.json');base=Path(design['project'])
    out=HERE/'replays'/variant;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.mps.set_per_process_memory_fraction(.7)
    started=time.perf_counter()
    if variant=='current':
        engine=ActionEngine();extractor=engine.model()
    else:
        Extractor=reference('qwen3vl.py').Qwen3VLEmbeddingFeatureExtractor
        extractor=Extractor(MODEL,device='mps',warmup=False,config=Extractor.Config(device='mps',warmup=False,compile=False))
        _=extractor._model
    extractor.extract_text_features(['视频中的场景和动作']);torch.mps.synchronize()
    result=dict(variant=variant,model_load_and_text_warmup_seconds=time.perf_counter()-started,trials=[])
    dump(out/'result.json',result)
    for mid in ([40,41] if variant=='current' else [41,40]):
        source=next(s for s in design['sources'] if s['id']==mid)
        with sqlite3.connect((ROOT/'data/ingestion.sqlite').as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            job=dict(db.execute("SELECT * FROM jobs WHERE media_id=? AND status='completed' ORDER BY updated DESC LIMIT 1",(mid,)).fetchone())
        original=ROOT/'data/uploads'/job['id']/('source'+job['suffix'])
        assert sha(original)==job['sha256']
        trial=out/str(mid);trial.mkdir();work=trial/'work';prepared=work/'prepared';prepared.mkdir(parents=True)
        project=work/'demo';video=prepared/source['path']
        row=dict(media_id=mid,duration=source['duration'],stages={},status='running',source_sha256=source['sha256'])
        result['trials'].append(row)
        def checkpoint():dump(out/'result.json',result)
        def measure(name,fn):
            t=time.perf_counter();value=fn();row['stages'][name]=time.perf_counter()-t
            checkpoint();print(variant,mid,name,round(row['stages'][name],3),flush=True);return value
        def command(cmd,logname):
            with (trial/logname).open('w') as log:
                subprocess.run(cmd,cwd=ROOT,env={**os.environ,'HF_HUB_OFFLINE':'1'},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        try:
            with ResourceSampler() as resource:
                clock=time.perf_counter()
                measure('probe',lambda:command([str(ROOT/'.media-runtime/bin/ffprobe'),'-v','error','-show_format','-show_streams','-of','json',str(original)],'probe.log'))
                measure('transcode',lambda:command([str(ROOT/'.media-runtime/bin/ffmpeg'),'-y','-v','error','-nostdin','-i',str(original),
                    '-map','0:v:0','-map','0:a:0?','-vf','scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2',
                    '-c:v','libx264','-crf','23','-preset','fast','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(video)],'transcode.log'))
                check_start=time.perf_counter();assert sha(video)==source['sha256'];row['sha_verification_seconds']=time.perf_counter()-check_start
                measure('copy_project',lambda:shutil.copytree(base,project))
                shutil.rmtree(project/'store'/MODEL)
                measure('owl_face_extract',lambda:command([sys.executable,'-u','-m','wise','extract-features',str(prepared),
                    '--project-dir',str(project),'--num-workers','0','--yes'],'frame-extraction.log'))
                measure('owl_face_index',lambda:command([sys.executable,'-u','-m','wise','create-index','--project-dir',str(project),'--overwrite'],'frame-index.log'))
                with sqlite3.connect(project/'metadata/internal.db') as db:
                    newmid=db.execute('SELECT m.id FROM media m JOIN source_collections s ON s.id=m.source_collection_id WHERE s.location=?',(str(prepared),)).fetchone()[0]
                if variant=='current':
                    catalog,records,vectors,missing=engine.prepare(project);assert len(missing)==1
                    new_count=len(list(windows(missing[0]['duration'])))
                    def qwen():
                        for ts,te in windows(missing[0]['duration']):
                            seg=dict(media_id=newmid,ts=ts,te=te)
                            vectors.append(engine.encode_segment(missing[0],seg));records.append(seg)
                    measure('qwen_decode_encode',qwen)
                    def write_qwen():
                        engine.finish(project,catalog,records,vectors);sync_native_qwen(project)
                    measure('qwen_index_write',write_qwen)
                else:
                    newsegments=[];newvectors=[]
                    Dataset=reference('video_segment_dataset.py').VideoSegmentDataset
                    def qwen():
                        ds=Dataset({str(newmid):str(video)},segment_duration=8.,segment_overlap=4.,num_frames_per_segment=16,
                            preprocessing_function_map={MODEL:extractor.preprocess_video_segment},thumbnails=True)
                        for _,chunks in ds:
                            chunk=chunks[MediaChunkType.VIDEO][MODEL]
                            vector=extractor.extract_video_segment_features(chunk.tensor).vectors[0];torch.mps.synchronize()
                            newsegments.append((newmid,float(chunk.pts),float(chunk.end_pts)));newvectors.append(vector)
                    measure('qwen_decode_encode',qwen);new_count=len(newsegments)
                    def write_qwen():
                        meta,oldvectors,_=read_snapshot(base,complete=True)
                        mapping={(s['media_id'],s['ts'],s['te']):v for s,v in zip(meta['segments'],oldvectors)}
                        with sqlite3.connect(project/'metadata/internal.db') as db:
                            existing=db.execute('SELECT id,media_id,timestamp,end_timestamp FROM vectors WHERE feature_extractor_id=? ORDER BY id',(MODEL,)).fetchall()
                            ids=[r[0] for r in existing];matrix=[mapping[tuple(r[1:])] for r in existing]
                            for seg,vector in zip(newsegments,newvectors):
                                ids.append(db.execute('INSERT INTO vectors(modality,feature_extractor_id,media_id,timestamp,end_timestamp) VALUES(?,?,?,?,?)',('VIDEO',MODEL,*seg)).lastrowid)
                                matrix.append(vector)
                        folder=project/'store'/MODEL;features=folder/'features';features.mkdir(parents=True)
                        store=FaissStore(ModalityType.VIDEO,str(features));store.enable_write();store.add(ids,np.stack(matrix));store.save_current_shard()
                        index=FeatureSearchIndex(ModalityType.VIDEO,MODEL,{'features_dir':features,'index_dir':folder/'index'});index.create_index('IndexFlatIP')
                    measure('qwen_index_write',write_qwen)
                row['new_segments']=new_count
                def validate():
                    p=WiseProject(project,read_only=True)
                    try:
                        ps=LocalWiseProjectService(p,APIConfig(command='serve',project_dir=project));ss=LocalSearchService(ps,None)
                        with sqlite3.connect(project/'metadata/internal.db') as db:
                            for model,index in ss.search_indices[ModalityType.VIDEO].items():
                                expected=db.execute('SELECT count(*) FROM vectors WHERE feature_extractor_id=?',(model,)).fetchone()[0]
                                assert expected==index.index.ntotal
                            assert db.execute('SELECT count(*) FROM media').fetchone()[0]==42
                    finally:p.db_engine.dispose();p.thumbsdb_engine.dispose()
                measure('validate_load',validate)
                row['wall_seconds']=time.perf_counter()-clock
                row['pipeline_seconds_excluding_sha']=row['wall_seconds']-row['sha_verification_seconds']
            row['resources']=resource.result;row['status']='completed';checkpoint()
        finally:
            # Only this trial's disposable directory; no production path accepted.
            assert work.is_relative_to(HERE/'replays') and work.name=='work'
            shutil.rmtree(work,ignore_errors=True)
            torch.mps.empty_cache()
    result['status']='completed';dump(out/'result.json',result)


def suite():
    import httpx
    import psutil
    audit()
    jobs=httpx.get('http://127.0.0.1:9670/mvp-api/jobs',timeout=20).json()['jobs']
    assert not [j for j in jobs if j['status'] not in ('completed','failed')], 'Active uploads'
    subprocess.run([sys.executable,str(ROOT/'scripts/manage.py'),'stop'],check=True)
    result=dict(status='running',started_epoch=time.time());dump(HERE/'replay-suite.json',result)
    try:
        for variant in ['current','original']:
            with (HERE/('replay-'+variant+'.log')).open('w') as log:
                proc=subprocess.Popen([sys.executable,str(Path(__file__)),'child','--variant',variant],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                start=time.monotonic();low=0
                while proc.poll() is None:
                    low=low+1 if psutil.virtual_memory().available<2*1024**3 else 0
                    if low>=6 or time.monotonic()-start>1500:
                        for sub in psutil.Process(proc.pid).children(recursive=True):sub.terminate()
                        proc.terminate();proc.wait(20)
                        raise RuntimeError('Replay memory/time guard')
                    time.sleep(.5)
                assert proc.returncode==0,f'{variant} replay failed; inspect log'
        result['status']='completed'
    except BaseException as exc:
        result['status']='failed';result['error']=repr(exc)
        raise
    finally:
        subprocess.run([sys.executable,str(ROOT/'scripts/manage.py'),'start'],check=True)
        result['service_restored']=False
        for _ in range(120):
            try:
                r=httpx.get('http://127.0.0.1:9670/mvp-api/actions/status',timeout=1)
                if r.status_code==200 and r.json()['model_loaded']:result['service_restored']=True;break
            except httpx.HTTPError:pass
            time.sleep(1)
        result['finished_epoch']=time.time();dump(HERE/'replay-suite.json',result);audit()
        assert result['service_restored']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['suite','child']);p.add_argument('--variant',choices=['current','original'])
    args=p.parse_args();suite() if args.mode=='suite' else child(args.variant)
