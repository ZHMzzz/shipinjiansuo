"""Compare serial, full concurrent and CPU-only concurrent scheduling in isolated projects.

Only the frame extraction child is instrumented. Qwen suspension, restoration,
new-media checks, transcode and index commands use current production code.
The publisher validates/loads indexes and swaps only an experiment-local pointer.
"""
import asyncio
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT/'scripts')]
from bench import fingerprint, sha, dump, OWL, FACE
from replay import project_hashes, prepare_runtime, validate_and_load, safe_cleanup

OUT = HERE/'thread-contention-20261028-rerun'
if not OUT.resolve().is_relative_to(HERE):
    raise ValueError('Recheck output must be a direct child of the experiment folder')
PRODUCTION = True


def qwen_rows(project, model):
    with sqlite3.connect((project/'metadata/internal.db').as_uri()+'?mode=ro', uri=True) as db:
        return db.execute('SELECT id,media_id,timestamp,end_timestamp FROM vectors '
                          'WHERE feature_extractor_id=? ORDER BY id', (model,)).fetchall()


async def main():
    import numpy as np
    import torch
    from fastapi import UploadFile
    from scripts.ingestion import IngestionManager
    from scripts.action_search import ActionEngine, SETTINGS, read_snapshot, windows
    from scripts.native_qwen import QWEN, MANIFEST, validate_native_qwen
    from thread_concurrency_support import TreeSampler as ResourceSampler, Schedule, query_checks, install_launcher, start_mps_sampler
    from thread_telemetry import install_hooks, apply_threads, HostSampler, quiet_sample, CONTROL
    install_launcher()
    from scripts.frame_worker import FrameWorker
    import threading
    import hashlib
    import faiss
    shared = FrameWorker(ROOT, threading.Event())
    workers = {mode:shared for mode in ('serial','full','cpu')}

    OUT.mkdir(exist_ok=False)
    dump(CONTROL,{'threads':1,'trial':'startup'})
    install_hooks()
    before = fingerprint()
    dump(OUT/'production-before.json', before)
    base = Path(before['project'])
    assert (base/MANIFEST).exists() and not (base/'action-index').exists()
    with sqlite3.connect((ROOT/'data/ingestion.sqlite').as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        job = dict(db.execute("SELECT * FROM jobs WHERE media_id=40 AND status='completed' ORDER BY updated DESC LIMIT 1").fetchone())
    source = ROOT/'data/uploads'/job['id']/('source'+job['suffix'])
    assert sha(source) == job['sha256']
    old_meta, old_matrix, _ = read_snapshot(base, complete=True)
    old_rows = qwen_rows(base, QWEN)
    assert any(m['id']==40 for m in old_meta['media'])
    native_hashes = project_hashes(base/'store'/QWEN)
    report = {'status':'running', 'settings':SETTINGS, 'source_media_id':40,
              'source_sha256':job['sha256'], 'baseline_project':str(base),
              'before_media_count':len(old_meta['media']), 'before_segments':len(old_matrix),
              'trials':[], 'scope':'actual process_job on isolated roots; publication/load is local; no HTTP upload or queue wait'}
    dump(OUT/'result.json', report)
    end_mps = start_mps_sampler()
    runtime = prepare_runtime()
    engine = ActionEngine()
    original_encode = engine.encode_segment
    def actual_thread_encode(*args, **kwargs):
        apply_threads()
        return original_encode(*args, **kwargs)
    engine.encode_segment = actual_thread_encode
    work_root = OUT/'work'; work_root.mkdir()
    queries = ['视频中的场景和动作', '骑自行车', '户外慢跑', '红色安全帽写字']
    try:
        started = time.perf_counter()
        query_vectors = [engine.query_vector(q)[0] for q in queries]
        torch.mps.synchronize()
        report['qwen_load_and_query_warmup_seconds'] = time.perf_counter()-started
        # Balanced 3x3 factorial: each setting has 3 warm repetitions.
        cells=[(m,t) for t in (1,2,4) for m in ('serial','full','cpu')]
        configurations=[('serial-warmup',1)]
        for r in range(3):
            order=cells[r*3:]+cells[:r*3]
            if r==1:order=list(reversed(order))
            configurations += [(f'{m}-t{t}-r{r+1}',t) for m,t in order]
        report['production_entrypoint']=PRODUCTION
        for name, threads in configurations:
            owl,face='fast_fp16_weights','coreml'
            dump(CONTROL,{'threads':threads,'trial':name})
            apply_threads()
            assert fingerprint() == before, 'Production changed since freeze; stop comparison'
            shared_worker = workers[name.split('-')[0]]
            trial = OUT/name; trial.mkdir()
            work = Path(tempfile.mkdtemp(prefix='replay-', dir=work_root))
            row = {'threads':threads,'name':name,'mode':name.split('-')[0],'owl':owl,'face':face,'status':'running','stages':{}}
            schedule = Schedule(row['mode'],engine,row)
            report['trials'].append(row)
            try:
                (work/'logs').mkdir(); (work/'data').mkdir()
                initial = work/'projects/releases/initial/demo'
                initial.parent.mkdir(parents=True, exist_ok=True)
                import subprocess
                subprocess.run(['/bin/cp', '-cR', str(base), str(initial)], check=True)
                assert project_hashes(initial) == before['project_files']
                row['initial_copy_hashes_match'] = True
                (work/'projects/demo').symlink_to(initial)
                (work/'.media-runtime').symlink_to(ROOT/'.media-runtime', target_is_directory=True)
                shutil.copy2(ROOT/'wise_config.yaml',work/'wise_config.yaml')
                encoded = []

                async def build(stage, progress):
                    assert project_hashes(stage/'store'/QWEN) == native_hashes
                    assert not (stage/'.qwen-store-pending').exists()
                    row['qwen_store_restored_before_encoding'] = True
                    started = time.perf_counter()
                    catalog, records, vectors, missing = engine.prepare(stage)
                    assert len(missing)==1
                    assert len(records)==len(old_matrix)
                    np.testing.assert_array_equal(np.asarray(vectors), old_matrix)
                    assert qwen_rows(stage,QWEN)==old_rows
                    row['only_missing_media_encoded'] = [m['id'] for m in missing]
                    for media in missing:
                        values = (await asyncio.to_thread(schedule.values_for,media)) if row['mode']!='serial' else None
                        if row['mode']=='serial':schedule.event('qwen_gpu_start')
                        for number,(ts, te) in enumerate(windows(float(media['duration']))):
                            segment = {'media_id':media['id'],'ts':ts,'te':te}
                            vector = (await asyncio.to_thread(engine.encode_segment,media,segment)) if values is None else values[number]
                            vectors.append(vector)
                            records.append(segment); encoded.append(segment)
                        if row['mode']=='serial':schedule.event('qwen_gpu_end')
                    row['stages']['qwen_prepare_and_encode'] = time.perf_counter()-started
                    started = time.perf_counter()
                    await asyncio.to_thread(engine.finish,stage,catalog,records,vectors)
                    row['stages']['qwen_write_native'] = time.perf_counter()-started

                async def publish(stage):
                    started = time.perf_counter()
                    validate_native_qwen(stage)
                    row['loaded'] = validate_and_load(stage,runtime)
                    assert not (stage/'action-index').exists()
                    pointer = work/'projects/demo'; temp = work/'projects/next'
                    temp.symlink_to(stage); os.replace(temp,pointer)
                    row['stages']['validate_load_local_publish'] = time.perf_counter()-started

                manager = IngestionManager(work,publish,build)
                manager.loop = asyncio.get_running_loop()
                manager.frame_worker = shared_worker
                original_extract = shared_worker.extract
                def extract(args, log, on_line=None):
                    stage = Path(args[args.index('--project-dir')+1])
                    assert not (stage/'store'/QWEN).exists()
                    assert (stage/'.qwen-store-pending').exists()
                    row['qwen_hidden_during_frame_extraction'] = True
                    started = time.perf_counter()
                    duration=next(j for j in manager.jobs() if j['id']==queued['id'])['duration']
                    schedule.start(args[0],duration)
                    try:
                        result = original_extract(args, log, on_line)
                        row['frame_worker_seconds']=time.perf_counter()-started
                        row['worker'] = {**result, 'pid': shared_worker.process.pid}
                        schedule.frame_done()
                        return result
                    finally:
                        row['stages']['feature_phase'] = time.perf_counter()-started
                shared_worker.extract = extract
                original_run = manager.run
                def run_command(command, log, on_line=None):
                    kind = 'transcode' if Path(command[0]).name=='ffmpeg' else 'index'
                    entry=next((i for i,arg in enumerate(command)
                                if arg=='extract-features' or Path(arg).name=='extract_frames.py'),None)
                    if entry is not None:
                        kind='extract'
                        stage=Path(command[command.index('--project-dir')+1])
                        assert not (stage/'store'/QWEN).exists()
                        assert (stage/'.qwen-store-pending').exists()
                        row['qwen_hidden_during_frame_extraction']=True
                        if not PRODUCTION:
                            command=[sys.executable,'-u',str(HERE/'bench.py'),'child',
                                     '--inputs',command[entry+1],
                                     '--project',str(stage),'--metrics',str(trial/'extract-metrics.json'),
                                     '--owl',owl,'--face',face]
                    started=time.perf_counter()
                    try: return original_run(command,log,on_line)
                    finally: row['stages'][kind]=time.perf_counter()-started
                manager.run=run_command
                with source.open('rb') as stream:
                    queued=manager.enqueue(UploadFile(file=stream,filename='兼容复核.mp4'))['job']
                row['background_samples']=quiet_sample()
                print('START',name,'gpu_background', [x['gpu'].get('Device Utilization %') for x in row['background_samples']],flush=True)
                with HostSampler(trial/'host.jsonl'), ResourceSampler() as resources:
                    started=time.perf_counter()
                    await asyncio.to_thread(manager.process_job,queued)
                    row['process_job_seconds']=time.perf_counter()-started
                row['resources']=resources.result
                finished=next(j for j in manager.jobs() if j['id']==queued['id'])
                assert finished['status']=='completed', finished
                stage=(work/'projects/demo').resolve(); new_id=finished['media_id']
                meta,matrix,index=read_snapshot(stage,complete=True)
                old_positions=[i for i,s in enumerate(meta['segments']) if s['media_id']!=new_id]
                new_positions=[i for i,s in enumerate(meta['segments']) if s['media_id']==new_id]
                np.testing.assert_array_equal(matrix[old_positions],old_matrix)
                assert [r for r in qwen_rows(stage,QWEN) if r[1]!=new_id]==old_rows
                assert len(encoded)==len(new_positions)==len(list(windows(finished['duration'])))
                row.update(old_qwen_vectors_and_ids_exact=True,new_segments=len(encoded),
                           new_media_id=new_id,duration=finished['duration'],
                           no_action_index=not (stage/'action-index').exists())
                np.save(trial/'new-qwen.npy',matrix[new_positions])
                row['new_qwen_sha256']=sha(trial/'new-qwen.npy')
                row['qwen_queries']=[]
                for query,vector in zip(queries,query_vectors):
                    scores,indices=index.search(vector,min(10,index.ntotal))
                    row['qwen_queries'].append({'query':query,'hits':[
                        {**meta['segments'][int(i)],'score':float(score)}
                        for score,i in zip(scores[0],indices[0]) if i>=0]})
                with sqlite3.connect(stage/'metadata/internal.db') as db:
                    row['new_vector_counts']=dict(db.execute('SELECT feature_extractor_id,count(*) FROM vectors WHERE media_id=? GROUP BY 1',(new_id,)))
                assert set(row['new_vector_counts'])=={OWL,FACE,QWEN}
                row['frame_signatures'] = {}
                with sqlite3.connect(stage/'metadata/internal.db') as db:
                    for model, table in [(OWL,'vector_metadata_owlv2'), (FACE,'vector_metadata_insightface')]:
                        ids=db.execute('SELECT id,timestamp FROM vectors WHERE media_id=? AND feature_extractor_id=? ORDER BY id',(new_id,model)).fetchall()
                        metadata=db.execute(f'SELECT t.* FROM {table} t JOIN vectors v ON t.vector_id=v.id WHERE v.media_id=? ORDER BY t.vector_id',(new_id,)).fetchall()
                        shard=sorted((stage/'store'/model/'features').glob('video-*.faiss'))[-1]
                        idx=faiss.read_index(str(shard))
                        assert idx.ntotal==row['new_vector_counts'][model]
                        matrix_bytes=idx.index.reconstruct_n(0,idx.ntotal).tobytes()
                        row['frame_signatures'][model]={'vectors':hashlib.sha256(matrix_bytes).hexdigest(),
                            'metadata':hashlib.sha256(json.dumps([[t for _,t in ids],[list(r[1:]) for r in metadata]]).encode()).hexdigest()}
                row['retrieval']=query_checks(stage,index,meta)
                if len(report['trials'])>1:
                    first=report['trials'][0]
                    row['frame_outputs_exact_vs_serial']=row['frame_signatures']==first['frame_signatures']
                    current=np.load(trial/'new-qwen.npy'); reference=np.load(OUT/'serial-warmup/new-qwen.npy')
                    row['qwen_vectors_exact_vs_serial']=bool(np.array_equal(current,reference))
                    row['qwen_max_abs_diff']=float(np.max(np.abs(current-reference)))
                    row['retrieval_exact_vs_serial']=row['retrieval']==first['retrieval']
                else:
                    row.update(frame_outputs_exact_vs_serial=True,qwen_vectors_exact_vs_serial=True,retrieval_exact_vs_serial=True)
                for relative,expected in before['project_files'].items():
                    if relative.startswith('store/') and '/features/' in relative and not relative.startswith('store/'+QWEN+'/'):
                        assert sha(stage/relative)==expected
                row['old_frame_feature_shards_unchanged']=True
                if PRODUCTION:
                    log_text='\n'.join(log.read_text() for log in (work/'logs').glob('*.log'))
                    row['runtime_proof']={marker:marker in log_text for marker in [
                        'OWLv2 weights=float16 device=mps',
                        'OWLv2 processor=Owlv2ImageProcessorFast device=cpu',
                        "det_10g.onnx providers=['CoreMLExecutionProvider'",
                        "w600k_r50.onnx providers=['CoreMLExecutionProvider'",
                        "genderage.onnx providers=['CoreMLExecutionProvider'"]}
                    if name == 'serial-warmup':
                        assert all(v for k,v in row['runtime_proof'].items() if k != 'OWLv2 weights=float16 device=mps'), row['runtime_proof']
                        assert 'OWL_SDPA_BACKEND' in log_text and '"attention": "sdpa"' in log_text
                        row['official_sdpa_backend_confirmed'] = True
                row['status']='completed'
                print(name,round(row['process_job_seconds'],3),row['new_vector_counts'],flush=True)
            finally:
                schedule.close()
                shared_worker.extract = original_extract
                for log in (work/'logs').glob('*.log'):
                    shutil.copy2(log,trial/log.name)
                safe_cleanup(work,work_root)
                import gc
                gc.collect()
                dump(OUT/'result.json',report)
        if not PRODUCTION:
            a,b=report['trials']
            np.testing.assert_array_equal(np.load(OUT/'baseline/new-qwen.npy'),np.load(OUT/'optimized/new-qwen.npy'))
            assert a['qwen_queries']==b['qwen_queries']
            report.update(new_qwen_vectors_exact=True,qwen_query_top10_identical=True,
                          time_reduction=1-b['process_job_seconds']/a['process_job_seconds'])
        report['status']='completed'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        shared.close()
        end_mps()
        engine.unload()
        after=fingerprint();dump(OUT/'production-after.json',after)
        report['production_unchanged']=after==before
        if after!=before:report['status']='invalid-production-changed'
        dump(OUT/'result.json',report)


if __name__=='__main__':
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    asyncio.run(main())
