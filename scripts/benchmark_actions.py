"""Bounded Qwen3-VL-Embedding-2B research benchmark using WISE's extractor.

Video filenames/labels are never passed to the encoder. Reports contain raw
rankings; no class-based reranking. A parent process samples RSS and system RAM.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/action-search'


def child(args):
    import cv2
    import numpy as np
    import psutil
    import torch
    from wise.feature.qwen3vl import Qwen3VLEmbeddingFeatureExtractor as Extractor
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.mps.set_per_process_memory_fraction(0.55)
    manifest = json.loads((EXP/'manifest.json').read_text())
    items = manifest['clips'][:args.limit] if args.limit else manifest['clips']
    report = {'model':'Qwen/Qwen3-VL-Embedding-2B', 'device':'mps', 'dtype':args.dtype,
              'max_total_pixels':args.pixels, 'frames_per_segment':16,
              'clips':[], 'queries':[], 'status':'running'}
    peaks = {'mps_allocated_bytes':0, 'mps_driver_bytes':0}
    done = threading.Event()
    def monitor_metal():
        while not done.wait(.05):
            peaks['mps_allocated_bytes'] = max(peaks['mps_allocated_bytes'], torch.mps.current_allocated_memory())
            peaks['mps_driver_bytes'] = max(peaks['mps_driver_bytes'], torch.mps.driver_allocated_memory())
    threading.Thread(target=monitor_metal, daemon=True).start()
    target = EXP/(args.name+'.json')
    def checkpoint():
        report['sampled_metal_peaks'] = dict(peaks)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    def memory():
        torch.mps.synchronize()
        return {'rss_bytes':psutil.Process().memory_info().rss,
                'mps_allocated_bytes':torch.mps.current_allocated_memory(),
                'mps_driver_bytes':torch.mps.driver_allocated_memory()}
    checkpoint()
    extractor = Extractor('hf/Qwen/Qwen3-VL-Embedding/2B', device='mps', warmup=False,
        config=Extractor.Config(device='mps', warmup=False, compile=False,
                                dtype=args.dtype, max_total_pixels=args.pixels))
    _ = extractor._model
    input_shape = {}
    def capture_shape(module, inputs, kwargs):
        if kwargs.get('video_grid_thw') is not None:
            input_shape['video_grid_thw'] = kwargs['video_grid_thw'].detach().cpu().tolist()
        input_shape['tokens'] = int(kwargs['input_ids'].shape[-1])
    extractor._model.register_forward_pre_hook(capture_shape, with_kwargs=True)
    report['load_seconds'] = time.perf_counter()-started
    report['loaded_memory'] = memory(); checkpoint()
    print('MODEL_LOADED', report['load_seconds'], report['loaded_memory'], flush=True)
    vectors=[]
    for item in items:
        t=time.perf_counter()
        cap=cv2.VideoCapture(str(EXP/item['file']))
        fps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or total < 1: raise RuntimeError(f"Cannot decode {item['id']}")
        frames=[]
        for index in np.linspace(0, total-1, 16).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index)); ok, frame=cap.read()
            if not ok: raise RuntimeError(f"Frame decode failed {item['id']}")
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release()
        tensor=torch.from_numpy(np.stack(frames)).permute(0,3,1,2)
        decode_s=time.perf_counter()-t
        t=time.perf_counter()
        vec=extractor.extract_video_segment_features(tensor).vectors
        torch.mps.synchronize()
        seconds=time.perf_counter()-t
        if not np.isfinite(vec).all(): raise RuntimeError('Non-finite video embedding')
        vectors.append(vec[0])
        entry={**item, 'duration':total/fps, 'decode_seconds':decode_s,
               'embedding_seconds':seconds, 'memory':memory(), 'model_input':dict(input_shape)}
        report['clips'].append(entry); checkpoint()
        print('CLIP',item['id'],round(seconds,3),entry['memory'],flush=True)
    matrix=np.stack(vectors)
    np.save(EXP/(args.name+'-vectors.npy'),matrix)
    import faiss
    t=time.perf_counter();index=faiss.IndexFlatIP(matrix.shape[1]);index.add(matrix)
    report['index_seconds']=time.perf_counter()-t
    faiss.write_index(index,str(EXP/(args.name+'.faiss')))
    for query in manifest['queries']:
        t=time.perf_counter()
        features=extractor.extract_text_features([query['text']]);torch.mps.synchronize()
        encode_s=time.perf_counter()-t
        if not np.isfinite(features).all(): raise RuntimeError('Non-finite query embedding')
        t=time.perf_counter();scores,ids=index.search(features,len(items));search_s=time.perf_counter()-t
        ranking=[{'id':items[int(i)]['id'],'score':float(s)} for i,s in zip(ids[0],scores[0])]
        result={**query,'encode_seconds':encode_s,'search_seconds':search_s,'ranking':ranking,'memory':memory()}
        report['queries'].append(result);checkpoint()
        print('QUERY',query['text'],ranking[:3],flush=True)
    report.update(status='completed', wall_seconds=time.perf_counter()-started,
                  total_video_seconds=sum(c['duration'] for c in report['clips']),
                  total_embedding_seconds=sum(c['embedding_seconds'] for c in report['clips']))
    checkpoint()
    done.set()


def parent(args):
    import psutil
    EXP.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(Path(__file__).resolve()),'--child','--name',args.name,
         '--dtype',args.dtype,'--pixels',str(args.pixels),'--limit',str(args.limit)]
    started=time.monotonic();samples=[];reason=None
    with (EXP/(args.name+'.log')).open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        handle=psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                rss=handle.memory_info().rss
                ram=psutil.virtual_memory();swap=psutil.swap_memory()
                samples.append({'seconds':round(time.monotonic()-started,2),'rss':rss,
                                'system_available':ram.available,'swap_used':swap.used})
                if rss>18*1024**3 or ram.available<2*1024**3:
                    reason='memory_guard'
                elif time.monotonic()-started>args.timeout:
                    reason='timeout'
                if reason:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    break
            except psutil.NoSuchProcess: break
            time.sleep(.2)
    code=proc.wait()
    metrics={'exit_code':code,'stop_reason':reason,'wall_seconds':time.monotonic()-started,
             'peak_rss_bytes':max((s['rss'] for s in samples),default=0),
             'minimum_system_available_bytes':min((s['system_available'] for s in samples),default=0),
             'samples':samples}
    (EXP/(args.name+'-resources.json')).write_text(json.dumps(metrics,indent=2))
    result_path=EXP/(args.name+'.json')
    if code and result_path.exists():
        result=json.loads(result_path.read_text())
        result.update(status='failed',stop_reason=reason or 'process_error',exit_code=code)
        result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if k!='samples'},indent=2),flush=True)
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true')
    p.add_argument('--name',default='qwen-fp16');p.add_argument('--dtype',default='float16')
    p.add_argument('--pixels',type=int,default=16*256*256)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--timeout',type=int,default=900)
    args=p.parse_args()
    if args.child: child(args)
    else: sys.exit(parent(args))
