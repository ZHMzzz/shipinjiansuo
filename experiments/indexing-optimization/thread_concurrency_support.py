"""Experiment-only scheduling, bounded CPU preparation and process-tree metrics."""
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
OUT=HERE/'thread-contention-20261028-rerun'


def prepare(engine,media,segment):
    """Exact production frame selection and CPU processor; no MPS transfers."""
    __import__("thread_telemetry").apply_threads()
    import cv2
    import numpy as np
    import torch
    from scripts.action_search import SETTINGS
    from wise.feature import qwen3vl as q
    extractor=engine.model()
    model,processor=extractor._model_and_processor
    path=Path(media['location'])/media['path']
    capture=cv2.VideoCapture(str(path))
    try:
        fps=capture.get(cv2.CAP_PROP_FPS);count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps<=0 or count<1:raise RuntimeError('Invalid video')
        first=min(count-1,max(0,math.ceil(segment['ts']*fps)))
        last=min(count-1,max(first,math.ceil(segment['te']*fps)-1))
        frames=[]
        for number in np.linspace(first,last,SETTINGS['frames']).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES,int(number));ok,frame=capture.read()
            if not ok:raise RuntimeError('Decode failed')
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        raw=np.stack(frames)
        tensor=torch.from_numpy(raw).permute(0,3,1,2)
        pil_frames=[q.TF.to_pil_image(f) for f in tensor]
        messages=[{'role':'user','content':[{'type':'video','video':pil_frames,'total_pixels':extractor.max_total_pixels}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        images,videos,kwargs=q.process_vision_info(messages,image_patch_size=q._IMAGE_PATCH_SIZE,
            return_video_metadata=True,return_video_kwargs=True)
        if videos is not None:
            videos,metadata=zip(*videos);videos,metadata=list(videos),list(metadata)
        else:metadata=None
        inputs=processor(text=[text],images=images,videos=videos,video_metadata=metadata,
                         return_tensors='pt',padding=True,do_resize=False,**kwargs)
        assert all(v.device.type=='cpu' for v in inputs.values())
        return inputs,hashlib.sha256(raw.tobytes()).hexdigest()
    finally:capture.release()


def forward(engine,inputs):
    __import__("thread_telemetry").apply_threads()
    import numpy as np
    import torch
    from scripts.qwen_projection import linear_video_projection
    extractor=engine.model();model,_=extractor._model_and_processor
    with torch.inference_mode(),linear_video_projection(extractor):
        inputs={k:v.to(extractor.DEVICE) for k,v in inputs.items()}
        output=model(**inputs)
        embedding=extractor._last_token_pool(output.last_hidden_state,inputs['attention_mask']).float()
        embedding=embedding/embedding.norm(dim=-1,keepdim=True)
        result=embedding.cpu().numpy()[0]
    torch.mps.synchronize()
    assert np.isfinite(result).all()
    return result.astype('float32')


class Schedule:
    def __init__(self,mode,engine,row):
        self.mode=mode;self.engine=engine;self.row=row
        self.executor=ThreadPoolExecutor(max_workers=1)
        self.stop=threading.Event();self.queue=__import__("thread_telemetry").TimedQueue.make(maxsize=2)
        self.future=None;self.segments=[];self.errors=[];self.events=[]
        self.origin=time.perf_counter()

    def event(self,name):self.events.append({'name':name,'seconds':time.perf_counter()-self.origin})

    def start(self,prepared,duration):
        from scripts.action_search import windows
        paths=list(Path(prepared).glob('*.mp4'));assert len(paths)==1
        self.media={'location':str(prepared),'path':paths[0].name,'duration':duration}
        self.segments=[{'media_id':0,'ts':a,'te':b} for a,b in windows(duration)]
        self.event('frame_start')
        if self.mode=='full':self.future=self.executor.submit(self.encode_all)
        elif self.mode=='cpu':self.future=self.executor.submit(self.produce)

    def encode_all(self):
        self.event('qwen_gpu_start')
        result=[]
        for segment in self.segments:
            if self.stop.is_set():raise RuntimeError('Experiment cancelled')
            result.append(self.engine.encode_segment(self.media,segment))
        self.event('qwen_gpu_end')
        return result

    def produce(self):
        self.event('cpu_prepare_start');hashes=[];peak=0
        try:
            for i,segment in enumerate(self.segments):
                if self.stop.is_set():return
                inputs,h=prepare(self.engine,self.media,segment);hashes.append(h)
                while not self.stop.is_set():
                    try:self.queue.put((i,inputs),timeout=.1);break
                    except queue.Full:pass
                peak=max(peak,self.queue.qsize())
            self.row['cpu_frame_hashes']=hashes
            self.row['cpu_queue_peak_items']=peak
            self.event('cpu_prepare_end')
        except BaseException as e:self.errors.append(e);raise

    def frame_done(self):
        self.event('frame_end')
        # This phase still precedes the unchanged create-index command: only
        # Qwen-vs-frame overlap differs, not index-vs-Qwen overlap.
        if self.mode=='full':
            started=time.monotonic()
            try:self.values=self.future.result(timeout=900)
            finally:__import__('thread_telemetry').emit('join_qwen_after_frames',started)

    def consume_cpu(self):
            self.values=[];self.event('qwen_gpu_start')
            for i in range(len(self.segments)):
                while True:
                    if self.errors:raise self.errors[0]
                    try:j,inputs=self.queue.get(timeout=.2);break
                    except queue.Empty:
                        if self.future.done():self.future.result();raise RuntimeError('Incomplete preparation')
                assert i==j
                self.values.append(forward(self.engine,inputs));del inputs
            self.future.result(timeout=30);self.event('qwen_gpu_end')

    def values_for(self,media):
        from scripts.action_search import windows
        assert [(x['ts'],x['te']) for x in self.segments]==list(windows(media['duration']))
        assert media['path']==self.media['path'] and media['location']==self.media['location']
        if self.mode=='cpu':self.consume_cpu()
        return self.values

    def close(self):
        self.stop.set();self.executor.shutdown(wait=True,cancel_futures=True)
        self.row['schedule_events']=self.events


class TreeSampler:
    """RSS/USS are process metrics, not additive with MPS allocator counters."""
    def snapshot(self):
        import psutil
        p=psutil.Process();procs=[p]+p.children(recursive=True)
        rss=0;uss=0;uss_ok=True;pids=[]
        for process in procs:
            try:
                rss+=process.memory_info().rss;pids.append(process.pid)
                try:uss+=process.memory_full_info().uss
                except (psutil.Error,AttributeError):uss_ok=False
            except psutil.Error:pass
        mem=psutil.virtual_memory();swap=psutil.swap_memory()
        return {'at':time.monotonic(),'rss':rss,'uss':uss if uss_ok else None,
                'available':mem.available,'swap':swap.used,'swap_out':swap.sout,'pids':pids}

    def __enter__(self):
        self.start=time.monotonic();self.rows=[self.snapshot()];self.stop=threading.Event()
        def run():
            while not self.stop.wait(.1):self.rows.append(self.snapshot())
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self

    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.rows.append(self.snapshot());self.end=time.monotonic()
        self.result={'scope':'benchmark parent plus recursive descendants; excludes unchanged live server',
            'interval_seconds':.1,'start':self.start,'end':self.end,
            'rss_peak_bytes':max(r['rss'] for r in self.rows),
            'uss_peak_bytes':max((r['uss'] for r in self.rows if r['uss'] is not None),default=None),
            'system_available_min_bytes':min(r['available'] for r in self.rows),
            'swap_used_before_bytes':self.rows[0]['swap'],'swap_used_peak_bytes':max(r['swap'] for r in self.rows),
            'swap_out_delta_bytes':max(r['swap_out'] for r in self.rows)-self.rows[0]['swap_out'],
            'samples':self.rows}


def start_mps_sampler():
    import torch
    stop=threading.Event()
    folder=OUT/'mps';folder.mkdir(parents=True,exist_ok=True)
    def run():
        with (folder/f'{os.getpid()}.jsonl').open('a',buffering=1) as f:
            while not stop.is_set():
                try:
                    f.write(json.dumps({'at':time.monotonic(),'pid':os.getpid(),
                         'allocated':torch.mps.current_allocated_memory(),
                         'driver':torch.mps.driver_allocated_memory()})+'\n')
                except Exception:pass
                stop.wait(.1)
    t=threading.Thread(target=run,daemon=True);t.start()
    def end():stop.set();t.join(timeout=2)
    return end


def install_launcher():
    original=subprocess.Popen
    def launch(command,*a,**kw):
        if isinstance(command,(list,tuple)) and len(command)>2:
            path=Path(command[2])
            if path.parent==ROOT/'scripts' and path.name in ('frame_worker.py','owl_sdpa.py'):
                command=[command[0],'-u',str(Path(__file__).resolve()),'child',*command[2:]]
        return original(command,*a,**kw)
    subprocess.Popen=launch


def query_checks(project,qwen_index,qwen_meta):
    import numpy as np
    import faiss
    import sqlite3
    from bench import OWL,FACE
    vectors=np.load(ROOT/'experiments/wise-upload-original/current-full/query-vectors.npy')
    labels=json.loads((ROOT/'experiments/wise-upload-original/design.json').read_text())
    vectors=vectors[:len(labels['queries'])]
    vectors=vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
    scores,ids=qwen_index.search(vectors.astype('float32'),10)
    text=[[{**qwen_meta['segments'][int(i)],'score':float(score)} for i,score in zip(ii,ss) if i>=0] for ii,ss in zip(ids,scores)]
    image_vectors=np.load(HERE/'query-baseline.npz');images={}
    ix=faiss.read_index(str(project/'store'/OWL/'index/video-IndexFlatIP.faiss'))
    with sqlite3.connect(project/'metadata/internal.db') as db:
        for key in image_vectors.files:
            ss,ii=ix.search(image_vectors[key],100)
            hits=[];video_hits=[];seen=set()
            for i,score in zip(ii[0],ss[0]):
                if i<0:continue
                m=db.execute('select v.media_id,v.timestamp,t.* from vectors v join vector_metadata_owlv2 t on t.vector_id=v.id where v.id=?',(int(i),)).fetchone()
                hit={'media_id':m[0],'time':m[1],'metadata':list(m[3:]),'score':float(score)}
                if len(hits)<10:hits.append(hit)
                if m[0] not in seen:video_hits.append(hit);seen.add(m[0])
            images[key]={'regions_top10':hits,'videos_top5':video_hits[:5]}
    del ix
    face=faiss.read_index(str(project/'store'/FACE/'index/video-IndexFlatIP.faiss'));people=[]
    with sqlite3.connect((ROOT/'data/people.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        queries=db.execute('select vector from people order by name').fetchall()
    with sqlite3.connect(project/'metadata/internal.db') as db:
        for (raw,) in queries:
            ss,ii=face.search(np.frombuffer(raw,dtype='float32').reshape(1,-1),face.ntotal);hits=[]
            for i,score in zip(ii[0],ss[0]):
                if i<0 or score<.45:continue
                m=db.execute('select media_id,timestamp from vectors where id=?',(int(i),)).fetchone()
                hits.append({'media_id':m[0],'time':m[1],'score':float(score)})
            people.append(hits)
    return {'qwen_top10':text,'owl':images,'registered_faces':people}


if __name__=='__main__':
    import runpy
    from thread_telemetry import install_hooks, wrap_input
    install_hooks();wrap_input()
    install_launcher();end=start_mps_sampler()
    target=sys.argv[2];sys.argv=sys.argv[2:]
    try:runpy.run_path(target,run_name='__main__')
    finally:end()
