"""Timing probes around unchanged production code; never modifies live indexes.

Inclusive nested timers are reported separately, never added twice. Protocol and
Qwen outputs must still match the frozen production-verification fixture.
"""
import json
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT/'scripts')]
OUT = HERE/'sdpa-current-profile-20261028'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'


def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def worker():
    import extract_frames
    extract_frames.configure([])
    import torch
    import wise._main.extract_features as module
    from wise.feature.transformers_owlv2 import TransformersOWLv2FeatureExtractor as Owl, TransformersOWLv2Model as OwlModel
    from wise.feature.insightface import InsightFaceFeatureExtractor as Face
    from transformers import Owlv2ImageProcessorFast
    from insightface.model_zoo.scrfd import SCRFD
    from insightface.model_zoo.arcface_onnx import ArcFaceONNX
    from insightface.model_zoo.attribute import Attribute
    from wise.feature.store.faiss_store import FaissStore
    from wise.repository import VectorRepo, ThumbnailRepo
    stats={};stack=[];phase=['frames'];jobs=[0]

    @contextmanager
    def timer(label):
        key=phase[0]+'/'+label
        parent=[key,0.0];stack.append(parent);start=time.perf_counter()
        try:yield
        finally:
            elapsed=time.perf_counter()-start;stack.pop()
            if stack:stack[-1][1]+=elapsed
            r=stats.setdefault(key,{'calls':0,'inclusive_seconds':0.0,'exclusive_seconds':0.0})
            r['calls']+=1;r['inclusive_seconds']+=elapsed;r['exclusive_seconds']+=elapsed-parent[1]

    def wrap(cls, method, label):
        original=getattr(cls,method)
        def measured(*a,**kw):
            with timer(label):return original(*a,**kw)
        setattr(cls,method,measured)

    init=module.initialise_feature_extractors
    def initialize(*a,**kw):
        phase[0]='initialization'
        try:
            with timer('all'):return init(*a,**kw)
        finally:phase[0]='frames'
    module.initialise_feature_extractors=initialize
    factory=module.FeatureExtractorFactory
    def make(feature_id,*a,**kw):
        with timer('owl_factory' if 'owlv2' in feature_id else 'face_factory'):
            return factory(feature_id,*a,**kw)
    module.FeatureExtractorFactory=make
    for cls,method,label in [(Owl,'extract_image_features','owl_features'),(Owl,'construct_features','owl_metadata'),
        (OwlModel,'get_image_features','owl_forward_including_rpc'),(Owlv2ImageProcessorFast,'preprocess','owl_preprocess'),
        (Face,'extract_image_features','face_features'),(Face,'preprocess_image','face_preprocess'),
        (SCRFD,'detect','face_detect'),(ArcFaceONNX,'get','face_recognition'),(Attribute,'get','face_age_gender'),
        (VectorRepo,'create','db_vector_insert'),(ThumbnailRepo,'create','db_thumbnail_insert'),
        (FaissStore,'add','faiss_add'),(FaissStore,'save_current_shard','faiss_save')]:wrap(cls,method,label)
    loader=module.get_dataloader
    class TimedLoader:
        def __init__(self,wrapped):self.wrapped=wrapped
        def __iter__(self):
            iterator=iter(self.wrapped)
            while True:
                try:
                    with timer('decode_data_next'):value=next(iterator)
                except StopIteration:return
                yield value
    module.get_dataloader=lambda *a,**kw:TimedLoader(loader(*a,**kw))
    original=extract_frames.main
    def extract(*a,**kw):
        stats.clear();phase[0]='frames';start=time.perf_counter()
        try:return original(*a,**kw)
        finally:
            torch.mps.synchronize()
            save(f'frame-{jobs[0]}.json',{'main_seconds':time.perf_counter()-start,'timers':stats,
                'note':'Nested inclusive timings overlap; OWL forward includes RPC and MPS copies. No extra per-event GPU synchronization.'})
            jobs[0]+=1
    extract_frames.main=extract
    from frame_worker import worker_main
    raise SystemExit(worker_main())


def parent():
    import asyncio
    import subprocess
    import shutil
    import cv2
    import verify_production_owl_sdpa as replay
    from scripts.action_search import ActionEngine
    from wise.feature.qwen3vl import Qwen3VLEmbeddingFeatureExtractor as Qwen
    replay.OUT=OUT
    original_popen=subprocess.Popen
    def launch(command,*a,**kw):
        command=list(command)
        if len(command)>2 and Path(command[2]).name=='frame_worker.py':
            command=[command[0],'-u',str(Path(__file__).resolve()),'worker']
        return original_popen(command,*a,**kw)
    subprocess.Popen=launch
    qwen=[];active=[None]
    capture=cv2.VideoCapture
    def timed_call(label,fn,*a,**kw):
        if active[0] is None:return fn(*a,**kw)
        start=time.perf_counter()
        try:return fn(*a,**kw)
        finally:
            d=active[0].setdefault(label,{'seconds':0.,'calls':0})
            d['seconds']+=time.perf_counter()-start;d['calls']+=1
    class Capture:
        def __init__(self,*a,**kw):self.inner=timed_call('decode_open',capture,*a,**kw)
        def __getattr__(self,name):return getattr(self.inner,name)
        def get(self,*a):return timed_call('decode_get',self.inner.get,*a)
        def set(self,*a):return timed_call('decode_seek',self.inner.set,*a)
        def read(self,*a):return timed_call('decode_read',self.inner.read,*a)
        def release(self):return timed_call('decode_release',self.inner.release)
    cv2.VideoCapture=Capture
    feature=Qwen.extract_video_segment_features
    Qwen.extract_video_segment_features=lambda self,*a,**kw:timed_call('model_features',feature,self,*a,**kw)
    encode=ActionEngine.encode_segment
    def segment(self,media,window):
        row={'segment':window};active[0]=row;start=time.perf_counter()
        try:return encode(self,media,window)
        finally:
            row['total_seconds']=time.perf_counter()-start;qwen.append(row);active[0]=None
    ActionEngine.encode_segment=segment
    copy=shutil.copytree;copies=[];depth=[0]
    def copytree(source,destination,*a,**kw):
        outer=depth[0]==0;depth[0]+=1;start=time.perf_counter()
        try:return copy(source,destination,*a,**kw)
        finally:
            depth[0]-=1
            if outer:copies.append({'source':str(source),'destination':str(destination),'seconds':time.perf_counter()-start})
    shutil.copytree=copytree
    try:asyncio.run(replay.main())
    finally:
        save('parent-probes.json',{'qwen_segments':qwen,'copytree':copies})


if __name__=='__main__':
    if sys.argv[1:] and sys.argv[1]=='worker':worker()
    else:parent()
