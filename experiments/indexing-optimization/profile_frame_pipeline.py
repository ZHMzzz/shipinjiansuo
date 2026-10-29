"""Read-only production-input replay; nested timers never summed twice."""
import time
ENTRY=time.perf_counter()
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import contextmanager
import bench

OUT=bench.OUT/'frame-breakdown-20261028'


def child(output, argv):
    import logging
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(levelname)s %(message)s')
    from frame_acceleration import enable
    enable()
    import torch
    import wise._main.extract_features as extract
    from wise.feature.transformers_owlv2 import TransformersOWLv2FeatureExtractor as Owl, TransformersOWLv2Model as OwlModel
    from wise.feature.insightface import InsightFaceFeatureExtractor as Face
    from wise.feature.store.faiss_store import FaissStore
    from transformers import Owlv2ImageProcessorFast
    from transformers.models.owlv2.modeling_owlv2 import Owlv2VisionTransformer
    from insightface.model_zoo.scrfd import SCRFD
    from insightface.model_zoo.retinaface import RetinaFace
    from insightface.model_zoo.arcface_onnx import ArcFaceONNX
    from insightface.model_zoo.attribute import Attribute
    from insightface.model_zoo import model_zoo
    from wise.repository import VectorRepo,ThumbnailRepo
    stats={}; stack=[]; phase=['frames']; state={'status':'running'}

    @contextmanager
    def timer(label,gpu=False):
        label=phase[0]+'/'+label
        if any(s[0]==label for s in stack):
            yield;return
        if gpu:torch.mps.synchronize()
        current=[label,0.0];stack.append(current);t=time.perf_counter()
        try:yield
        finally:
            if gpu:torch.mps.synchronize()
            dt=time.perf_counter()-t;stack.pop()
            if stack:stack[-1][1]+=dt
            row=stats.setdefault(label,{'calls':0,'inclusive_seconds':0.,'exclusive_seconds':0.})
            row['calls']+=1;row['inclusive_seconds']+=dt;row['exclusive_seconds']+=dt-current[1]

    def wrap(owner,name,label,gpu=False):
        original=getattr(owner,name)
        def wrapped(*a,**kw):
            with timer(label,gpu):return original(*a,**kw)
        setattr(owner,name,wrapped)

    init=extract.initialise_feature_extractors
    def init_wrapper(*a,**kw):
        phase[0]='initialization'
        try:
            with timer('all',True):return init(*a,**kw)
        finally:phase[0]='frames'
    extract.initialise_feature_extractors=init_wrapper
    factory=extract.FeatureExtractorFactory
    def factory_wrapper(feature_id,*a,**kw):
        label='owl_factory' if 'owlv2' in feature_id else 'face_factory'
        with timer(label,True):return factory(feature_id,*a,**kw)
    extract.FeatureExtractorFactory=factory_wrapper
    session=model_zoo.PickableInferenceSession.__init__
    def session_wrapper(self,model_path,*a,**kw):
        with timer('face_session_'+Path(model_path).name):return session(self,model_path,*a,**kw)
    model_zoo.PickableInferenceSession.__init__=session_wrapper
    wrap(Owl,'extract_image_features','owl_features',True)
    wrap(Owl,'construct_features','owl_metadata')
    wrap(OwlModel,'get_image_features','owl_forward',True)
    wrap(Owlv2VisionTransformer,'forward','owl_vision',True)
    wrap(Owlv2ImageProcessorFast,'preprocess','owl_preprocess')
    wrap(Face,'extract_image_features','face_features')
    wrap(Face,'preprocess_image','face_preprocess')
    wrap(SCRFD,'detect','face_detect')
    wrap(RetinaFace,'detect','face_detect')
    wrap(ArcFaceONNX,'get','face_recognition')
    wrap(Attribute,'get','face_age_gender')
    wrap(VectorRepo,'create','db_vector_insert')
    wrap(ThumbnailRepo,'create','db_thumbnail_insert')
    wrap(FaissStore,'add','faiss_add')
    wrap(FaissStore,'save_current_shard','faiss_save')
    loader=extract.get_dataloader
    class TimedLoader:
        def __init__(self,wrapped):self.wrapped=wrapped
        def __iter__(self):
            iterator=iter(self.wrapped)
            while True:
                try:
                    with timer('data_next'):value=next(iterator)
                except StopIteration:return
                yield value
    extract.get_dataloader=lambda *a,**kw:TimedLoader(loader(*a,**kw))
    state['import_and_probe_setup_seconds']=time.perf_counter()-ENTRY
    started=time.perf_counter()
    try:
        extract.main(['extract-features',*argv]);state['status']='completed'
    finally:
        torch.mps.synchronize()
        state.update(main_seconds=time.perf_counter()-started,timers=stats,
                     total_process_body_seconds=time.perf_counter()-ENTRY,
                     note='Inclusive parent/child timers overlap; use exclusive sums or top-level events. Synchronization and probe setup add overhead.')
        bench.dump(output,state)


def main():
    from native_qwen import suspend_native_store
    from action_search import media_catalog
    import numpy as np
    import faiss
    faiss.omp_set_num_threads(1)
    OUT.mkdir(exist_ok=False);before=bench.fingerprint();bench.dump(OUT/'production-before.json',before)
    base=Path(before['project']);media=next(m for m in media_catalog(base) if m['id']==40)
    source=Path(media['location'])/media['path'];source=source if source.is_absolute() else bench.ROOT/source
    source_hash=bench.sha(source)
    result={'status':'running','source_media':media,'input_sha256':source_hash,'trials':[],
            'scope':'Frame extraction child only, same prepared 35.44s video, full project clone, original production acceleration, no Qwen/re-index/publication.'}
    work_root=OUT/'work';work_root.mkdir()
    vectors={}
    try:
        for mode in ('baseline','profile'):
            assert bench.fingerprint()==before
            work=Path(tempfile.mkdtemp(prefix='frame-',dir=work_root)); project=work/'project';inputs=work/'inputs';inputs.mkdir()
            (inputs/source.name).symlink_to(source.resolve());shutil.copy2(bench.ROOT/'wise_config.yaml',work/'wise_config.yaml')
            subprocess.run(['/bin/cp','-cR',str(base),str(project)],check=True)
            args=[str(inputs),'--project-dir',str(project),'--num-workers','0','--yes']
            command=([sys.executable,'-u',str(bench.ROOT/'scripts/extract_frames.py'),*args] if mode=='baseline' else
                     [sys.executable,'-u',__file__,'child',str(OUT/'profile.json'),*args])
            try:
                with suspend_native_store(project):
                    t=time.perf_counter()
                    with (OUT/f'{mode}.log').open('w') as log:
                        ret=subprocess.run(command,cwd=work,env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'},stdout=log,stderr=subprocess.STDOUT,timeout=600)
                    elapsed=time.perf_counter()-t
                assert ret.returncode==0,f'{mode} exit {ret.returncode}'
                with sqlite3.connect(project/'metadata/internal.db') as db:
                    new_id=db.execute('select max(id) from media').fetchone()[0]
                    counts=dict(db.execute('select feature_extractor_id,count(*) from vectors where media_id=? group by 1',(new_id,)))
                    assert set(counts)=={bench.OWL,bench.FACE}
                    metadata={}
                    for table in ('vector_metadata_owlv2','vector_metadata_insightface'):
                        metadata[table]=db.execute(f'SELECT t.* FROM {table} t JOIN vectors v ON t.vector_id=v.id WHERE v.media_id=? ORDER BY t.vector_id',(new_id,)).fetchall()
                matrices={}
                for model in counts:
                    shard=sorted((project/'store'/model/'features').glob('video-*.faiss'))[-1]
                    ix=faiss.read_index(str(shard));ids=faiss.vector_to_array(ix.id_map)
                    assert len(ids)==counts[model]
                    matrices[model]=ix.index.reconstruct_n(0,ix.ntotal)
                vectors[mode]=(matrices,metadata)
                result['trials'].append({'mode':mode,'wall_seconds':elapsed,'counts':counts,'new_media_id':new_id})
                bench.dump(OUT/'result.json',result);print(mode,elapsed,counts,flush=True)
            finally:
                assert work.parent==work_root and work.name.startswith('frame-');shutil.rmtree(work)
        result['outputs_exact']={m:bool(np.array_equal(vectors['baseline'][0][m],vectors['profile'][0][m])) for m in vectors['baseline'][0]}
        result['metadata_exact']=vectors['baseline'][1]==vectors['profile'][1]
        assert all(result['outputs_exact'].values()) and result['metadata_exact']
        assert bench.sha(source)==source_hash
        result['status']='completed'
    finally:
        result['production_unchanged']=bench.fingerprint()==before
        assert result['production_unchanged']
        bench.dump(OUT/'result.json',result)

if __name__=='__main__':
    child(sys.argv[2],sys.argv[3:]) if len(sys.argv)>1 and sys.argv[1]=='child' else main()
