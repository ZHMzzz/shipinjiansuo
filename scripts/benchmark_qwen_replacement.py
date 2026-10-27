"""Isolated replacement evaluation. Never writes to production indexes/configuration.

Run prepare/live, then siglip and qwen in separate processes. Labels are only
used after scoring. Query images are never inserted into the sampled corpus.
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/qwen-replacement'
CLIP = 'mlfoundations/open_clip/ViT-B-16-SigLIP2-256/webli'
MAX_PIXELS = 262144


def dump(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT/name
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(p)


def fingerprint():
    p = (ROOT/'projects/demo').resolve()
    files = sorted(x for x in p.rglob('*') if x.is_file() and
                   (x.suffix in ('.faiss', '.npy') or x.name in ('internal.db', 'metadata.json')))
    return {'project': str(p), 'files': {str(f.relative_to(p)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}


def load_design():
    d = json.loads((OUT/'design.json').read_text())
    assert fingerprint() == d['production_before'], 'Production snapshot changed; do not compare different libraries'
    return d


def prepare():
    from action_search import media_catalog
    project = (ROOT/'projects/demo').resolve()
    media = media_catalog(project)
    queries = []
    sources = [(7,'jogging-OV-CGjSLNbg_000053_000063'),
               (3,'coughing-6XvsLPDioVA_000000_000010'),
               (6,'frying-vegetables-hxK9mej0_zw_000086_000096'),
               (9,'shouting-84cxLPOEnGg_000011_000021'),
               (1,'backpack-street'),(2,'backpack-walking')]
    for mid, name in sources:
        queries.append(dict(id=f'screenshot_{mid}', category='screenshot', image=f'data/queries/{name}-preview.jpg',
                            text='', expected_media=[mid], known_source_seconds=2.25 if mid==7 else None))
    bag='data/queries/backpack-independent.jpg'
    queries.append(dict(id='bag_image',category='appearance',image=bag,text='',expected_media=[1,2,17]))
    for i, text in enumerate(['在街道上','一个人背着背包走在街道上','寻找图中这类背包，被人背着走在户外街道上']):
        queries.append(dict(id=f'mixed_street_{i}',category='mixed',image=bag,text=text,expected_media=[1]))
    for i, text in enumerate(['在室内','一个人背着背包在室内行走','寻找图中这类背包，被人背着在室内行走']):
        queries.append(dict(id=f'mixed_indoor_{i}',category='mixed',image=bag,text=text,expected_media=[2,17],
                            valid_ranges={'17':[8,16]}))
    for i,(text,expected) in enumerate([('一个人背着背包走路',[1,2,17]),('一个人在跑步',[7,8]),('锅里正在炒蔬菜',[5,6])]):
        queries.append(dict(id=f'text_{i}',category='text',image=None,text=text,expected_media=expected))
    frames=[]
    folder=OUT/'frames';folder.mkdir(exist_ok=True)
    for m in media:
        source=Path(m['location'])/m['path']
        if not source.is_absolute():source=ROOT/source
        cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert fps>0 and count>0
        for ts in np.arange(.5,max(.51,m['duration']),2.):
            n=min(count-1,round(float(ts)*fps));cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,f=cap.read();assert ok
            dest=folder/f'{m["id"]}-{n}.png';assert cv2.imwrite(str(dest),f)
            frames.append(dict(media_id=m['id'],ts=n/fps,image=str(dest.relative_to(ROOT))))
        cap.release()
    dump('design.json',dict(production_before=fingerprint(),media=media,frames=frames,queries=queries,
         image_max_pixels=MAX_PIXELS,frame_sampling='0.5s then every 2s; clamp to last decodable frame',
         qwen_frame_format='official system: Represent the user\'s input.; image before text',
         qwen_segment_format='legacy WISE user-only, matching existing video embeddings',
         repeats=3,notes='Known duplicate video 17 contains backpack footage at 8–16s. No same-item identity labels.'))
    print('PREPARED',len(media),len(frames),len(queries),flush=True)


def good(hit,q):
    if hit['media_id'] not in q['expected_media']:return False
    span=q.get('valid_ranges',{}).get(str(hit['media_id']))
    if not span:return True
    if 'te' in hit:return max(0,min(hit['te'],span[1])-max(hit['ts'],span[0]))/(hit['te']-hit['ts'])>=.5
    return span[0]<=hit['ts']<span[1]


def annotate(hits,q):
    first=next((i+1 for i,h in enumerate(hits) if good(h,q)),None)
    return {'first_relevant_rank':first,'top1':first==1,'top5':first is not None and first<=5}


def ranking(vector,matrix,records):
    started=time.perf_counter();scores=matrix@vector;order=np.argsort(-scores,kind='stable')
    hits=[];seen=set()
    for idx in order:
        item=records[int(idx)]
        if item['media_id'] in seen:continue
        seen.add(item['media_id'])
        hits.append({k:v for k,v in item.items() if k!='image'}|{'score':float(scores[idx])})
    return hits,time.perf_counter()-started


def live():
    import requests
    d=load_design();report={'queries':[]}
    for q in d['queries']:
        times=[];hits=[]
        for repeat in range(3):
            params=dict(search_in='video',feature_extractor_id=CLIP,start=0,end=100,thumbnails_to_send=0,add_prefix='false')
            if q['text']:params['text_queries']=q['text']
            files={'image_file_queries':('query.jpg',(ROOT/q['image']).read_bytes(),'image/jpeg')} if q['image'] else None
            t=time.perf_counter();r=requests.post('http://127.0.0.1:9670/demo/search',params=params,files=files,timeout=120)
            r.raise_for_status();times.append(time.perf_counter()-t)
            data=r.json()['video_results'];hits=[];seen=set()
            for h in data['unmerged_windows']:
                mid=int(h['media_id'])
                if mid in seen:continue
                seen.add(mid);hits.append(dict(media_id=mid,ts=h.get('thumbnail_ts',h['ts']),score=h['distance']))
        report['queries'].append(dict(id=q['id'],category=q['category'],http_seconds=times,ranking=hits,**annotate(hits,q)))
        dump('live-siglip.json',report);print('LIVE',q['id'],report['queries'][-1]['first_relevant_rank'],flush=True)


class QwenEncoder:
    def __init__(self):
        from wise.feature.qwen3vl import Qwen3VLEmbeddingFeatureExtractor as Extractor
        self.ext=Extractor('hf/Qwen/Qwen3-VL-Embedding/2B',device='mps',warmup=False,
            config=Extractor.Config(device='mps',compile=False,warmup=False,dtype='float16',max_total_pixels=1048576))
        self.model,self.processor=self.ext._model_and_processor

    def encode(self,image=None,text='',legacy=False):
        import torch
        from qwen_vl_utils import process_vision_info
        content=[]
        if image is not None:content.append(dict(type='image',image=image,min_pixels=4096,max_pixels=MAX_PIXELS))
        if text:content.append(dict(type='text',text=text))
        messages=[] if legacy else [dict(role='system',content=[dict(type='text',text="Represent the user's input.")])]
        messages.append(dict(role='user',content=content))
        prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        images,_,kwargs=process_vision_info(messages,image_patch_size=16,return_video_metadata=True,return_video_kwargs=True)
        inputs=self.processor(text=[prompt],images=images,return_tensors='pt',padding=True,do_resize=False,**kwargs)
        inputs={k:v.to('mps') for k,v in inputs.items()}
        with torch.inference_mode():
            output=self.model(**inputs)
            v=self.ext._last_token_pool(output.last_hidden_state,inputs['attention_mask']).float()
            v=v/v.norm(dim=-1,keepdim=True)
        self.tokens=int(inputs['input_ids'].shape[-1])
        return v.cpu().numpy()[0]


class SiglipEncoder:
    def __init__(self):
        from wise.feature.mlfoundation_openclip import MlfoundationOpenClipFeatureExtractor as Extractor
        self.ext=Extractor(CLIP,device='mps',warmup=False,config=Extractor.Config(device='mps',compile=False,warmup=False))
        _=self.ext.model.model;_=self.ext.processor;_=self.ext.tokenizer

    def encode(self,image=None,text='',legacy=False):
        vecs=[]
        if image is not None:vecs.append(self.ext.extract_image_features(self.ext.preprocess_image([image]))[0].vectors[0])
        if text:vecs.append(2*self.ext.extract_text_features([text])[0])
        v=sum(vecs);return v/np.linalg.norm(v)


def reference_check(encoder,q):
    """Use the fixed official formatting, preprocessing and pooling with shared weights."""
    import torch
    spec=importlib.util.spec_from_file_location('official_embedding',OUT/'reference/qwen3_vl_embedding.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    ref=mod.Qwen3VLEmbedder.__new__(mod.Qwen3VLEmbedder)
    ref.model=encoder.model;ref.processor=encoder.processor
    for k,v in dict(max_length=8192,min_pixels=4096,max_pixels=MAX_PIXELS,total_pixels=1048576,
                    fps=1,num_frames=16,max_frames=16,default_instruction="Represent the user's input.").items():setattr(ref,k,v)
    im=Image.open(ROOT/q['image']).convert('RGB')
    own=encoder.encode(im,q['text'])
    with torch.inference_mode():official=ref.process([dict(image=im,text=q['text'])]).float().cpu().numpy()[0]
    official/=np.linalg.norm(official)
    result=dict(cosine=float(own@official),max_abs_difference=float(np.max(np.abs(own-official))))
    assert result['cosine']>.999 and result['max_abs_difference']<.002,result
    return result


def run(kind):
    import psutil
    import torch
    from action_reranker import ResourceSampler
    d=load_design();torch.set_num_threads(1);torch.mps.set_per_process_memory_fraction(.55)
    report=dict(kind=kind,queries=[],frame_timings=[],status='running')
    def guard():
        if psutil.virtual_memory().available<2*1024**3:raise RuntimeError('Available RAM below 2 GiB')
    with ResourceSampler() as resources:
        guard();t=time.perf_counter();encoder=QwenEncoder() if kind=='qwen' else SiglipEncoder()
        torch.mps.synchronize();report['load_seconds']=time.perf_counter()-t
        if kind=='qwen':report['official_reference_check']=reference_check(encoder,next(q for q in d['queries'] if q['category']=='mixed'))
        print('LOADED',kind,report['load_seconds'],flush=True)
        encoder.encode(Image.open(ROOT/d['frames'][0]['image']).convert('RGB'),'warmup')
        matrix=[]
        for i,f in enumerate(d['frames']):
            guard();im=Image.open(ROOT/f['image']).convert('RGB');t=time.perf_counter();v=encoder.encode(im)
            torch.mps.synchronize();report['frame_timings'].append(time.perf_counter()-t)
            assert np.isfinite(v).all() and abs(np.linalg.norm(v)-1)<1e-4
            matrix.append(v)
            if i%10==0:print('FRAME',kind,i+1,'/',len(d['frames']),flush=True);dump(kind+'.json',report)
        matrix=np.stack(matrix);np.save(OUT/(kind+'-frames.npy'),matrix)
        if kind=='qwen':
            p=Path(d['production_before']['project']);segments=json.loads((p/'action-index/metadata.json').read_text())['segments']
            seg_matrix=np.load(p/'action-index/vectors.npy')
        for q in d['queries']:
            guard();im=Image.open(ROOT/q['image']).convert('RGB') if q['image'] else None
            row=dict(id=q['id'],category=q['category'],encode_seconds=[])
            for _ in range(d['repeats']):
                t=time.perf_counter();v=encoder.encode(im,q['text']);torch.mps.synchronize()
                row['encode_seconds'].append(time.perf_counter()-t)
            assert np.isfinite(v).all()
            hits,seconds=ranking(v,matrix,d['frames']);row.update(ranking=hits,search_seconds=seconds,**annotate(hits,q))
            if kind=='qwen':
                timings=[]
                for _ in range(d['repeats']):
                    t=time.perf_counter();legacy=encoder.encode(im,q['text'],legacy=True);torch.mps.synchronize();timings.append(time.perf_counter()-t)
                sh,st=ranking(legacy,seg_matrix,segments)
                row['existing_segments']=dict(encode_seconds=timings,ranking=sh,search_seconds=st,**annotate(sh,q))
            report['queries'].append(row);dump(kind+'.json',report)
            print('QUERY',kind,q['id'],row['first_relevant_rank'],row.get('existing_segments',{}).get('first_relevant_rank'),flush=True)
    report['resources']=resources.result;report['status']='completed';dump(kind+'.json',report)


def refine(image,hits,media):
    def desc(frame):
        a=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(160,90)).astype(np.float32).reshape(-1)
        a-=a.mean();return a/max(float(np.linalg.norm(a)),1e-8)
    target=desc(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR));out=[]
    for hit in hits[:5]:
        m=next(m for m in media if m['id']==hit['media_id']);p=Path(m['location'])/m['path']
        if not p.is_absolute():p=ROOT/p
        cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS);stride=max(1,round(fps/10));best=(-2,0);idx=0
        while idx/fps<60:
            ok,f=cap.read()
            if not ok:break
            if idx%stride==0:
                s=float(target@desc(f))
                if s>best[0]:best=(s,idx/fps)
            idx+=1
        cap.release();out.append(dict(media_id=hit['media_id'],score=best[0],ts=best[1]))
    return sorted(out,key=lambda x:-x['score'])


def finish():
    d=load_design();reports={k:json.loads((OUT/(k+'.json')).read_text()) for k in ['live-siglip','siglip','qwen']}
    preview_times={Path(m['file']).stem: m['query_preview_time'] for m in json.loads((ROOT/'data/manifest.json').read_text())}
    summaries={};traces=[]
    for method in ['live-siglip','siglip','qwen','qwen-existing-segments']:
        rows=reports['qwen' if method=='qwen-existing-segments' else method]['queries'];groups={}
        for row,q in zip(rows,d['queries']):
            r=row['existing_segments'] if method=='qwen-existing-segments' else row
            group=groups.setdefault(q['category'],dict(count=0,top1=0,top5=0,seconds=[]))
            group['count']+=1;group['top1']+=int(r['top1']);group['top5']+=int(r['top5'])
            group['seconds'].extend(r.get('encode_seconds',r.get('http_seconds',[])))
            if q['category']=='screenshot':
                t=time.perf_counter();hits=refine(Image.open(ROOT/q['image']).convert('RGB'),r['ranking'],d['media'])
                expected_ts=preview_times[Path(q['image']).stem.removesuffix('-preview')]
                error=abs(hits[0]['ts']-expected_ts) if good(hits[0],q) else None
                traces.append(dict(method=method,id=q['id'],seconds=time.perf_counter()-t,ranking=hits,
                                   source_seconds=expected_ts,absolute_time_error=error,**annotate(hits,q)))
        for g in groups.values():
            times=g.pop('seconds');g.update(median_seconds=float(np.median(times)),min_seconds=min(times),max_seconds=max(times))
        summaries[method]=groups
    dump('summary.json',summaries);dump('trace-refinement.json',traces)
    dump('verification.json',dict(production_unchanged=fingerprint()==d['production_before'],production_after=fingerprint(),
         official_reference_check=reports['qwen']['official_reference_check'],
         frame_vector_shapes={k:list(np.load(OUT/(k+'-frames.npy')).shape) for k in ['siglip','qwen']}))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','live','siglip','qwen','finish'])
    phase=parser.parse_args().phase
    if phase in ['siglip','qwen']:run(phase)
    else:globals()[phase]()
