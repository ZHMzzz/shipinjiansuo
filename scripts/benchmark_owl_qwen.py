"""Read-only production comparison; synthetic controls are explicitly separate.

Reference crop coordinates/labels are frozen before inference. Exact-source
recall is a diagnostic, not category accuracy; same-class other videos may be
valid hits. Controlled patches reuse query pixels and are an optimistic test.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps

from benchmark_qwen_replacement import ROOT,QwenEncoder,fingerprint

OUT=ROOT/'experiments/owl-qwen-comparison'
OLD=ROOT/'experiments/qwen-replacement'
OWL='transformers/owlv2/google/owlv2-base-patch16-ensemble'
CROPS=[
 ('bag_street',1,[.33,.48,.61,.99],[1,2,17],'backpack'),
 ('bag_indoor',2,[.51,.34,.92,.86],[1,2,17],'backpack'),
 ('glasses',3,[.35,.14,.68,.41],[3,4,24],'eyeglasses'),
 ('pan',5,[.30,.67,.58,.88],[5,6],'pan'),
 ('sunglasses_pair',13,[.48,.49,.76,.74],[13,14],'sunglasses'),
 ('sunglasses_single',14,[.44,.20,.70,.43],[13,14],'sunglasses'),
 ('toy',15,[.42,.50,.68,.84],[15,16],'toy'),
 ('helmet_large',17,[.48,.18,.62,.34],[17,19,20,21],'helmet'),
 ('helmet_small',19,[.26,.82,.35,.91],[17,19,20,21],'helmet'),
 ('excavator',21,[.47,0,.80,.24],[21],'excavator')]


def dump(name,data):
    OUT.mkdir(exist_ok=True,parents=True);p=OUT/name;t=p.with_suffix('.tmp')
    t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(p)


def design():
    d=json.loads((OUT/'design.json').read_text())
    assert fingerprint()==d['production_before'],'Production library changed'
    for f,h in d['input_sha256'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
    return d


def prepare():
    old=json.loads((OLD/'design.json').read_text());assert fingerprint()==old['production_before']
    queries=[];(OUT/'queries').mkdir(exist_ok=True);(OUT/'controls').mkdir(exist_ok=True)
    for name,mid,box,known,label in CROPS:
        media=next(m for m in old['media'] if m['id']==mid)
        path=Path(media['location'])/media['path']
        if not path.is_absolute():path=ROOT/path
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS)
        assert fps>0
        cap.set(cv2.CAP_PROP_POS_FRAMES,round(1.25*fps));ok,frame=cap.read();cap.release();assert ok
        Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(OUT/f'source-{mid}.png')
        src=OUT/f'source-{mid}.png';im=Image.open(src).convert('RGB');w,h=im.size
        px=[round(box[0]*w),round(box[1]*h),round(box[2]*w),round(box[3]*h)]
        crop=im.crop(px);dest=OUT/'queries'/f'{name}.png';crop.save(dest)
        source_ids=[mid]+([17] if mid==2 else [16] if mid==15 else [])
        queries.append(dict(id=name,image=str(dest.relative_to(ROOT)),source_media=source_ids,
            known_class_media=known,label=label,source_ts=1.25,source_image=str(src.relative_to(ROOT)),
            crop_xyxy=px,crop_area_fraction=(px[2]-px[0])*(px[3]-px[1])/(w*h)))
    queries.append(dict(id='independent_bag',image='data/queries/backpack-independent.jpg',source_media=[],
                       known_class_media=[1,2,17],label='backpack'))
    controls=[];targets=['bag_street','bag_indoor','helmet_large']
    backgrounds=[]
    for mid in [11,12]:
        frame=next(f for f in old['frames'] if f['media_id']==mid)
        backgrounds.append((f'background_{mid}',ImageOps.fit(Image.open(ROOT/frame['image']).convert('RGB'),(640,360))))
    for bgid,bg in backgrounds:
        dest=OUT/'controls'/f'{bgid}-empty.png';bg.save(dest)
        controls.append(dict(id=f'{bgid}-empty',background=bgid,area=0,target=None,image=str(dest.relative_to(ROOT)),boxes={}))
        for area in [.01,.04,.16]:
            for target in targets:
                crop=Image.open(ROOT/next(q for q in queries if q['id']==target)['image']).convert('RGB')
                aspect=crop.width/crop.height;h=round(math.sqrt(640*360*area/aspect));w=round(h*aspect)
                patch=crop.resize((w,h),Image.Resampling.LANCZOS);x=(640-w)//2;y=(360-h)//2
                im=bg.copy();im.paste(patch,(x,y));name=f'{bgid}-{int(area*100)}-{target}'
                dest=OUT/'controls'/f'{name}.png';im.save(dest)
                controls.append(dict(id=name,background=bgid,area=area,target=target,image=str(dest.relative_to(ROOT)),
                                     boxes={target:[x/640,y/360,w/640,h/360]}))
        # All three known patches in one image: query-specific localization test.
        im=bg.copy();boxes={}
        for j,target in enumerate(targets):
            crop=Image.open(ROOT/next(q for q in queries if q['id']==target)['image']).convert('RGB')
            aspect=crop.width/crop.height;h=round(math.sqrt(640*360*.04/aspect));w=round(h*aspect)
            x=round((j+.5)*640/3-w/2);y=(360-h)//2;im.paste(crop.resize((w,h),Image.Resampling.LANCZOS),(x,y))
            boxes[target]=[x/640,y/360,w/640,h/360]
        dest=OUT/'controls'/f'{bgid}-multi.png';im.save(dest)
        controls.append(dict(id=f'{bgid}-multi',background=bgid,area=.04,target='multi',image=str(dest.relative_to(ROOT)),boxes=boxes))
    files={q['image'] for q in queries}|{q['source_image'] for q in queries if 'source_image' in q}|{c['image'] for c in controls}
    d=dict(production_before=fingerprint(),media=old['media'],frames=old['frames'],queries=queries,controls=controls,
           controlled_targets=targets,repeats=3,qwen_image_max_pixels=262144,
           input_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sorted(files)},
           limitations=['10 same-source crops, 1 independent category image; no independent identity test',
                        'Known class positives are not exhaustive labels for all frames',
                        'Synthetic rectangular patches retain crop context and share query pixels',
                        'Qwen frames use official template; existing segments use legacy template'])
    dump('design.json',d)
    for name,items in [('queries',queries),('controls',controls)]:
        width=300;cellh=220;sheet=Image.new('RGB',(width*4,cellh*math.ceil(len(items)/4)),'#ededed');draw=ImageDraw.Draw(sheet)
        for i,item in enumerate(items):
            im=ImageOps.contain(Image.open(ROOT/item['image']),(290,185));x=(i%4)*width;y=(i//4)*cellh
            sheet.paste(im,(x,y));draw.text((x,y+190),item['id'],fill='black')
        sheet.save(OUT/(name+'-overview.jpg'))
    print('PREPARED',len(queries),'queries',len(controls),'controlled images',flush=True)


def relevant(h,q,key):
    if h['media_id'] not in q[key]:return False
    if h['media_id']==17:
        span=[0,8] if q['label']=='helmet' else [8,16] if q['label']=='backpack' else None
        if span:
            if 'te' in h:return max(0,min(h['te'],span[1])-max(h['ts'],span[0]))/(h['te']-h['ts'])>=.5
            return span[0]<=h['ts']<span[1]
    return True


def metrics(hits,q):
    return {key+'_rank':next((i+1 for i,h in enumerate(hits) if relevant(h,q,key)),None)
            for key in ['source_media','known_class_media']}


def rank(v,matrix,records):
    t=time.perf_counter();scores=matrix@v;order=np.argsort(-scores,kind='stable');seen=set();hits=[]
    for idx in order:
        h=records[int(idx)];mid=h['media_id']
        if mid in seen:continue
        seen.add(mid);hits.append({k:v for k,v in h.items() if k!='image'}|{'score':float(scores[idx])})
    return hits,time.perf_counter()-t


def live():
    import requests
    d=design();report={'queries':[],'method':'online-owlv2'}
    for q in d['queries']:
        times=[];hits=[]
        for _ in range(d['repeats']):
            t=time.perf_counter();r=requests.post('http://127.0.0.1:9670/demo/search',
                params=dict(search_in='video',feature_extractor_id=OWL,start=0,end=100,thumbnails_to_send=0,add_prefix='false'),
                files={'image_file_queries':('query.png',(ROOT/q['image']).read_bytes(),'image/png')},timeout=120)
            r.raise_for_status();times.append(time.perf_counter()-t);data=r.json()['video_results'];hits=[];seen=set()
            for h in data['unmerged_windows']:
                mid=int(h['media_id'])
                if mid in seen:continue
                seen.add(mid);hits.append(dict(media_id=mid,ts=h.get('thumbnail_ts',h['ts']),score=h['distance'],bbox=h.get('bbox')))
        report['queries'].append(dict(id=q['id'],http_seconds=times,ranking=hits,**metrics(hits,q)))
        dump('live-owl.json',report);print('LIVE',q['id'],metrics(hits,q),flush=True)


class OwlEncoder:
    def __init__(self):
        from wise.feature.transformers_owlv2 import TransformersOWLv2FeatureExtractor as Ext
        self.ext=Ext(OWL,device='mps',warmup=False,config=Ext.Config(device='mps',compile=False,warmup=False))
        _=self.ext.model.model;_=self.ext.processor

    def features(self,im):
        return self.ext.extract_image_features(self.ext.preprocess_image([im]))[0]

    def encode(self,im):
        f=self.features(im)
        if not len(f.vectors):raise ValueError('No object query features')
        v=self.ext.transform_internal_image_queries_hook(f.vectors[:1].copy())[0]
        return v/np.linalg.norm(v)


def run(method):
    import torch,psutil
    from resource_metrics import ResourceSampler
    d=design();torch.set_num_threads(1);torch.mps.set_per_process_memory_fraction(.55)
    report=dict(method=method,status='running',queries=[],encode_gallery_seconds=[],controlled_scores={})
    def guard():
        if psutil.virtual_memory().available<2*1024**3:raise RuntimeError('Less than 2 GiB available RAM')
    with ResourceSampler() as resources:
        guard();t=time.perf_counter();enc=OwlEncoder() if method=='owl' else QwenEncoder()
        torch.mps.synchronize();report['load_seconds']=time.perf_counter()-t
        enc.encode(Image.open(ROOT/d['queries'][0]['image']).convert('RGB'))
        print('LOADED',method,report['load_seconds'],flush=True)
        matrix=[];records=[];control_features={}
        items=(d['frames'] if method=='owl' else [])+d['controls']
        for i,item in enumerate(items):
            guard();im=Image.open(ROOT/item['image']).convert('RGB');t=time.perf_counter()
            if method=='owl':
                f=enc.features(im);vs=f.vectors
                rs=[{k:v for k,v in item.items() if k not in ('image','boxes')}|{'bbox':dict(zip(('x','y','w','h'),meta.bbox))} for meta in f.metadata]
            else:
                vs=enc.encode(im)[None,:];rs=[item]
            torch.mps.synchronize();report['encode_gallery_seconds'].append(time.perf_counter()-t)
            assert np.isfinite(vs).all()
            if 'media_id' in item:
                matrix.extend(vs);records.extend(rs)
            else:control_features[item['id']]=(vs,rs)
            if i%10==0:dump(method+'.json',report);print('IMAGE',method,i+1,'/',len(items),flush=True)
        if method=='owl':
            matrix=np.asarray(matrix,dtype='float32');np.save(OUT/'owl-regions.npy',matrix);dump('owl-regions.json',records)
        else:
            matrix=np.load(OLD/'qwen-frames.npy');records=d['frames']
            p=Path(d['production_before']['project']);segmatrix=np.load(p/'action-index/vectors.npy')
            segments=json.loads((p/'action-index/metadata.json').read_text())['segments']
        report['gallery_vector_shape']=list(matrix.shape)
        for q in d['queries']:
            guard();im=Image.open(ROOT/q['image']).convert('RGB');times=[]
            for _ in range(d['repeats']):
                t=time.perf_counter();v=enc.encode(im);torch.mps.synchronize();times.append(time.perf_counter()-t)
            hits,seconds=rank(v,matrix,records)
            row=dict(id=q['id'],encode_seconds=times,search_seconds=seconds,ranking=hits,**metrics(hits,q))
            if method=='qwen':
                times=[]
                for _ in range(d['repeats']):
                    t=time.perf_counter();lv=enc.encode(im,legacy=True);torch.mps.synchronize();times.append(time.perf_counter()-t)
                sh,st=rank(lv,segmatrix,segments);row['existing_segments']=dict(encode_seconds=times,search_seconds=st,ranking=sh,**metrics(sh,q))
            if q['id'] in d['controlled_targets']:
                scored=[]
                for item in d['controls']:
                    vs,rs=control_features[item['id']]
                    if len(vs):
                        scores=vs@v;n=int(np.argmax(scores));score=float(scores[n]);bbox=rs[n].get('bbox')
                    else:score=None;bbox=None
                    scored.append(dict(id=item['id'],score=score,bbox=bbox))
                report['controlled_scores'][q['id']]=scored
            report['queries'].append(row);dump(method+'.json',report)
            print('QUERY',method,q['id'],metrics(hits,q),row.get('existing_segments',{}).get('source_media_rank'),flush=True)
    report['resources']=resources.result;report['status']='completed';dump(method+'.json',report)


def iou(box,gt):
    if not box:return 0.
    a=[box[k] for k in ['x','y','w','h']];b=gt
    inter=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    return inter/max(a[2]*a[3]+b[2]*b[3]-inter,1e-9)


def finish():
    import requests
    d=design();reports={k:json.loads((OUT/(k+'.json')).read_text()) for k in ['owl','qwen','live-owl']};summary={};controls=[]
    for method in ['live-owl','owl','qwen','qwen-segments']:
        rows=reports['qwen' if method=='qwen-segments' else method]['queries'];out={}
        for key in ['source_media','known_class_media']:
            pairs=[(q,r['existing_segments'] if method=='qwen-segments' else r) for q,r in zip(d['queries'],rows) if q[key]]
            ranks=[r[key+'_rank'] for q,r in pairs]
            out[key]=dict(count=len(ranks),top1=sum(n==1 for n in ranks),top5=sum(n is not None and n<=5 for n in ranks))
        times=[t for row in rows for t in (row['existing_segments'] if method=='qwen-segments' else row).get('encode_seconds',row.get('http_seconds',[]))]
        out['median_seconds']=float(np.median(times));out['min_seconds']=min(times);out['max_seconds']=max(times);summary[method]=out
    for method in ['owl','qwen']:
        report=reports[method]
        for target,scores in report['controlled_scores'].items():
            lookup={s['id']:s for s in scores}
            for bgid in ['background_11','background_12']:
                for area in [.01,.04,.16]:
                    candidates=[c for c in d['controls'] if c['background']==bgid and (c['area']==area and c['target']!='multi' or c['target'] is None)]
                    ordered=sorted(candidates,key=lambda c:lookup[c['id']]['score'] if lookup[c['id']]['score'] is not None else -float('inf'),reverse=True)
                    correct=next(c for c in candidates if c['target']==target)
                    rankpos=next(i+1 for i,c in enumerate(ordered) if c['id']==correct['id'])
                    controls.append(dict(method=method,target=target,background=bgid,area=area,rank=rankpos,
                        ranking=[dict(id=c['id'],score=lookup[c['id']]['score']) for c in ordered],
                        iou=iou(lookup[correct['id']]['bbox'],correct['boxes'][target]) if method=='owl' else None))
                multi=next(c for c in d['controls'] if c['background']==bgid and c['target']=='multi')
                controls.append(dict(method=method,target=target,background=bgid,area='multi',
                    iou=iou(lookup[multi['id']]['bbox'],multi['boxes'][target]) if method=='owl' else None))
    dump('summary.json',summary);dump('controlled-results.json',controls)
    state=requests.get('http://127.0.0.1:9670/mvp-api/actions/status',timeout=10);state.raise_for_status()
    dump('verification.json',dict(production_unchanged=fingerprint()==d['production_before'],production_after=fingerprint(),service=state.json(),
         qwen_reused_frames_sha256=hashlib.sha256((OLD/'qwen-frames.npy').read_bytes()).hexdigest()))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    for method in ['owl','qwen']:
        for area in [.01,.04,.16,'multi']:
            rows=[r for r in controls if r['method']==method and r['area']==area]
            print(method,area,'first',sum(r.get('rank')==1 for r in rows),'/',len(rows),
                  'localized_iou_50',sum(r['iou'] is not None and r['iou']>=.5 for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','live','owl','qwen','finish']);args=p.parse_args()
    if args.phase in ['owl','qwen']:run(args.phase)
    else:globals()[args.phase]()
