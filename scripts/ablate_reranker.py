"""Independent diagnostic; no Demo configuration/index mutations.
Run after stopping Demo. Checkpoint each score; restart Demo outside this script.
Manual crop coordinates and configurations are fixed before inference.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from action_reranker import VideoReranker, ResourceSampler, SETTINGS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/reranker-ablation'
OLD = ROOT/'experiments/action-search'
IDS = ['helmet_indoor','helmet_outdoor','helmet_squat','worker_barehead','fall']
QUERIES = [('no_helmet','作业人员没有佩戴安全帽'),('helmet','工人佩戴安全帽')]
SPECIAL = ('Retrieve visual evidence that strictly satisfies the query, including negation. '
           'For safety-helmet queries, inspect visible heads and distinguish a safety helmet '
           'from exposed hair or scalp. Construction scenery and safety vests alone do not '
           'satisfy a helmet condition. If the query asks for a person without a safety helmet, '
           'at least one clearly visible head must lack a safety helmet; an obscured head is '
           'not sufficient evidence. If it asks for a person wearing a safety helmet, at least '
           'one visible person must wear one. Judge visible appearance, not legal requirements.')
# Anchor of all variants except time_corrected is time_corrected: isolate metadata first.
CONFIGS = [dict(id='baseline',frames=16,pixels=1048576,real_time=False),
 dict(id='time_corrected',frames=16,pixels=1048576,real_time=True),
 dict(id='pixels_2x',frames=16,pixels=2097152,real_time=True),
 dict(id='frames8_budget',frames=8,pixels=1048576,real_time=True),
 dict(id='frames4_budget',frames=4,pixels=1048576,real_time=True),
 dict(id='frames8_fixed_resolution',frames=8,pixels=1048576,real_time=True,fixed_resolution=True),
 dict(id='frames4_fixed_resolution',frames=4,pixels=1048576,real_time=True,fixed_resolution=True),
 dict(id='ppe_instruction',frames=16,pixels=1048576,real_time=True,instruction=SPECIAL),
 dict(id='person_crop',frames=16,pixels=1048576,real_time=True,crop='person'),
 dict(id='head_crop',frames=16,pixels=1048576,real_time=True,crop='head')]

# Normalized static regions. For the warehouse actor, centers follow manually reviewed frames.
STATIC = {
 'helmet_indoor': {'person':[.36,.13,.80,1.0], 'head':[.43,.12,.78,.64]},
 'helmet_outdoor': {'person':[.23,.10,.55,1.0], 'head':[.27,.09,.43,.46]},
 'helmet_squat': {'person':[.28,.05,.66,1.0], 'head':[.34,.05,.55,.44]}}
FALL_HEAD = [(144,30),(151,25),(158,29),(153,36),(175,40),(252,97),
             (232,94),(235,94),(239,94),(235,92),(226,103),(238,105),
             (240,89),(242,91),(242,93),(244,92)]
BARE_HEAD = [(x, y) for x,y in zip([112,116,119,120,121,124,126,126,127,128,130,131,138,140,143,143],
                                  [24,24,24,24,26,26,27,27,28,28,28,28,28,28,29,29])]


def region(name, kind, fraction):
    if name in STATIC: return STATIC[name][kind]
    if name=='worker_barehead':
        if kind=='person': return [.25,0,.55,.95]
        x,y=BARE_HEAD[min(15,round(fraction*15))]
        return [(x-24)/320,max(0,y-24)/180,(x+24)/320,(y+32)/180]
    i=min(15,round(fraction*15))
    if kind=='person':
        return [110/320,0,195/320,1] if i<4 else [75/320,15/180,290/320,1]
    x,y=FALL_HEAD[i]
    return [(x-27)/320,max(0,y-27)/180,(x+27)/320,min(180,y+30)/180]


def decode(name,n,crop=None):
    path=OLD/'videos'/f'{name}.mp4'
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or count<1: raise RuntimeError('Cannot decode '+name)
    indices=np.linspace(0,count-1,n).astype(int).tolist();frames=[];boxes=[];original_size=None
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,f=cap.read()
            if not ok: raise RuntimeError('Frame failed')
            im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));original_size=im.size
            box=None
            if crop:
                r=region(name,crop,i/max(1,count-1));w,h=im.size
                box=[round(r[0]*w),round(r[1]*h),round(r[2]*w),round(r[3]*h)]
                im=im.crop(box)
                # Keep identical crop shape for every time step of each clip.
                im=ImageOps.pad(im,(192,224) if crop=='head' else (256,320),method=Image.Resampling.BICUBIC,color=(127,127,127))
            im.thumbnail((1024,1024));frames.append(im);boxes.append(box)
    finally: cap.release()
    return frames,{'fps':fps,'frames_indices':indices,'total_num_frames':count},original_size,boxes


def dump(name,data):
    OUT.mkdir(exist_ok=True)
    p=OUT/name;temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(p)


def prepare():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((OLD/'manifest.json').read_text())
    clips=[c for c in manifest['clips'] if c['id'] in IDS]
    for c in clips: assert hashlib.sha256((OLD/c['file']).read_bytes()).hexdigest()==c['sha256']
    dump('design.json',{'settings':SETTINGS,'configs':CONFIGS,'queries':QUERIES,'clips':clips,
         'static_regions':STATIC,'fall_head_centers_320x180':FALL_HEAD,'bare_head_centers_320x180':BARE_HEAD,
         'notes':['Regions manually specified before scoring. Not automatic detection.',
                  'Crops resampled to fixed size for stacking; this magnifies existing pixels, does not restore missing detail.',
                  'Both barehead clips share a source; five-clip ranking is a diagnostic, not accuracy validation.',
                  'Frame-count variants compare to time_corrected, not baseline.']})
    for name in IDS:
        frames,_,_,_=decode(name,16,'head')
        sheet=Image.new('RGB',(8*144,2*190),'white');d=ImageDraw.Draw(sheet)
        for i,im in enumerate(frames):
            im=im.copy();im.thumbnail((140,165));x=i%8*144;y=i//8*190;sheet.paste(im,(x,y+20));d.text((x,y),str(i),fill='black')
        sheet.save(OUT/(name+'-head-crops.jpg'))
    print('Prepared fixed experiment design and crop review sheets',flush=True)


def score(model,query,cfg,frames=None,metadata=None,picture=None):
    import torch
    from qwen_vl_utils import process_vision_info
    content=[{'type':'text','text':'<Instruct>: '+cfg.get('instruction',SETTINGS['instruction'])},
             {'type':'text','text':'<Query>:'},{'type':'text','text':query},
             {'type':'text','text':'\n<Document>:'}]
    if picture is not None:
        content.append({'type':'image','image':picture,'min_pixels':4096,'max_pixels':262144})
    else:
        video={'type':'video','video':frames,'total_pixels':cfg['pixels']}
        if cfg.get('fixed_resolution'):video.update(resized_height=256,resized_width=448)
        content.append(video)
    messages=[{'role':'system','content':[{'type':'text','text':
        'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'}]},
        {'role':'user','content':content}]
    text=model.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    images,videos,kwargs=process_vision_info(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
    if videos is not None:
        tensors,metas=zip(*videos);tensors=list(tensors);metas=list(metas)
        if cfg.get('real_time'):metas=[metadata]
        shape=list(tensors[0].shape)
    else:tensors=metas=None;shape=None
    inputs=model.processor(text=[text],images=images,videos=tensors,video_metadata=metas,
        padding=True,truncation=False,do_resize=False,return_tensors='pt',**kwargs)
    tokens=inputs['input_ids'].shape[-1]
    if tokens>8192:raise RuntimeError('Too many tokens')
    detail={'input_shape':shape,'tokens':tokens,'metadata':metas,
            'video_grid_thw':inputs['video_grid_thw'].tolist() if 'video_grid_thw' in inputs else None,
            'image_grid_thw':inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else None}
    inputs=inputs.to('mps');started=time.perf_counter()
    with torch.inference_mode():
        hidden=model.model(**inputs).last_hidden_state[:,-1]
        value=torch.sigmoid(hidden.float()@model.score_weight.float()).item()
    torch.mps.synchronize()
    if not math.isfinite(value):raise RuntimeError('Invalid score')
    return value,{**detail,'forward_seconds':time.perf_counter()-started}


def main():
    import psutil,torch,urllib.request
    try:urllib.request.urlopen('http://127.0.0.1:9670/mvp-api/actions/status',timeout=1)
    except OSError:pass
    else:raise RuntimeError('Stop Demo to avoid duplicate large-model memory')
    design=json.loads((OUT/'design.json').read_text())
    assert design['configs']==CONFIGS,'Prepare design again after changing configurations'
    model=VideoReranker();load=model.load()
    report={'status':'running','model_load_seconds':load,'settings':SETTINGS,'runs':[],'created':time.time()}
    for cfg in CONFIGS:
        rows=[]
        with ResourceSampler() as resources:
            for name in IDS:
                frames,meta,original_size,boxes=decode(name,cfg['frames'],cfg.get('crop'))
                for category,query in QUERIES:
                    if psutil.virtual_memory().available<2*1024**3:raise RuntimeError('Low memory')
                    t=time.perf_counter();value,detail=score(model,query,cfg,frames,meta)
                    row={'clip':name,'category':category,'query':query,'score':value,
                         'seconds':time.perf_counter()-t,'source_size':original_size,'crop_boxes':boxes,**detail}
                    rows.append(row)
                    dump('checkpoint.json',{'config':cfg,'rows':rows,'finished_runs':report['runs']})
        rankings={category:sorted([r for r in rows if r['category']==category],key=lambda x:-x['score']) for category,_ in QUERIES}
        entry={'config':cfg,'rows':rows,'resources':resources.result,'rankings':{k:[r['clip'] for r in v] for k,v in rankings.items()}}
        report['runs'].append(entry);dump('video-results.json',report)
        if cfg['id']=='baseline':
            archived=json.loads((ROOT/'experiments/reranker/fixture.json').read_text())
            diffs=[]
            for row in rows:
                q=next(q for q in archived['queries'] if q['query']==row['query'])
                old=next(h['rerank_score'] for h in q['reranked'] if h['media_id']==row['clip'])
                diffs.append(abs(row['score']-old))
            report['baseline_max_difference']=max(diffs)
            assert max(diffs)<.002, f'Baseline mismatch: {max(diffs)}'
            print('BASELINE_REPRODUCED',max(diffs),flush=True)
        print('CONFIG',cfg['id'],round(resources.result['total_seconds'],2),entry['rankings'],flush=True)
    report['status']='completed';dump('video-results.json',report)
    pairs=json.loads((OUT/'paired/manifest.json').read_text())
    paired={'status':'running','rows':[],'manifest':pairs}
    queries=QUERIES+[('no_helmet_neutral','一个人没有佩戴安全帽'),('helmet_neutral','一个人佩戴安全帽')]
    for instruction in ['generic','ppe']:
        cfg={} if instruction=='generic' else {'instruction':SPECIAL}
        with ResourceSampler() as resources:
            for item in pairs['items']:
                im=Image.open(OUT/'paired'/item['file']).convert('RGB')
                for category,query in queries:
                    t=time.perf_counter();value,detail=score(model,query,cfg,picture=im)
                    paired['rows'].append({**item,'instruction':instruction,'category':category,'query':query,'score':value,
                                           'seconds':time.perf_counter()-t,**detail})
                    dump('paired-results.json',paired)
        paired[instruction+'_resources']=resources.result
        print('PAIRED',instruction,round(resources.result['total_seconds'],2),flush=True)
    paired['status']='completed';dump('paired-results.json',paired);model.unload()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else main()
