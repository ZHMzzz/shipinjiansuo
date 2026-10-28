#!/usr/bin/env python3
"""Freeze native-resolution inputs for the offline two-model PPE experiment."""
from pathlib import Path
import cv2
import hashlib
import json
import math
import time
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def main():
    if (OUT/'labels.json').exists():
        raise SystemExit('Frozen labels already exist; use a new experiment folder instead of overwriting this dataset.')
    specs=[]
    for name in ['helmet_indoor','helmet_outdoor','helmet_squat','worker_barehead','fall']:
        specs.append(dict(id='fixture_'+name, group='historical_diagnostic', source_path=str(ROOT/'experiments/action-search/videos'/f'{name}.mp4'),media_id=None,start=0,end=None,expected_no_helmet='unknown',note='Existing diagnostic clip; labels pending pixel review.'))
    snapshot=json.loads((ROOT/'experiments/qwen8b-full-library/library-snapshot.json').read_text())
    known={m['id']:m for m in snapshot['media']}
    for cid,mid,start,end in [('library21',21,0,8),('hat_before',27,0,4),('hat_transition',27,4,8),('hat_after',27,8,14),('library30',30,0,None),('library41',41,0,None)]:
        if mid in known:
            m=known[mid];p=Path(m['location'])/m['path']
            if not p.is_absolute():p=ROOT/p
        else:
            p=next((ROOT/'data/uploads').rglob('抖音20261027-906986.mp4'))
        specs.append(dict(id=cid,group='current_library',source_path=str(p),media_id=mid,start=start,end=end,expected_no_helmet='unknown',note='Current library clip; labels pending pixel review.'))
    image=ROOT/'experiments/instruct-verifier/head-out-of-frame.jpg'
    specs.append(dict(id='head_out_of_frame',group='visibility_diagnostic',type='image',source_path=str(image),media_id=None,start=0,end=0,expected_no_helmet='unknown',note='Head is outside the image; absence of helmet detections cannot prove no helmet.'))
    for c in specs:
        source=Path(c['source_path']);c['source_sha256']=sha256(source);frames=[]
        folder=OUT/'frames'/c['id'];folder.mkdir(parents=True,exist_ok=True)
        started=time.perf_counter()
        if c.get('type')=='image':
            im=cv2.imread(str(source)); seq=[(0,0,im)];fps=None
        else:
            cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));duration=count/fps
            if c['end'] is None:c['end']=duration
            c['source_fps']=fps;c['source_frame_count']=count
            seq=[]
            for i in range(math.ceil((c['end']-c['start'])*2-1e-8)):
                requested=c['start']+i/2
                number=min(count-1,max(0,round(requested*fps)))
                cap.set(cv2.CAP_PROP_POS_FRAMES,number);ok,im=cap.read()
                if not ok:raise RuntimeError(f'Cannot decode {source} frame {number}')
                seq.append((number,number/fps,im))
            cap.release()
        for i,(number,ts,im) in enumerate(seq):
            file=folder/f'{i:04d}.png';cv2.imwrite(str(file),im)
            frames.append(dict(id=f'{c["id"]}_{i:04d}',path=str(file),timestamp=ts,source_frame=number,width=im.shape[1],height=im.shape[0],sha256=sha256(file)))
        c['frames']=frames;c['decode_and_write_seconds']=time.perf_counter()-started
        # Pixel review only. Native inputs above remain unmodified.
        for page,offset in enumerate(range(0,len(frames),24)):
            tiles=frames[offset:offset+24];cw,ch=320,340
            sheet=Image.new('RGB',(cw*4,ch*math.ceil(len(tiles)/4)),(238,238,238));draw=ImageDraw.Draw(sheet)
            for j,f in enumerate(tiles):
                pic=Image.open(f['path']);pic.thumbnail((cw,ch-25))
                x=(j%4)*cw;y=(j//4)*ch;sheet.paste(pic,(x+(cw-pic.width)//2,y+25));draw.text((x+5,y+5),f'{f["id"]} t={f["timestamp"]:.3f}',fill='black')
            sheet.save(OUT/f'label-review-{c["id"]}-{page+1}.jpg',quality=95)
    data=dict(schema=1,sampling_fps=2,input_policy='Native decoded BGR pixels saved as PNG; no model-dependent crop or preprocessing. Half-open clip intervals.',label_status='pending_pre_prediction_pixel_review',cases=specs,limitations=['Existing clips are correlated and some include overlays; not an independent dataset or object-detection mAP benchmark.','No model prediction was used to create labels.'])
    (OUT/'dataset.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(json.dumps({'cases':len(specs),'frames':sum(len(c['frames']) for c in specs),'dataset':str(OUT/'dataset.json')}))

if __name__=='__main__':main()
