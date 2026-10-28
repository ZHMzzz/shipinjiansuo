"""Read-only production snapshot, exact production frame sampling and review sheets."""
from pathlib import Path
import hashlib
import json
import math
import os
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN = ROOT / '.scratch/qwen8b-full-library'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))

def protect():
    project = (ROOT/'projects/demo').resolve()
    meta = json.loads((project/'action-index/metadata.json').read_text())
    draft = json.loads((PLAN/'library-snapshot.json').read_text())
    assert meta['media'] == draft['media'] and meta['segments'] == draft['segments'], 'Library changed'
    paths = list(json.loads((ROOT/'experiments/qwen8b-int8/production-before.json').read_text())['files'])
    paths += [str(p) for p in (Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots').rglob('*') if p.is_file()]
    save(HERE/'production-before.json', {'created_at': time.time(), 'project': str(project),
        'pointer': os.readlink(ROOT/'projects/demo'), 'files': {p: sha(p) for p in sorted(set(paths))}})
    save(HERE/'library-snapshot.json', {**meta, 'project': str(project)})
    return meta

def sheets(mid, times, images, label):
    from PIL import Image, ImageDraw
    folder = HERE/'review'
    folder.mkdir(exist_ok=True)
    for page in range(math.ceil(len(images)/40)):
        group = images[page*40:(page+1)*40]
        canvas = Image.new('RGB', (1200, 30+math.ceil(len(group)/5)*175), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((5,5), f'Media {mid:02d} / {label} / page {page+1}', fill='black')
        for j, im in enumerate(group):
            im = im.copy(); im.thumbnail((236,148))
            x, y = (j%5)*240, 30+(j//5)*175
            canvas.paste(im, (x+(240-im.width)//2,y))
            draw.text((x+4,y+150), f'{times[page*40+j]:.3f}s', fill='black')
        canvas.save(folder/f'{mid:02d}-{label}-{page+1}.jpg', quality=90)

def main():
    import cv2
    import numpy as np
    from PIL import Image
    meta = protect()
    (HERE/'frames').mkdir(exist_ok=True)
    clips, source_info = [], []
    for m in meta['media']:
        path = Path(m['location'])/m['path']
        if not path.is_absolute(): path=ROOT/path
        cap = cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert count > 0 and fps > 0
        def read(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(n)); ok,f=cap.read(); assert ok, (m['id'],n)
            return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        times = list(np.arange(0,count/fps,1)) + [(count-1)/fps]
        imgs = [Image.fromarray(read(min(count-1,round(t*fps)))) for t in times]
        sheets(m['id'], times, imgs, '1fps')
        if m['id'] in [15,16,17,18,19,21,25,26,27,28,29,30,32,40]:
            times=list(np.arange(0,count/fps,.25))
            imgs=[Image.fromarray(read(min(count-1,round(t*fps)))) for t in times]
            sheets(m['id'],times,imgs,'4fps')
        for s in (x for x in meta['segments'] if x['media_id']==m['id']):
            ident=f'm{m["id"]:02d}-t{int(s["ts"]):03d}'
            first=min(count-1,max(0,math.ceil(s['ts']*fps)))
            last=min(count-1,max(first,math.ceil(s['te']*fps)-1))
            indices=np.linspace(first,last,16).astype(int)
            started=time.perf_counter(); array=np.stack([read(n) for n in indices]); elapsed=time.perf_counter()-started
            np.savez_compressed(HERE/'frames'/f'{ident}.npz',frames=array)
            clips.append({'id':ident,**s,'file':m['path'],'source_path':str(path),'fps':fps,
                'source_indices':indices.tolist(),'sample_times':(indices/fps).tolist(),
                'rgb_shape':list(array.shape),'rgb_sha256':hashlib.sha256(array.tobytes()).hexdigest(),
                'decode_seconds':elapsed})
        cap.release()
        source_info.append({**m,'resolved_path':str(path),'sha256':sha(path),'frame_count':count,'fps':fps})
        print('PREPARED',m['id'],len(clips),flush=True)
    save(HERE/'dataset-unlabeled.json',{'clips':clips,'sources':source_info,'queries':[],
        'settings':meta['settings'],'created_at':time.time(), 'sampling':'exact scripts/action_search.py frame-index formula'})

if __name__=='__main__': main()
