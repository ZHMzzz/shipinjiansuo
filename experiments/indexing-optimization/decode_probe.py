"""CPU-only sampling comparison. No inference, production edits or service stop."""
import gc
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.action_search import windows, SETTINGS


def indices(path,duration):
    cap=cv2.VideoCapture(str(path))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:cap.release()
    result=[]
    for start,end in windows(duration):
        first=min(count-1,max(0,math.ceil(start*fps)))
        last=min(count-1,max(first,math.ceil(end*fps)-1))
        result.append(np.linspace(first,last,SETTINGS['frames']).astype(int).tolist())
    return result


def seek_frames(path,groups):
    result=[]
    for group in groups:
        cap=cv2.VideoCapture(str(path))
        try:
            for number in group:
                cap.set(cv2.CAP_PROP_POS_FRAMES,number)
                ok,frame=cap.read()
                if not ok:raise RuntimeError(f'Cannot seek frame {number}')
                result.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        finally:cap.release()
    return result


def sequential_frames(path,groups):
    targets={i for group in groups for i in group};frames={}
    cap=cv2.VideoCapture(str(path))
    try:
        for number in range(max(targets)+1):
            if not cap.grab():raise RuntimeError(f'Cannot grab frame {number}')
            if number in targets:
                ok,frame=cap.retrieve()
                if not ok:raise RuntimeError(f'Cannot retrieve frame {number}')
                frames[number]=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    finally:cap.release()
    return [frames[i] for group in groups for i in group]


def run():
    output=Path(__file__).with_name('decode-probe-20261028.json')
    if output.exists():raise ValueError('Do not overwrite existing measurements')
    project=(ROOT/'projects/demo').resolve()
    with sqlite3.connect((project/'metadata/internal.db').as_uri()+'?mode=ro',uri=True) as db:
        rows=db.execute('SELECT m.id,m.path,m.duration,s.location FROM media m JOIN source_collections s ON s.id=m.source_collection_id WHERE m.id IN (40,41) ORDER BY m.id').fetchall()
    report={'scope':'CPU decoding/sample selection only; no Qwen preprocessing or inference',
            'order':['seek','sequential','sequential','seek','seek','sequential'],'videos':[]}
    for mid,name,duration,location in rows:
        path=Path(location)/name
        if not path.is_absolute():path=ROOT/path
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        groups=indices(path,duration);runs=[];reference=None
        for variant in report['order']:
            start=time.perf_counter()
            frames=(seek_frames if variant=='seek' else sequential_frames)(path,groups)
            elapsed=time.perf_counter()-start
            # Pixel hashing is intentionally outside timing.
            hashes=[hashlib.sha256(memoryview(frame)).hexdigest() for frame in frames]
            if reference is None:reference=hashes
            runs.append({'variant':variant,'seconds':elapsed,'all_pixels_equal':hashes==reference})
            del frames;gc.collect()
        medians={variant:statistics.median(r['seconds'] for r in runs if r['variant']==variant)
                 for variant in ('seek','sequential')}
        assert before==hashlib.sha256(path.read_bytes()).hexdigest()
        result={'media_id':mid,'duration':duration,'source_sha256':before,'windows':len(groups),
                'sampled_frames':sum(map(len,groups)),'unique_sampled_frames':len({i for g in groups for i in g}),
                'runs':runs,'median_seconds':medians,'sampling_indices':groups,
                'all_pixels_equal':all(r['all_pixels_equal'] for r in runs)}
        report['videos'].append(result)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(mid,medians,'same_pixels',result['all_pixels_equal'],flush=True)


if __name__=='__main__':run()
