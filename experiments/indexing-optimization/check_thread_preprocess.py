"""Small CPU-only follow-up: fixed pixels, OWLv2 fast processor, threads 1/2/4."""
import json
from pathlib import Path
import sqlite3
import statistics
import time
import cv2
import numpy as np
import torch
from transformers import Owlv2ImageProcessorFast

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'thread-contention-20261028-rerun'
project=(ROOT/'projects/demo').resolve()
with sqlite3.connect((project/'metadata/internal.db').as_uri()+'?mode=ro',uri=True) as db:
    location,path=db.execute('SELECT s.location,m.path FROM media m JOIN source_collections s ON s.id=m.source_collection_id WHERE m.id=40').fetchone()
video=Path(location)/path
if not video.is_absolute():video=ROOT/video
capture=cv2.VideoCapture(str(video));frames=[]
for seconds in (0,10,20):
    capture.set(cv2.CAP_PROP_POS_MSEC,seconds*1000);ok,frame=capture.read();assert ok
    frames.append(torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0))
capture.release()
processor=Owlv2ImageProcessorFast.from_pretrained('google/owlv2-base-patch16-ensemble',local_files_only=True)
results={};references=None
for n in (1,2,4):
    torch.set_num_threads(n);assert torch.get_num_threads()==n
    outputs=[processor(images=f,return_tensors='pt',device='cpu')['pixel_values'].clone() for f in frames]
    if references is None:references=outputs
    results[n]={'outputs_exact_vs_1':all(torch.equal(a,b) for a,b in zip(outputs,references)),
                'max_abs_diff':max(float((a-b).abs().max()) for a,b in zip(outputs,references)),
                'different_values':sum(int(torch.count_nonzero(a!=b)) for a,b in zip(outputs,references)),
                'timings':[]}
for r in range(9):
    ns=(1,2,4);ns=ns[r%3:]+ns[:r%3]
    for n in ns:
        torch.set_num_threads(n)
        start=time.perf_counter()
        for f in frames:processor(images=f,return_tensors='pt',device='cpu')
        results[n]['timings'].append(time.perf_counter()-start)
for v in results.values():v['mean_seconds_per_three_frames']=statistics.mean(v['timings'])
result={'scope':'CPU-only fast image processor on identical decoded pixels, three frames; not end-to-end indexing nor accuracy test','results':results}
(OUT/'cpu-preprocessor.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
