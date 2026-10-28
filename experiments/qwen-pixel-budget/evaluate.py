"""Summarize the fixed single-video ablation without mutating any production file."""
import json
from pathlib import Path
import statistics
import numpy as np

HERE=Path(__file__).resolve().parent
def load(name):return json.loads((HERE/name).read_text())
design=load('design.json');n=len(design['segments']);baseline=np.load(HERE/'production-vectors.npy')
query=np.load(HERE/'low-1-queries.npy')
reports={name:load(name+'.json') for name in design['order']}
assert all(r['status']=='completed' and r['dtype']=='torch.float16' for r in reports.values())
raw_hashes={i:set() for i in range(n)}
summary={'media':design['media'],'segments':design['segments'],'variants':{},'queries':[],'validation':{}}
for i in range(n):
    for r in reports.values():
        for k in range(2):raw_hashes[i].add(r['segments'][k*n+i]['raw_hash'])
assert all(len(h)==1 for h in raw_hashes.values())
summary['validation']['same_raw_frames_all_runs']=True
summary['validation']['baseline_max_abs_error']=0.
summary['validation']['max_text_vector_abs_error']=0.
for name in design['order']:
    vectors=np.load(HERE/(name+'-vectors.npy'));assert vectors.shape==(n*2,baseline.shape[1])
    delta=float(np.max(np.abs(query-np.load(HERE/(name+'-queries.npy')))))
    summary['validation']['max_text_vector_abs_error']=max(delta,summary['validation']['max_text_vector_abs_error'])
    if name.startswith('low'):
        error=float(np.max(abs(vectors-np.tile(baseline,(2,1)))))
        summary['validation']['baseline_max_abs_error']=max(error,summary['validation']['baseline_max_abs_error'])
assert summary['validation']['baseline_max_abs_error']<1e-5
assert summary['validation']['max_text_vector_abs_error']<1e-5
for variant in ['low','high']:
    chosen=[r for k,r in reports.items() if k.startswith(variant)]
    times=[x['wall_seconds_excluding_hash'] for r in chosen for x in r['rounds']]
    shapes={tuple(s['model_input']['grid'][0]) for r in chosen for s in r['segments']}
    tokens=sorted({s['model_input']['tokens'] for r in chosen for s in r['segments']})
    summary['variants'][variant]={'budget':chosen[0]['budget'],'full_video_seconds':times,'median_seconds':statistics.median(times),
        'min_seconds':min(times),'max_seconds':max(times),'model_grid_shapes':[list(s) for s in sorted(shapes)],'tokens':tokens,
        'frame_sizes_height_width':[[s[1]*16,s[2]*16] for s in sorted(shapes)],
        'mps_driver_peak_gib':[r['resources']['sampled_peak']['mps_driver_bytes']/2**30 for r in chosen],
        'load_seconds':[r['load_seconds'] for r in chosen]}
for qi,(text,match) in enumerate(design['queries']):
    row={'text':text,'matches_video':match}
    for variant in ['low','high']:
        trials=[]
        for name in design['order']:
            if not name.startswith(variant):continue
            matrix=np.load(HERE/(name+'-vectors.npy'))
            for k in range(2):
                scores=matrix[k*n:(k+1)*n]@query[qi];best=int(np.argmax(scores))
                trials.append({'max_score':float(scores[best]),'best_segment_start':design['segments'][best]['ts'],
                    'all_segment_scores':[float(v) for v in scores]})
        row[variant]={'mean_max_score':statistics.mean(t['max_score'] for t in trials),'trials':trials}
    summary['queries'].append(row)
summary['time_ratio']=summary['variants']['high']['median_seconds']/summary['variants']['low']['median_seconds']
(HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
