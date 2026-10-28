"""CPU-only comparison of independent OWLv2 / Qwen microbatch trials."""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import bench
from evaluate import Gallery
from compare_face_metadata import iou
OUT=Path(os.environ.get('MICROBATCH_OUT', bench.OUT/'microbatch-20261028')).resolve()
def load(name):return json.loads((OUT/name).read_text())
def timing(prefix):
    result={}
    for batch in [1,2]:
        rows=[]
        for rep in [1,2]:
            if prefix=='owl':
                r=load(f'owl-b{batch}-r{rep}/timing.json');m=load(f'owl-b{batch}-r{rep}/metrics.json')
                row={'total_seconds':r['wise_total_seconds'],'extract_seconds':r['extract_wall_seconds'],'index_seconds':r['index_wall_seconds'],'model_and_preprocess_seconds':m['timers']['owl_features']['seconds'],'forward_seconds':m['timers']['owl_forward']['seconds'],'resources':m['resources']}
            else:
                r=load(f'qwen-b{batch}-r{rep}.json')
                row={k:r[k] for k in ['encode_seconds','decode_seconds','decode_and_encode_seconds','load_seconds','resources']}
            rows.append(row)
        result[str(batch)]={'runs':rows,'mean':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='resources'},'peak_mps_driver_gib':max(r['resources']['sampled_peak']['mps_driver_bytes'] for r in rows)/1024**3,'peak_rss_gib':max(r['resources']['sampled_peak']['rss_bytes'] for r in rows)/1024**3,'min_system_available_gib':min(r['resources']['min_system_available_bytes'] for r in rows)/1024**3}
    result['time_reduction_percent']={k:100*(1-result['2']['mean'][k]/result['1']['mean'][k]) for k in result['1']['mean']}
    return result

def owl():
    canonical={Path(m['path']).name:m['id'] for m in json.loads((bench.OUT/'queries.json').read_text())['media']}
    a,b=[Gallery(OUT/f'owl-b{x}-r1/project',canonical) for x in [1,2]]
    groups=defaultdict(list)
    for r in b.records.values():groups[(r['filename'],r['timestamp'])].append(r)
    counts_a=Counter((r['filename'],r['timestamp']) for r in a.records.values());counts_b=Counter((r['filename'],r['timestamp']) for r in b.records.values())
    paired=[];used=set()
    exact=defaultdict(list)
    for r in b.records.values():exact[(r['filename'],r['timestamp'],tuple(r['bbox']))].append(r)
    for r in a.records.values():
        same=[s for s in exact.get((r['filename'],r['timestamp'],tuple(r['bbox'])),[]) if s['vector_id'] not in used]
        if same:
            s=same[0];used.add(s['vector_id']);paired.append((r['vector_id'],s['vector_id'],1.0));continue
        candidates=[s for s in groups[(r['filename'],r['timestamp'])] if s['vector_id'] not in used]
        if not candidates:continue
        s=max(candidates,key=lambda x:iou(r['bbox'],x['bbox']))
        overlap=iou(r['bbox'],s['bbox'])
        if overlap<.9:continue
        used.add(s['vector_id']);paired.append((r['vector_id'],s['vector_id'],overlap))
    av=a.index.reconstruct_batch(np.array([p[0] for p in paired],dtype='int64'));bv=b.index.reconstruct_batch(np.array([p[1] for p in paired],dtype='int64'))
    cosine=np.sum(av*bv,axis=1)/(np.linalg.norm(av,axis=1)*np.linalg.norm(bv,axis=1))
    report={'timing':timing('owl'),'regions':{'batch1':len(a.records),'batch2':len(b.records),'frames':len(counts_a),'frames_count_changed':sum(counts_a[k]!=counts_b[k] for k in counts_a.keys()|counts_b.keys()),'aligned_iou_ge_09':len(paired),'cosine_min':float(cosine.min()),'cosine_median':float(np.median(cosine))},'quality':load('owl-quality.json')['summary']}
    bench.dump(OUT/'owl-summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

def qwen():
    import faiss
    faiss.omp_set_num_threads(1)
    rows=[load(f'qwen-b{b}-r{r}.json') for b,r in [(1,1),(2,1),(2,2),(1,2)]]
    assert all(r['segments']==rows[0]['segments'] and r['input_hashes']==rows[0]['input_hashes'] for r in rows)
    arrays=[np.load(OUT/f'qwen-b{b}-r{r}.npy') for b,r in [(1,1),(2,1),(2,2),(1,2)]]
    assert all(np.isfinite(a).all() for a in arrays)
    cosine=np.sum(arrays[0]*arrays[1],axis=1)/(np.linalg.norm(arrays[0],axis=1)*np.linalg.norm(arrays[1],axis=1))
    design=json.loads((bench.OUT.parent/'wise-upload-original/design.json').read_text());queries=design['queries']
    querypath=bench.OUT.parent/'wise-upload-original/current-full/query-vectors.npy'
    q=np.load(querypath);q=q/np.linalg.norm(q,axis=1,keepdims=True)
    meta=load('qwen-background.json');background=np.load(OUT/'qwen-background.npy');lookup={(s['media_id'],s['ts'],s['te']):i for i,s in enumerate(meta['segments'])}
    where=[lookup[(s['media_id'],s['ts'],s['te'])] for s in rows[0]['segments']]
    full=[]
    for a in arrays[:2]:
        v=background.copy();v[where]=a;full.append(v)
    known=set(design['labeled_media_ids']);selected={s['media_id'] for s in rows[0]['segments']}
    details=[]
    for scope,ss,galleries in [('subset',rows[0]['segments'],arrays[:2]),('fixed_background',meta['segments'],full)]:
        ids=np.array([s['media_id'] for s in ss]);allowed=np.array([i for i,m in enumerate(ids) if m in known])
        indexes=[]
        for matrix in galleries:
            index=faiss.IndexFlatIP(matrix.shape[1]);index.add(np.ascontiguousarray(matrix[allowed],dtype='float32'));indexes.append(index)
        for qi,query in enumerate(queries):
            if query['answer_status']=='ungraded':continue
            grades={m:0 for m in known}
            for k,g in query['relevance'].items():
                mid=int(k.split('-')[0][1:]);grades[mid]=max(grades.get(mid,0),g)
            positives={m for m,g in grades.items() if g==2 and m in ids}
            rankings=[];segment_orders=[];score_arrays=[]
            for index in indexes:
                distances,positions=index.search(np.ascontiguousarray(q[qi:qi+1],dtype='float32'),len(allowed))
                order=allowed[positions[0]];scores=np.zeros(len(allowed),dtype='float32');scores[positions[0]]=distances[0]
                segment_orders.append(order.tolist());score_arrays.append(scores)
                ranked=list(dict.fromkeys(int(ids[i]) for i in order));rankings.append(ranked)
            first_ranks=[next((i+1 for i,m in enumerate(r) if m in positives),None) for r in rankings]
            details.append({'scope':scope,'query_id':query['id'],'text':query['text'],'answer_status':query['answer_status'],'positive_media_in_test':sorted(positives),'top1_same':rankings[0][0]==rankings[1][0],'top5_same_order':rankings[0][:5]==rankings[1][:5],'top5_overlap':len(set(rankings[0][:5])&set(rankings[1][:5]))/5,'segment_top1_same':segment_orders[0][0]==segment_orders[1][0],'first_positive_ranks':first_ranks,'video_top5':[r[:5] for r in rankings],'max_abs_score_change':float(np.max(np.abs(score_arrays[0]-score_arrays[1])))})
    summaries=[]
    for scope in ['subset','fixed_background']:
        rr=[r for r in details if r['scope']==scope];answerable=[r for r in rr if r['positive_media_in_test']]
        summaries.append({'scope':scope,'query_count':len(rr),'top1_same':sum(r['top1_same'] for r in rr),'top5_same_order':sum(r['top5_same_order'] for r in rr),'mean_top5_overlap':float(np.mean([r['top5_overlap'] for r in rr])),'segment_top1_same':sum(r['segment_top1_same'] for r in rr),'answerable_in_gallery':len(answerable),'hit_at_1':[sum(r['first_positive_ranks'][i]==1 for r in answerable) for i in [0,1]],'hit_at_5':[sum(r['first_positive_ranks'][i] is not None and r['first_positive_ranks'][i]<=5 for r in answerable) for i in [0,1]],'regressed_first_positive_rank':[r['query_id'] for r in answerable if r['first_positive_ranks'][1]>r['first_positive_ranks'][0]]})
    report={'timing':timing('qwen'),'videos':len(selected),'segments':len(where),'batch2_group_counts':dict(Counter(g['count'] for g in rows[1]['groups'])),'all_inputs_identical':True,'all_vectors_finite':True,'cosine_min':float(cosine.min()),'cosine_median':float(np.median(cosine)),'max_abs_vector_diff':float(np.max(np.abs(arrays[0]-arrays[1]))),'repeat_max_abs_diff':{'batch1':float(np.max(np.abs(arrays[0]-arrays[3]))),'batch2':float(np.max(np.abs(arrays[1]-arrays[2])))} ,'mixed_shape_validation':rows[1]['mixed_shape_cosines'],'preprocessing_validation':rows[1]['preprocessing_validation'],'query_vectors_sha256':bench.sha(querypath),'quality':summaries,'queries':details,'limitations':['Only selected video segments re-encoded; background held fixed. Query vectors held fixed.','FAISS IndexFlatIP scores; video ranks use best segment score. No frontend temporal merging or person gates are replayed.','Small fixed regression set, not a general retrieval accuracy estimate. No-answer query rankings are stability checks only.']}
    bench.dump(OUT/'qwen-summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['queries','timing','preprocessing_validation']},ensure_ascii=False,indent=2))
if __name__=='__main__':
    import sys
    owl() if sys.argv[1]=='owl' else qwen()
