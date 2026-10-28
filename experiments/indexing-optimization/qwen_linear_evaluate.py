"""Fresh complete-gallery vector and real WISE search regression, fixed queries."""
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import sqlite3
import statistics
import time
import numpy as np
import bench
from qwen_linear import OUT

def load(name):return json.loads((OUT/name).read_text())
def key(s):return (s['media_id'],s['ts'],s['te'])
def compare_arrays(a,b):
    x=a.astype('float64');y=b.astype('float64');norms=np.linalg.norm(x,axis=1);norms2=np.linalg.norm(y,axis=1)
    cosine=np.clip(np.sum(x*y,axis=1)/(norms*norms2),-1,1);delta=np.abs(x-y)
    return {'all_finite':bool(np.isfinite(x).all() and np.isfinite(y).all()),'cosine_min':float(cosine.min()),'cosine_p5':float(np.percentile(cosine,5)),'cosine_median':float(np.median(cosine)),'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'cosines':cosine.tolist(),'norm_range_conv':[float(norms.min()),float(norms.max())],'norm_range_linear':[float(norms2.min()),float(norms2.max())]}

def prepare():
    design=load('design.json');order=[key(s) for s in design['segments']];weights=set();matrices={};hashes={}
    for v in ['conv','linear']:
        mapping={};hs={}
        for part in ['sample','rest']:
            d=load(f'{v}-{part}-r1.json');assert d['status']=='completed';weights.add(d['projection_weight_sha256'])
            matrix=np.load(OUT/f'{v}-{part}-r1.npy');assert len(matrix)==len(d['segments'])
            for s,x,h in zip(d['segments'],matrix,d['input_hashes']):assert key(s) not in mapping;mapping[key(s)]=x;hs[key(s)]=h
        assert set(mapping)==set(order)
        matrices[v]=np.stack([mapping[k] for k in order]);hashes[v]=[hs[k] for k in order];np.save(OUT/f'{v}-full.npy',matrices[v])
    assert len(weights)==1 and hashes['conv']==hashes['linear']
    repeat={}
    for v in ['conv','linear']:
        a=load(f'{v}-sample-r1.json');b=load(f'{v}-sample-r2.json');assert a['segments']==b['segments'] and a['input_hashes']==b['input_hashes'] and b['status']=='completed'
        repeat[v]=compare_arrays(np.load(OUT/f'{v}-sample-r1.npy'),np.load(OUT/f'{v}-sample-r2.npy'))
    delta=compare_arrays(matrices['conv'],matrices['linear'])
    # Informational comparison only: correctness is tested against fresh conv,
    # never gated on compatibility with pre-existing production embeddings.
    old=compare_arrays(np.load(OUT/'production-reference.npy'),matrices['conv'])
    bench.dump(OUT/'vectors.json',{'videos':len(design['media']),'segments':len(order),'inputs_identical':True,'projection_weights_identical':True,'conv_vs_linear':delta,'repeats':repeat,'production_vs_fresh_conv_informational':old})
    return design,matrices

def quality(order,grades):
    positives={m for m,g in grades.items() if g==2}
    if not positives:return None
    first=next((i+1 for i,m in enumerate(order) if m in positives),None)
    return {'first_positive_rank':first,'hit1':int(first==1),'hit5':int(first is not None and first<=5),'mrr':1/first if first else 0,'recall5':len(set(order[:5])&positives)/len(positives)}

class Identity:
    def transform_faiss_distances_hook(self,x):return x

def runtime(variant,design,matrix):
    from action_search import media_catalog
    from native_qwen import write_native_qwen,QWEN
    from wise.api.services.embedding import EmbeddingService
    from wise.api.services.project import LocalWiseProjectService
    from wise.api.services.search import LocalSearchService
    from wise.config import APIConfig
    from wise.wise_project import WiseProject
    folder=OUT/f'{variant}-wise-project';assert not folder.exists();(folder/'metadata').mkdir(parents=True)
    source=Path(design['project'])
    with sqlite3.connect((source/'metadata/internal.db').as_uri()+'?mode=ro',uri=True) as a,sqlite3.connect(folder/'metadata/internal.db') as b:a.backup(b)
    shutil.copy2(source/'thumbs.db',folder/'thumbs.db')
    started=time.perf_counter();write_native_qwen(folder,media_catalog(folder),design['segments'],matrix);write_seconds=time.perf_counter()-started
    face=bench.FACE
    shutil.copytree(source/'store'/face,folder/'store'/face)
    config=APIConfig(command='serve',project_dir=folder,query_prefix='',max_search_results=100)
    project=LocalWiseProjectService(WiseProject(folder,read_only=True),config)
    search=LocalSearchService(project,EmbeddingService({QWEN:Identity(),bench.FACE:Identity()}))
    with sqlite3.connect(folder/'metadata/internal.db') as db:ids=[(r[0],r[1]) for r in db.execute('SELECT id,media_id FROM vectors WHERE feature_extractor_id=? ORDER BY id',(QWEN,))]
    return config,project,search,ids,write_seconds

def evaluate():
    import faiss
    faiss.omp_set_num_threads(1)
    from native_qwen import QWEN
    from wise.api.standalone.search import search_output_to_response
    from wise.data_models import MediaType,ModalityType
    design,matrices=prepare()
    labels_path=bench.ROOT/design['queries_source'];labels=json.loads(labels_path.read_text());known=set(labels['labeled_media_ids'])
    querypath=bench.ROOT/design['query_vectors_source'];q=np.load(querypath);assert len(q)>=len(labels['queries']);q=q/np.linalg.norm(q,axis=1,keepdims=True)
    assert np.isfinite(q).all()
    runtimes={v:runtime(v,design,m) for v,m in matrices.items()};rows=[]
    for scope in ['labeled_40','full_library']:
        for i,query in enumerate(labels['queries']):
            if scope=='labeled_40' and query['answer_status']=='ungraded':continue
            grades={m:0 for m in known}
            for cid,g in query.get('relevance',{}).items():
                mid=int(cid.split('-')[0][1:]);grades[mid]=max(grades.get(mid,0),g)
            row={'scope':scope,'id':query['id'],'text':query['text'],'answer_status':query['answer_status']}
            all_scores=[]
            for variant,(config,project,search,ids,_) in runtimes.items():
                constraint=np.asarray([vid for vid,mid in ids if mid in known],dtype='int64') if scope=='labeled_40' else None
                raw=search.search_with_feature(np.ascontiguousarray(q[i:i+1],dtype='float32'),modality_type=ModalityType.VIDEO,feature_extractor_id=QWEN,start=0,end=min(100,len(ids)),filter_specs={},vector_id_constraint=constraint)
                response=search_output_to_response(config,project,[],MediaType.VIDEO,raw,0)
                seen=set();ranking=[]
                for h in response.video_results.merged_windows:
                    mid=int(h.media_id)
                    if mid in seen:continue
                    seen.add(mid);ranking.append({'media_id':mid,'score':float(h.distance),'thumbnail_ts':float(h.thumbnail_ts),'ts':float(h.ts),'te':float(h.te),'vector_id':int(h.vector_id)})
                order=[r['media_id'] for r in ranking]
                row[variant]={'top10':ranking[:10],'metrics':quality(order,grades) if scope=='labeled_40' else None}
                # Score differences over every vector, independently of top-100 truncation.
                all_scores.append(matrices[variant]@q[i])
            a=row['conv']['top10'];b=row['linear']['top10']
            row.update(top1_same=a[0]['media_id']==b[0]['media_id'],top5_same_order=[x['media_id'] for x in a[:5]]==[x['media_id'] for x in b[:5]],top5_overlap=len({x['media_id'] for x in a[:5]}&{x['media_id'] for x in b[:5]})/5,top1_segment_same=a[0]['vector_id']==b[0]['vector_id'],max_abs_score_difference=float(np.max(np.abs(all_scores[0]-all_scores[1]))))
            rows.append(row)
    summary=[]
    for scope in ['labeled_40','full_library']:
        rr=[r for r in rows if r['scope']==scope];answered=[r for r in rr if r['conv']['metrics'] is not None]
        entry={'scope':scope,'queries':len(rr),'top1_same':sum(r['top1_same'] for r in rr),'top5_same_order':sum(r['top5_same_order'] for r in rr),'mean_top5_overlap':statistics.mean(r['top5_overlap'] for r in rr),'top1_segment_same':sum(r['top1_segment_same'] for r in rr),'changed_query_ids':[r['id'] for r in rr if not r['top5_same_order'] or not r['top1_segment_same']]}
        if answered:
            entry['answerable_queries']=len(answered)
            for v in ['conv','linear']:entry[v]={k:sum(r[v]['metrics'][k] for r in answered) if k.startswith('hit') else statistics.mean(r[v]['metrics'][k] for r in answered) for k in ['hit1','hit5','mrr','recall5']}
            entry['worse_first_positive_rank']=[r['id'] for r in answered if (r['linear']['metrics']['first_positive_rank'] or 999)>(r['conv']['metrics']['first_positive_rank'] or 999)]
        summary.append(entry)
    bench.dump(OUT/'quality.json',{'summary':summary,'queries':rows,'query_vectors_sha256':bench.sha(querypath),'labels_sha256':bench.sha(labels_path),'wise_write_validate_seconds':{v:r[-1] for v,r in runtimes.items()},'method':'Fresh full galleries; native WISE store write and validation, LocalSearchService.search_with_feature and search_output_to_response merged_windows. Fixed query vectors. No person fusion or frontend HTTP replay.'})
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    fusion(design,labels,q,runtimes)

def fusion(design,labels,queries,runtimes):
    from action_search import ActionEngine
    from combined_search import person_scene_search
    from library_search import normalize
    from wise.api.standalone.search import search_output_to_response
    from wise.data_models import MediaType
    # Do not persist names, photos or face embeddings in experiment results.
    with sqlite3.connect((bench.ROOT/'data/people.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        people=[np.frombuffer(row[0],dtype='float32').reshape(1,-1).copy() for row in db.execute('SELECT vector FROM people WHERE model=? ORDER BY rowid',(bench.FACE,))]
    engines={v:ActionEngine() for v in runtimes}
    catalog=[{'id':m['id'],'name':m['path'],'duration':m['duration']} for m in design['media']]
    comparisons=[]
    for pi,face in enumerate(people):
        for qi,query in enumerate(labels['queries']):
            row={'registration_ordinal':pi+1,'query_id':query['id']}
            evidence_by_variant={}
            for variant,(config,project,search,_,_) in runtimes.items():
                engine=engines[variant];engine.query_vector=lambda text,vector=queries[qi:qi+1]:(vector,True)
                raw,evidence=person_scene_search(face,query['text'],.45,face_model=bench.FACE,search=search,actions=engine,project=project.wise_project._project_dir)
                payload=search_output_to_response(config,project,[],MediaType.VIDEO,raw,1).model_dump(mode='json')
                if payload.get('video_results'):
                    for hit in payload['video_results']['unmerged_windows']:hit.update(evidence.get(str(hit['vector_id']),{}))
                result=normalize(payload,catalog,'person_scene',20,.45)
                row[variant]=[{'media_id':int(x['id']),'start':x['start'],'end':x['end'],'scene_ts':x.get('scene_ts'),'scene_te':x.get('scene_te')} for x in result['items']]
                evidence_by_variant[variant]=evidence
            a,b=row['conv'],row['linear'];row['same_video_order']=[x['media_id'] for x in a]==[x['media_id'] for x in b]
            row['same_windows']=a==b;row['same_result_set']={x['media_id'] for x in a}=={x['media_id'] for x in b}
            ea,eb=evidence_by_variant['conv'],evidence_by_variant['linear']
            assert set(ea)==set(eb) and all(ea[k]['face_similarity']==eb[k]['face_similarity'] for k in ea),'Face evidence unexpectedly changed'
            row['max_scene_score_delta']=max((abs(ea[k]['scene_similarity']-eb[k]['scene_similarity']) for k in ea),default=0)
            comparisons.append(row)
    report={'registrations':len(people),'queries_per_registration':len(labels['queries']),'comparisons':len(comparisons),'nonempty_baseline':sum(bool(r['conv']) for r in comparisons),'nonempty_linear':sum(bool(r['linear']) for r in comparisons),'same_video_order':sum(r['same_video_order'] for r in comparisons),'same_result_set':sum(r['same_result_set'] for r in comparisons),'same_windows':sum(r['same_windows'] for r in comparisons),'face_evidence_identical':True,'changes':[r for r in comparisons if not r['same_windows']],'scope':'Existing face vectors, person_scene_search fusion, WISE response, and current semantic thresholds/normalize. Fixed scene query vectors. Stability only; no exhaustive identity+scene ground truth.'}
    bench.dump(OUT/'person-scene-quality.json',report)
    print('person-scene',json.dumps({k:v for k,v in report.items() if k!='changes'},ensure_ascii=False))

if __name__=='__main__':evaluate()
