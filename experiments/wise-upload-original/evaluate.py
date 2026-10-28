"""Evaluate frozen video labels through WISE search/segment organization."""
import csv
import json
import math
from pathlib import Path
import shutil
import sqlite3
import statistics
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(HERE)]
from benchmark import MODEL, audit, dump, load
from wise.api.common import VectorQueryTerm
from wise.api.services.embedding import EmbeddingService
from wise.api.services.project import LocalWiseProjectService
from wise.api.services.search import LocalSearchService
from wise.api.standalone.search import search_output_to_response
from wise.config import APIConfig
from wise.data_models import MediaType,ModalityType
from wise.wise_project import WiseProject


def stats(values):
    return dict(n=len(values),sum=sum(values),median=float(np.median(values)),
                p95=float(np.percentile(values,95)),min=min(values),max=max(values)) if values else {}


def quality(order,grades):
    positives={k for k,g in grades.items() if g==2}
    if not positives:return None
    first=next((i+1 for i,k in enumerate(order) if k in positives),None)
    dcg=lambda values:sum((2**g-1)/math.log2(i+2) for i,g in enumerate(values))
    return {'hit@1':int(bool(set(order[:1])&positives)),
            'hit@3':int(bool(set(order[:3])&positives)),
            'hit@5':int(bool(set(order[:5])&positives)),
            'recall@5':len(set(order[:5])&positives)/len(positives),
            'mrr':1/first if first else 0,
            'ndcg@5':dcg([grades[k] for k in order[:5]])/dcg(sorted(grades.values(),reverse=True)[:5]),
            'first_relevant_rank':first}


def aggregates(rows):
    keys=['hit@1','hit@3','hit@5','recall@5','mrr','ndcg@5']
    return {'queries':len(rows),**{k:statistics.mean(r[k] for r in rows) for k in keys}}


class Identity:
    def transform_faiss_distances_hook(self,x):return x


def service(name):
    result=load(name+'/result.json');assert result['status']=='completed'
    folder=HERE/name;project_path=folder/'wise-project'
    if project_path.exists():shutil.rmtree(project_path)
    (project_path/'metadata').mkdir(parents=True)
    source=Path(load('design.json')['project'])
    shutil.copy2(source/'metadata/internal.db',project_path/'metadata/internal.db')
    shutil.copy2(source/'thumbs.db',project_path/'thumbs.db')
    with sqlite3.connect(project_path/'metadata/internal.db') as db:
        db.execute('PRAGMA foreign_keys=ON');db.execute('DELETE FROM vectors')
        db.executemany('INSERT INTO vectors(id,modality,feature_extractor_id,media_id,timestamp,end_timestamp) VALUES(?,?,?,?,?,?)',
            [(i+1,'VIDEO',MODEL,s['media_id'],s['ts'],s['te']) for i,s in enumerate(result['segments'])])
        db.commit();db.execute('VACUUM')
    model=project_path/'store'/MODEL;model.mkdir(parents=True)
    (model/'features').symlink_to(folder/'features',target_is_directory=True)
    (model/'index').symlink_to(folder/'index',target_is_directory=True)
    project=WiseProject(project_path,read_only=True)
    config=APIConfig(command='serve',project_dir=project_path,query_prefix='',max_search_results=100)
    ps=LocalWiseProjectService(project,config)
    search=LocalSearchService(ps,EmbeddingService({MODEL:Identity()}))
    return result,config,ps,search


def evaluate():
    design=load('design.json');sources={s['id']:s for s in design['sources']}
    known=set(design['labeled_media_ids']);current=service('current-full');original=service('original-full')
    queries=design['queries'];n=len(queries)
    qcurrent=np.load(HERE/'current-full/query-vectors.npy')[:n]
    qoriginal=np.load(HERE/'original-full/query-vectors.npy')[:n]
    live=np.load(HERE/'production-vectors.npy');fresh=np.load(HERE/'current-full/vectors.npy')
    assert live.shape==fresh.shape
    current_check=dict(max_abs_error=float(np.max(np.abs(live-fresh))),
        min_cosine=float(np.min(np.sum(live*fresh,axis=1))),
        segments_match=[(s['media_id'],s['ts'],s['te']) for s in current[0]['segments']]==
                       [(s['media_id'],s['ts'],s['te']) for s in design['production_segments']])
    assert current_check['segments_match'] and current_check['min_cosine']>.99999,current_check
    variants={'A_current':(current,qcurrent),'B_original_upload':(original,qcurrent),'C_original_both':(original,qoriginal)}
    rows=[];timings={k:[] for k in variants}
    for name,(runtime,vectors) in variants.items():
        result,config,project,search=runtime
        constraint=np.array([i+1 for i,s in enumerate(result['segments']) if s['media_id'] in known],dtype='int64')
        for qi,query in enumerate(queries):
            ungraded=query['answer_status']=='ungraded'
            allowed=None if ungraded else constraint
            vector=np.ascontiguousarray(vectors[qi:qi+1],dtype='float32');vector/=np.linalg.norm(vector)
            started=time.perf_counter()
            raw=search.search_with_feature(vector,modality_type=ModalityType.VIDEO,feature_extractor_id=MODEL,
                start=0,end=min(100,len(result['segments'])),filter_specs={},vector_id_constraint=allowed)
            response=search_output_to_response(config,project,[],MediaType.VIDEO,raw,0)
            seconds=time.perf_counter()-started;timings[name].append(seconds)
            ranked=[];seen=set()
            for h in response.video_results.merged_windows:
                mid=int(h.media_id)
                if mid in seen:continue
                seen.add(mid);ranked.append(dict(media_id=mid,name=sources[mid]['path'],score=h.distance,
                    start=h.thumbnail_ts,merged_start=h.ts,end=h.te,vector_id=h.vector_id))
            grades={mid:0 for mid in known}
            for cid,grade in query.get('relevance',{}).items():
                mid=int(cid.split('-')[0][1:]);grades[mid]=max(grades[mid],grade)
            row=dict(variant=name,query_id=query['id'],text=query['text'],group=query['group'],
                answer_status=query['answer_status'],candidate_segments=len(raw.ids),video_count=len(ranked),
                ranking=ranked,metrics=None if ungraded else quality([r['media_id'] for r in ranked],grades),
                native_search_response_seconds=seconds)
            rows.append(row)
    summary={'baseline_vector_validation':current_check,'variants':{},'performance':{},'source_videos':len(sources),
             'labeled_videos':len(known),'labeled_queries':68,'extra_diagnostic_queries':5}
    for name in variants:
        selected=[r for r in rows if r['variant']==name and r['metrics']]
        summary['variants'][name]={'all':aggregates([r['metrics'] for r in selected]),
            'groups':{g:aggregates([r['metrics'] for r in selected if r['group']==g]) for g in sorted({r['group'] for r in selected})},
            'native_search_response_seconds':stats(timings[name])}
    for name in ['current-full','original-full','current-check']:
        r=load(name+'/result.json');assert r['status']=='completed'
        summary['performance'][name]=dict(segments=len(r['segments']),
            video_seconds=sum(s['duration'] for s in design['sources'] if s['id'] in {v['media_id'] for v in r['videos']}),
            main_video_wall_seconds=sum(v['wall_seconds'] for v in r['videos'] if not v['repeat']),
            main_segment_encode_seconds=sum(s['encode_seconds'] for s in r['segments']),
            repeat_video_wall_seconds=sum(v['wall_seconds'] for v in r['videos'] if v['repeat']),
            model_load_seconds=r['model_load_seconds'],feature_store_seconds=r['feature_store_seconds'],
            faiss_build_seconds=r['faiss_build_seconds'],
            query_all_seconds=stats([q['seconds'] for q in r['queries']]),
            query_first_round_seconds=stats([q['seconds'] for q in r['queries'] if q['round']==0]),
            query_repeated_seconds=stats([q['seconds'] for q in r['queries'] if q['round']>0]),
            resources=r['resources'])
    a={r['query_id']:r for r in rows if r['variant']=='A_current'}
    summary['changes']={}
    for variant in ['B_original_upload','C_original_both']:
        changes=[]
        for b in [r for r in rows if r['variant']==variant and r['metrics']]:
            prev=a[b['query_id']]
            if b['metrics']!=prev['metrics']:
                changes.append(dict(query_id=b['query_id'],text=b['text'],before=prev['metrics'],after=b['metrics'],
                    before_top=prev['ranking'][:5],after_top=b['ranking'][:5]))
        summary['changes'][variant]=changes
    dump(HERE/'query-results.json',rows);dump(HERE/'summary.json',summary)
    with (HERE/'query-comparison.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['variant','query_id','text','first_relevant_rank','hit@1','hit@5','top_video','top_score'])
        for r in rows:
            m=r['metrics'] or {};writer.writerow([r['variant'],r['query_id'],r['text'],m.get('first_relevant_rank'),m.get('hit@1'),m.get('hit@5'),r['ranking'][0]['name'],r['ranking'][0]['score']])
    for runtime in (current,original):
        runtime[2].wise_project.db_engine.dispose();runtime[2].wise_project.thumbsdb_engine.dispose()
    audit()
    print(json.dumps({k:v['all'] for k,v in summary['variants'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':evaluate()
