"""Evaluate immutable labels, validate vectors/alignment, audit production and export all ranks."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import statistics
import ast
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ASSETS=ROOT/'experiments/qwen8b-int8'

def load(name):return json.loads((HERE/name).read_text())
def save(name,data):(HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def score(order,grades):
    rel={k for k,g in grades.items() if g==2}
    if not rel:return None
    first=next((i+1 for i,k in enumerate(order) if k in rel),None)
    out={f'hit@{k}':int(bool(set(order[:k])&rel)) for k in [1,3,5]}
    out.update({f'recall@{k}':len(set(order[:k])&rel)/len(rel) for k in [5,10]})
    out['mrr']=1/first if first else 0
    def dcg(gs):return sum((2**g-1)/math.log2(i+2) for i,g in enumerate(gs))
    out['ndcg@5']=dcg([grades[k] for k in order[:5]])/dcg(sorted(grades.values(),reverse=True)[:5])
    return out

def grouped(order,grades,key):
    grouped_grades={}
    for cid,g in grades.items():grouped_grades[key(cid)]=max(grouped_grades.get(key(cid),0),g)
    seen=set();collapsed=[]
    for cid in order:
        k=key(cid)
        if k not in seen:collapsed.append(k);seen.add(k)
    return score(collapsed,grouped_grades)

def aggregate(rows):
    return {'queries':len(rows),**{k:statistics.mean(x[k] for x in rows) for k in rows[0]}} if rows else {}

def stats(values):
    return {'n':len(values),'sum':sum(values),'median':float(np.median(values)),
        'p95':float(np.percentile(values,95)),'min':min(values),'max':max(values)}

def audit():
    before=load('production-before.json')
    after={p:sha(p) for p in before['files']}
    changed=[p for p,h in after.items() if before['files'][p]!=h]
    # Concurrent workspace work changed result grouping, not sampling or encoding.
    # Keep the discrepancy visible; never waive model, index, or pointer changes.
    concurrent_code_changes=[p for p in changed if p==str(ROOT/'scripts/action_search.py')]
    protected_changes=[p for p in changed if p not in concurrent_code_changes]
    concurrent_diff_verified=True
    if concurrent_code_changes:
        original=(HERE/'production-action-search-before.txt').read_text()
        assert hashlib.sha256(original.encode()).hexdigest()==before['files'][str(ROOT/'scripts/action_search.py')]
        def protected_ast(source):
            tree=ast.parse(source)
            tree.body=[n for n in tree.body if not (isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='select_hits')]
            for node in tree.body:
                if isinstance(node,ast.ClassDef) and node.name=='ActionEngine':
                    node.body=[n for n in node.body if not (isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='search')]
            return ast.dump(tree,include_attributes=False)
        concurrent_diff_verified=protected_ast(original)==protected_ast((ROOT/'scripts/action_search.py').read_text())
        assert concurrent_diff_verified,'Concurrent edit affected code outside result selection; inspect before accepting audit'
    pointer_same=os.readlink(ROOT/'projects/demo')==before['pointer']
    source_changes=[s['resolved_path'] for s in load('dataset.json')['sources'] if sha(s['resolved_path'])!=s['sha256']]
    result={'project_pointer_unchanged':pointer_same,'checked_file_count':len(after),'changed_files':changed,
        'concurrent_code_changes':concurrent_code_changes,'model_index_config_changes':protected_changes,
        'concurrent_change_limited_to_result_selection':concurrent_diff_verified,
        'changed_source_videos':source_changes,'checked_source_video_count':40,'files':after}
    save('production-after.json',result)
    assert pointer_same and not protected_changes and not source_changes,'Production/source changed; inspect audit'
    return result

def evaluate(name,data):
    result=load(name+'.json');resources=load(name+'-resources.json')
    assert result['status']=='completed' and resources['exit_code']==0
    assert len(result['clips'])==129 and len(result['queries'])==204
    matrix=np.load(HERE/(name+'-vectors.npy'));queries=np.load(HERE/(name+'-query-vectors.npy'))
    dim=2048 if name=='2b-fp16' else 4096
    assert matrix.shape==(129,dim) and queries.shape==(204,dim)
    assert np.isfinite(matrix).all() and np.isfinite(queries).all()
    assert np.allclose(np.linalg.norm(matrix,axis=1),1,atol=1e-5)
    assert np.allclose(np.linalg.norm(queries,axis=1),1,atol=1e-5)
    first=result['queries'][:68];clips={c['id']:c for c in data['clips']};rows=[]
    for query,raw in zip(data['queries'],first):
        assert query['id']==raw['id'] and query['text']==raw['text']
        order=[r['id'] for r in raw['ranking']];grades=query['relevance']
        # Validate saved FAISS rankings against stored vectors, allowing exact ties.
        scores=matrix@queries[raw['query_index']]
        index={c['id']:i for i,c in enumerate(data['clips'])}
        observed=np.array([r['score'] for r in raw['ranking']])
        assert np.allclose(observed,[scores[index[c]] for c in order],atol=2e-5)
        assert all(a>=b-1e-6 for a,b in zip(observed,observed[1:]))
        def source_key(cid):
            c=clips[cid];mid=c['media_id']
            if mid in [3,34]:return 'source-car'
            if mid in [14,39]:return 'source-park'
            if mid in [15,16]:return 'source-toddler'
            if query['id'] in ['B02','B17','H10','P02'] and (mid==2 or (mid==17 and c['ts'] in [4,8,12])):return 'source-backpack'
            return f'media-{mid}'
        # Same near-overlap suppression as production, applied only as a supplemental view.
        kept=[]
        for cid in order:
            c=clips[cid]
            duplicate=any(c['media_id']==clips[k]['media_id'] and
                max(0,min(c['te'],clips[k]['te'])-max(c['ts'],clips[k]['ts']))/
                max(1e-6,max(c['te'],clips[k]['te'])-min(c['ts'],clips[k]['ts']))>=.5 for k in kept)
            if not duplicate:kept.append(cid)
        row={'id':query['id'],'text':query['text'],'group':query['group'],
            'segment':score(order,grades),'video':grouped(order,grades,lambda cid:clips[cid]['media_id']),
            'source_group':grouped(order,grades,source_key),'production_overlap_filter':score(kept,grades),
            'first_full_match_rank':next((i+1 for i,cid in enumerate(order) if grades[cid]==2),None),
            'top5':[{'segment':r['id'],'file':clips[r['id']]['file'],'ts':clips[r['id']]['ts'],
                'te':clips[r['id']]['te'],'score':r['score'],'grade':grades[r['id']]} for r in raw['ranking'][:5]]}
        rows.append(row)
    groups={}
    for group,prefixes in [('main',('B','H')),('basic',('B',)),('hard',('H',)),('paraphrase',('P',)),('all_answerable',('B','H','P'))]:
        selected=[r for r in rows if r['id'].startswith(prefixes)]
        groups[group]={level:aggregate([r[level] for r in selected]) for level in ['segment','video','source_group','production_overlap_filter']}
    paired=[]
    by_id={r['id']:r for r in rows}
    for q in data['queries']:
        if q['paired_with']:
            base=by_id[q['paired_with']]; para=by_id[q['id']]
            paired.append({'base':base['id'],'paraphrase':para['id'],
                'base_hit1':base['segment']['hit@1'],'paraphrase_hit1':para['segment']['hit@1'],
                'same_top1_video':clips[base['top5'][0]['segment']]['media_id']==clips[para['top5'][0]['segment']]['media_id'],
                'same_top1_segment':base['top5'][0]['segment']==para['top5'][0]['segment']})
    warm=[q for q in result['queries'] if q['round']>0]
    phases={}
    for phase in ['loading','video','video_repeat','query']:
        samples=[x for x in resources['samples'] if x.get('phase')==phase]
        if samples:phases[phase]={'peak_rss_bytes':max(x['rss_bytes'] for x in samples),
            'peak_footprint_bytes':max(x['physical_footprint_bytes'] for x in samples),
            'minimum_available_bytes':min(x['system_available_bytes'] for x in samples),
            'initial_swap_bytes':samples[0]['swap_used_bytes'],'peak_swap_bytes':max(x['swap_used_bytes'] for x in samples)}
    phases['video']['peak_metal_bytes']=result['video_phase_memory']['sampled_metal_peak_bytes']
    phases['query']['peak_metal_bytes']=result['final_memory']['sampled_metal_peak_bytes']
    stable=all([r['id'] for r in q['ranking']]==[r['id'] for r in first[q['query_index']]['ranking']] for q in result['queries'])
    output={'quality':groups,'paired_paraphrases':paired,'queries':rows,
        'performance':{'model_load_seconds':result['model_load_seconds'],
            'all_segments_wall_seconds':result['all_segments_wall_seconds'],
            'all_segments_encode_seconds':sum(c['encode_seconds'] for c in result['clips']),
            'segment_encoding':stats([c['encode_seconds'] for c in result['clips']]),
            'warm_segment_encoding':stats([c['encode_seconds'] for c in result['clips'][1:]]),
            'first_query_seconds':result['queries'][0]['encode_seconds'],
            'warm_query_encoding':stats([q['encode_seconds'] for q in warm]),
            'query_rounds':{str(i):stats([q['encode_seconds']+q['search_seconds'] for q in result['queries'] if q['round']==i]) for i in range(3)},
            'warm_query_with_search':stats([q['encode_seconds']+q['search_seconds'] for q in warm]),
            'search_seconds':stats([q['search_seconds'] for q in warm])},
        'resources':{'phases':phases,**{k:v for k,v in resources.items() if k!='samples'}},
        'validation':{'rankings_identical_across_rounds':stable,'vector_dimension':dim,
            'min_repeat_cosine':min(c['cosine_to_first'] for c in result['warm_video_repeats'])},
        'runtime':result['runtime']}
    if name=='2b-fp16':
        production=np.load(Path(load('production-before.json')['project'])/'action-index/vectors.npy')
        production=production/np.linalg.norm(production,axis=1,keepdims=True)
        cosine=np.sum(production*matrix,axis=1)
        output['validation']['production_vector_cosine']={'min':float(cosine.min()),'median':float(np.median(cosine)),
            'below_0999':[data['clips'][i]['id'] for i,x in enumerate(cosine) if x<.999]}
    return output,result

def main():
    frozen=load('labels-freeze.json')
    assert all(sha(HERE/p)==h for p,h in frozen['files'].items()),'Frozen labels changed'
    data=load('dataset.json');a,ra=evaluate('2b-fp16',data);b,rb=evaluate('8b-q8',data)
    reference=json.loads((ASSETS/'template-reference.json').read_text())
    expected=[x for x in reference['tokens'] if x!=151656];alignment=[]
    for ca,cb in zip(ra['clips'],rb['clips']):
        assert ca['id']==cb['id'];grid=ca['input']['video_grid_thw'][0]
        assert cb['input']['text_token_ids']==expected
        assert cb['input']['visual_chunk_tokens']==[grid[1]*grid[2]//4]*grid[0]
        assert ca['input']['tokens']==cb['input']['tokens']
        alignment.append({'segment':ca['id'],'tokens':ca['input']['tokens'],'matched':True})
    production=audit()
    candidate_manifest=json.loads((ASSETS/'model-provenance.json').read_text())
    candidate_checks=[]
    for item in candidate_manifest['files']:
        path=ASSETS/'models'/item['file']
        digest=sha(path)
        assert digest==item['sha256'] and path.stat().st_size==item['bytes'],'Candidate weight changed'
        candidate_checks.append({'file':item['file'],'sha256':digest,'bytes':path.stat().st_size})
    comparisons=[]
    for qa,qb in zip(a['queries'],b['queries']):
        item={'id':qa['id'],'query':qa['text'],'2b':qa,'8b':qb}
        if qa['segment']:
            item['delta_hit1']=qb['segment']['hit@1']-qa['segment']['hit@1']
            item['delta_mrr']=qb['segment']['mrr']-qa['segment']['mrr']
            item['delta_ndcg5']=qb['segment']['ndcg@5']-qa['segment']['ndcg@5']
        comparisons.append(item)
    summary={'2b-fp16':a,'8b-q8':b,'input_alignment':alignment,'production_model_index_config_unchanged':True,
        'concurrent_code_changes':production['concurrent_code_changes'],
        'candidate_weight_validation':candidate_checks,
        'protected_files':production['checked_file_count'],'queries':68,'main_queries':56,'segments':129,'videos':40,
        'label_freeze':frozen,'comparisons':comparisons}
    save('summary.json',summary)
    with (HERE/'query-comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['id','query','2B_rank','8B_rank','2B_top1','8B_top1','2B_nDCG5','8B_nDCG5'])
        for c in comparisons:
            qa,qb=c['2b'],c['8b'];w.writerow([c['id'],c['query'],qa['first_full_match_rank'],qb['first_full_match_rank'],
                qa['top5'][0]['segment'],qb['top5'][0]['segment'],(qa['segment'] or {}).get('ndcg@5'),(qb['segment'] or {}).get('ndcg@5')])
    with (HERE/'top5-results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['model','query_id','query','rank','file','start','end','score','relevance'])
        for name,model in [('2B',a),('8B Q8',b)]:
            for q in model['queries']:
                for rank,c in enumerate(q['top5'],1):w.writerow([name,q['id'],q['text'],rank,c['file'],c['ts'],c['te'],c['score'],c['grade']])
    print(json.dumps({name:{'main':model['quality']['main'],'performance':model['performance'],'phases':model['resources']['phases'],'validation':model['validation']} for name,model in [('2B',a),('8B',b)]},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
