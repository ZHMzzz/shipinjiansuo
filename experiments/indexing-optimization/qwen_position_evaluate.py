"""Compare fresh Linear baseline vs cache; reuse real WISE retrieval evaluator."""
import json
import statistics
import numpy as np
import bench
from qwen_position_cache import OUT

def load(name):return json.loads((OUT/name).read_text())
def key(s):return (s['media_id'],s['ts'],s['te'])

def main():
    design=load('design.json');order=[key(s) for s in design['segments']]
    matrices={};hashes={};summary={'sample':{},'full_library':{},'resources':{},'cache':{}}
    for variant in ('baseline','cached'):
        runs=[load(f'{variant}-sample-r{r}.json') for r in (1,2)]
        rest=load(f'{variant}-rest-r1.json')
        assert all(r['status']=='completed' for r in runs+[rest])
        mapping={};hs={}
        for part,d in [('sample',runs[0]),('rest',rest)]:
            vectors=np.load(OUT/f'{variant}-{part}-r1.npy')
            for s,v,h in zip(d['segments'],vectors,d['input_hashes']):
                assert key(s) not in mapping
                mapping[key(s)]=v;hs[key(s)]=h
        assert set(mapping)==set(order)
        matrices[variant]=np.stack([mapping[k] for k in order]);hashes[variant]=[hs[k] for k in order]
        np.save(OUT/f'{variant}-full.npy',matrices[variant])
        summary['sample'][variant]={k:statistics.mean(d[k] for d in runs) for k in ('encode_seconds','decode_and_encode_seconds')}
        summary['sample'][variant]['runs_seconds']=[d['encode_seconds'] for d in runs]
        summary['sample'][variant]['repeat_vectors_exact']=bool(np.array_equal(np.load(OUT/f'{variant}-sample-r1.npy'),np.load(OUT/f'{variant}-sample-r2.npy')))
        summary['full_library'][variant]={k:runs[0][k]+rest[k] for k in ('encode_seconds','decode_and_encode_seconds')}
        summary['resources'][variant]={'peak_mps_driver_gib':max(d['resources']['sampled_peak']['mps_driver_bytes'] for d in runs+[rest])/1024**3,
            'peak_rss_gib':max(d['resources']['sampled_peak']['rss_bytes'] for d in runs+[rest])/1024**3,
            'minimum_available_gib':min(d['resources']['min_system_available_bytes'] for d in runs+[rest])/1024**3}
        summary['cache'][variant]=[d['cache'] for d in runs+[rest]]
    assert hashes['baseline']==hashes['cached']
    for scope in ('sample','full_library'):
        summary[scope]['time_reduction_percent']={k:100*(1-summary[scope]['cached'][k]/summary[scope]['baseline'][k]) for k in ('encode_seconds','decode_and_encode_seconds')}
    summary['vector_comparison']={'exact_equal':bool(np.array_equal(matrices['baseline'],matrices['cached'])),
        'max_abs':float(np.abs(matrices['baseline']-matrices['cached']).max()),'videos':len(design['media']),'segments':len(order),'inputs_identical':True}
    # Adapter only: evaluator's legacy variable names conv/linear are renamed
    # in saved evidence. BOTH variants here already use production Linear.
    import qwen_linear_evaluate as evaluate
    evaluate.OUT=OUT
    evaluate.prepare=lambda:(design,{'conv':matrices['baseline'],'linear':matrices['cached']})
    original_runtime=evaluate.runtime
    evaluate.runtime=lambda variant,design,matrix:original_runtime(
        {'conv':'baseline','linear':'cached'}[variant],design,matrix)
    evaluate.evaluate()
    def rename(value):
        if isinstance(value,dict):return {k.replace('conv','baseline').replace('linear','cached'):rename(v) for k,v in value.items()}
        if isinstance(value,list):return [rename(v) for v in value]
        return value
    for name in ('quality.json','person-scene-quality.json'):
        d=rename(load(name));d['variants']={'baseline':'Production Linear without position cache','cached':'Production Linear with position cache'};bench.dump(OUT/name,d)
    summary['quality']=load('quality.json')['summary']
    summary['person_scene']=load('person-scene-quality.json')
    summary['scope']='Qwen per-segment cumulative time; full gallery composed of sample r1 plus rest r1; cold cache after model warmup in each process. Excludes model load, transcode, OWL/face and publication.'
    summary['production_unchanged']=load('production-before.json')==bench.fingerprint()
    assert summary['production_unchanged']
    bench.dump(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
