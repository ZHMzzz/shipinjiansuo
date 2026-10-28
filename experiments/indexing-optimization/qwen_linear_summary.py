"""Summarize measured speed and observed retrieval regression, no extrapolation."""
import json
import statistics
from pathlib import Path
import bench
from qwen_linear import OUT

def load(name):return json.loads((OUT/name).read_text())
def main():
    result={'sample':{},'full_library':{},'resources':{}}
    keys=['encode_seconds','decode_and_encode_seconds','decode_and_wrapper_seconds']
    for variant in ['conv','linear']:
        sample=[load(f'{variant}-sample-r{r}.json') for r in [1,2]]
        rest=load(f'{variant}-rest-r1.json');assert all(r['status']=='completed' for r in sample+[rest])
        result['sample'][variant]={'runs':[{k:r[k] for k in keys} for r in sample],'mean':{k:statistics.mean(r[k] for r in sample) for k in keys}}
        result['full_library'][variant]={k:sample[0][k]+rest[k] for k in keys}
        result['resources'][variant]={'peak_mps_driver_gib':max(r['resources']['sampled_peak']['mps_driver_bytes'] for r in sample+[rest])/1024**3,'peak_rss_gib':max(r['resources']['sampled_peak']['rss_bytes'] for r in sample+[rest])/1024**3,'minimum_available_gib':min(r['resources']['min_system_available_bytes'] for r in sample+[rest])/1024**3}
    result['sample']['time_reduction_percent']={k:100*(1-result['sample']['linear']['mean'][k]/result['sample']['conv']['mean'][k]) for k in keys}
    result['full_library']['time_reduction_percent']={k:100*(1-result['full_library']['linear'][k]/result['full_library']['conv'][k]) for k in keys}
    result['full_library']['scope']='Fresh complete gallery composed of sample r1 plus rest r1. Sum of per-segment durations, not one uninterrupted full upload. Excludes load, warmup, transcode, OWLv2, InsightFace, publication.'
    vector=load('vectors.json');quality=load('quality.json');fusion=load('person-scene-quality.json')
    result['vectors']={k:v for k,v in vector.items() if k not in ['repeats','production_vs_fresh_conv_informational']}
    result['quality']=quality['summary'];result['person_scene']={k:v for k,v in fusion.items() if k not in ['changes']}
    result['wise_write_validate_seconds']=quality['wise_write_validate_seconds']
    before=load('production-before.json');after=bench.fingerprint();assert before==after
    result['production_unchanged']=True
    result['source_sha256']={name:bench.sha(bench.OUT/name) for name in ['qwen_linear.py','qwen_linear_evaluate.py','qwen_linear_summary.py']}
    bench.dump(OUT/'summary.json',result)
    concise={k:v for k,v in result.items() if k!='vectors'};concise['cosine_min']=vector['conv_vs_linear']['cosine_min'];print(json.dumps(concise,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
