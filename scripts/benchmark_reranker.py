"""Reproducible live API timings and fixed, labelled ten-clip comparison.

Run with scripts/env.sh. --fixture requires the demo server to be stopped to
avoid duplicate model memory. It uses archived embedding rankings on the exact
same hashed clips, never labels or filenames in the reranker's input.
"""
import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/reranker'
QUERIES = ['作业人员没有佩戴安全帽', '工人佩戴安全帽', '人员翻越围栏', '一个人失去平衡摔倒在地']


def write(name, report):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(report, ensure_ascii=False, indent=2))


def live():
    project = (ROOT/'projects/demo').resolve()
    report = {'project': str(project), 'created':time.time(), 'runs':[], 'status':'running'}
    for query in QUERIES:
        # Cold here means first request for this query; actual load times are recorded.
        for label, enabled in [('original',False), ('original_repeat',False),
                               ('rerank_first',True), ('rerank_repeat',True)]:
            started = time.perf_counter()
            form = urllib.parse.urlencode({'query':query,'rerank':str(enabled).lower(),
                                          'limit':10,'rerank_candidates':10}).encode()
            result = json.load(urllib.request.urlopen('http://127.0.0.1:9670/mvp-api/actions/search',
                                                       form, timeout=300))
            elapsed = time.perf_counter()-started
            if enabled and not result['rerank']['applied']:
                raise RuntimeError(result['rerank'].get('error','rerank failed'))
            report['runs'].append({'query':query,'mode':label,'http_seconds':elapsed,'result':result})
            write('live.json',report)
            top = result['video_results']['unmerged_windows'][:3]
            print(query, label, round(elapsed,3), [(h['media_id'],h['ts'],round(h.get('rerank_score',h['distance']),4)) for h in top],flush=True)
    assert (ROOT/'projects/demo').resolve() == project, 'Library changed during benchmark'
    report['status']='completed';write('live.json',report)


def fixture():
    from action_reranker import VideoReranker, ResourceSampler, SETTINGS, read_frames
    try:
        urllib.request.urlopen('http://127.0.0.1:9670/mvp-api/actions/status',timeout=1)
    except OSError:
        pass
    else:
        raise RuntimeError('Stop the demo before running the isolated fixture benchmark')
    folder = ROOT/'experiments/action-search'
    manifest = json.loads((folder/'manifest.json').read_text())
    baseline = json.loads((folder/'qwen-fp16.json').read_text())
    items = {c['id']:c for c in manifest['clips']}
    for c in items.values():
        assert hashlib.sha256((folder/c['file']).read_bytes()).hexdigest()==c['sha256']
    model = VideoReranker()
    report = {'created':time.time(), 'settings':SETTINGS, 'status':'running', 'queries':[],
              'baseline':'../action-search/qwen-fp16.json (archived ranking, not new timing)',
              'limitations':manifest['notes'], 'clips':manifest['clips']}
    for query in QUERIES:
        q = next(q for q in baseline['queries'] if q['text']==query)
        catalog = []
        hits = []
        for rank, h in enumerate(q['ranking']):
            c=items[h['id']]
            clip=next(c for c in baseline['clips'] if c['id']==h['id'])
            catalog.append({'id':h['id'],'location':str(folder),'path':c['file']})
            hits.append({'media_id':h['id'],'ts':0.,'te':clip['duration'],'distance':h['score']})
        with ResourceSampler() as resources:
            ranked, timing = model.rank(query,hits,catalog)
        expected=set(q['expected'])
        def metrics(rows):
            flags=[h['media_id'] in expected for h in rows]
            return {'first_expected_rank':next((i+1 for i,v in enumerate(flags) if v),None),
                    'precision_at_3':sum(flags[:3])/3}
        entry={'query':query,'expected':q['expected'],'original':hits,'reranked':ranked,
               'original_metrics':metrics(hits),'reranked_metrics':metrics(ranked),
               'timing':timing,'resources':resources.result}
        report['queries'].append(entry);write('fixture.json',report)
        print(query, entry['original_metrics'],entry['reranked_metrics'],
              [(h['media_id'],round(h['rerank_score'],5)) for h in ranked[:5]],flush=True)
        if len(report['queries']) == 1:
            # Exercise the reviewed, pinned official process() against the same
            # weights/pixels. Only bypass its CUDA/CPU loader and list resampler.
            import importlib.util
            import torch
            source=OUT/'reference/qwen3_vl_reranker.py'
            spec=importlib.util.spec_from_file_location('qwen_official_reference',source)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            ref=module.Qwen3VLReranker.__new__(module.Qwen3VLReranker)
            ref.model=model.model;ref.processor=model.processor
            ref.max_length=8192;ref.min_pixels=4096;ref.max_pixels=1310720
            ref.total_pixels=SETTINGS['max_total_pixels'];ref.fps=1
            ref.num_frames=ref.max_frames=None;ref.default_instruction=SETTINGS['instruction']
            ref.score_linear=torch.nn.Linear(model.score_weight.numel(),1,bias=False,device='mps',dtype=torch.float16)
            with torch.no_grad():ref.score_linear.weight.copy_(model.score_weight.unsqueeze(0))
            frames=read_frames(catalog[0],hits[0])
            official=ref.process({'query':{'text':query},'documents':[{'video':frames}]})[0]
            local=next(h['rerank_score'] for h in ranked if h['media_id']==hits[0]['media_id'])
            report['official_parity']={'clip':hits[0]['media_id'],'official_score':official,
                'local_score':local,'absolute_difference':abs(official-local),'tolerance':.002}
            write('fixture.json',report)
            assert abs(official-local)<.002, report['official_parity']
            print('OFFICIAL_PARITY',report['official_parity'],flush=True)
            del ref
    model.unload()
    report['status']='completed';write('fixture.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fixture',action='store_true')
    args=parser.parse_args()
    fixture() if args.fixture else live()
