"""Read-only pre/post restart checks. Never uploads or changes registrations."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import requests

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from bench import fingerprint, dump, OWL, FACE


def snapshot():
    base='http://127.0.0.1:9670'
    response=requests.get(base+'/mvp-api/jobs',timeout=15);response.raise_for_status()
    active=[j for j in response.json()['jobs'] if j['status'] not in ('completed','failed')]
    if active:raise RuntimeError('Active uploads; do not restart yet')
    state=requests.get(base+'/mvp-api/actions/status',timeout=15);state.raise_for_status()
    assert state.json()['status']=='ready' and state.json()['model_loaded']
    checks=[]
    cases=[('text-cycle','骑自行车',None,None),('text-run','户外慢跑',None,None),
           ('object',None,'backpack-independent.jpg',OWL),
           ('face',None,'coughing-6XvsLPDioVA_000000_000010-preview.jpg',FACE)]
    for label,query,photo,model in cases:
        if query:
            response=requests.post(base+'/mvp-api/search',data={'query':query},timeout=120)
        else:
            response=requests.post(base+'/demo/search',params={'search_in':'video',
                'feature_extractor_id':model,'end':100,'thumbnails_to_send':0,'add_prefix':'false'},
                files={'image_file_queries':(photo,(ROOT/'data/queries'/photo).read_bytes(),'image/jpeg')},timeout=120)
        response.raise_for_status()
        hits=response.json()['video_results']['unmerged_windows']
        assert hits,label
        checks.append({'label':label,'count':len(hits),'top10':[
            {key:hit.get(key) for key in ('media_id','ts','te','distance','bbox')}
            for hit in hits[:10]]})
    page=requests.get(base+'/mvp/',timeout=15);page.raise_for_status()
    return {'checked_epoch':time.time(),'server_pid':(ROOT/'logs/server.pid').read_text().strip(),
            'qwen_status':state.json(),'queries':checks,'production':fingerprint()}


def main(phase):
    output=HERE/'deployment-20261028';output.mkdir(exist_ok=True)
    path=output/(phase+'.json')
    if path.exists():raise ValueError('Do not overwrite deployment evidence')
    result=snapshot();dump(path,result)
    if phase=='after':
        before=json.loads((output/'before.json').read_text())
        assert result['production']==before['production'],'Production data/config changed across restart'
        assert result['server_pid']!=before['server_pid'],'Service was not restarted'
        for old,new in zip(before['queries'],result['queries']):
            assert old['label']==new['label'] and old['count']==new['count']
            for a,b in zip(old['top10'],new['top10']):
                assert {k:v for k,v in a.items() if k!='distance'}=={k:v for k,v in b.items() if k!='distance'}
                np.testing.assert_allclose(a['distance'],b['distance'],rtol=0,atol=1e-6)
        dump(output/'verification.json',{'status':'passed','service_restarted':True,
            'query_rankings_and_metadata_unchanged':True,'score_absolute_tolerance':1e-6,
            'production_unchanged':True,'videos':result['qwen_status']['indexed_videos'],
            'qwen_segments':result['qwen_status']['indexed_segments']})
    print(phase, 'verified',result['server_pid'],[(r['label'],r['count']) for r in result['queries']],flush=True)


if __name__=='__main__':main(sys.argv[1])
