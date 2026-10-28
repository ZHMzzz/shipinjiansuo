"""Offline same-window conjunction experiment; never publish or write indexes."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.action_search import ActionEngine,read_snapshot,SETTINGS
HERE=Path(__file__).resolve().parent

def fingerprint(project):
    return {str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((project/'action-index').iterdir()) if p.is_file()}

def main():
    extending = '--extend' in sys.argv
    if (HERE/'scores.json').exists() and not extending:raise SystemExit('scores.json already exists; preserve original experiment')
    design=json.loads((HERE/'design.json').read_text())
    project=(ROOT/'projects/demo').resolve(); before=fingerprint(project)
    meta,matrix,index=read_snapshot(project,complete=True)
    texts=list(dict.fromkeys(t for c in design['cases'] for t in [c['query'],*c['conditions']]))
    engine=ActionEngine(); records={}
    if extending:
        previous=json.loads((HERE/'scores.json').read_text())
        assert previous['index_after']==before and previous['project']==str(project)
        records=previous['texts']
        texts=[t for t in texts if t not in records]
    engine.query_vector('视频中的场景和动作')
    for n,text in enumerate(texts):
        t=time.perf_counter();vector,cached=engine.query_vector(text)
        seconds=time.perf_counter()-t
        records[text]={'seconds':seconds,'cached':cached,'scores':(matrix@vector[0]).tolist()}
        print(f'{n+1}/{len(texts)} {seconds:.3f}s {text}',flush=True)
        (HERE/'progress.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    report={'project':str(project),'settings':SETTINGS,'index_before':before,'index_after':fingerprint(project),
            'pointer_unchanged':(ROOT/'projects/demo').resolve()==project,'model_load_seconds':engine.last_load_seconds,
            'segments':meta['segments'],'media':[{k:v for k,v in m.items() if k!='location'} for m in meta['media']],
            'texts':records}
    assert report['index_before']==report['index_after'] and report['pointer_unchanged']
    output='scores-extended.json' if extending else 'scores.json'
    (HERE/output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (HERE/'progress.json').unlink()
    print('Completed; index and project pointer unchanged.',flush=True)
if __name__=='__main__':main()
