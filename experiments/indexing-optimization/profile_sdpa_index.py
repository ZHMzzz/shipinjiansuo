"""Separate import/startup from actual index creation on an isolated full clone."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'scripts')]
OUT=HERE/'sdpa-current-profile-20261028'


def child(args):
    start=time.perf_counter();rows=[]
    from wise.index.feature_search_index import FeatureSearchIndex
    original=FeatureSearchIndex.create_index
    def create(self,*a,**kw):
        t=time.perf_counter()
        try:return original(self,*a,**kw)
        finally:rows.append({'feature_id':self.feature_extractor_id,'seconds':time.perf_counter()-t,'offset_seconds':t-start})
    FeatureSearchIndex.create_index=create
    import runpy
    sys.argv=['wise',*args]
    try:runpy.run_module('wise',run_name='__main__')
    finally:(OUT/'index-profile.json').write_text(json.dumps({'body_seconds':time.perf_counter()-start,'create_calls':rows},indent=2))


def main():
    from bench import fingerprint,dump
    from native_qwen import suspend_native_store
    import shutil
    before=fingerprint();folder=Path(tempfile.mkdtemp(prefix='index-',dir=OUT));project=folder/'project';report=[]
    try:
        subprocess.run(['/bin/cp','-cR',before['project'],str(project)],check=True)
        with suspend_native_store(project):
            for mode in ('baseline','profile'):
                cmd=([sys.executable,'-u','-m','wise'] if mode=='baseline' else [sys.executable,'-u',__file__,'child'])+['create-index','--project-dir',str(project),'--overwrite']
                t=time.perf_counter()
                with (OUT/('index-'+mode+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                report.append({'mode':mode,'wall_seconds':time.perf_counter()-t})
        assert fingerprint()==before
        dump(OUT/'index-isolated.json',{'trials':report,'production_unchanged':True,'scope':'existing 44-video clone; Qwen store temporarily hidden; independent from model-loaded upload run'})
        print(report)
    finally:shutil.rmtree(folder)

if __name__=='__main__':child(sys.argv[2:]) if sys.argv[1:] and sys.argv[1]=='child' else main()
