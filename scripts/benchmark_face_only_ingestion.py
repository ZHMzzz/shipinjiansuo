"""Measure retained WISE face-only stages in disposable project copies.

No production publication; Qwen/transcode costs remain common and are not
included. Historical full-pipeline logs are a separate observational baseline.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

from benchmark_qwen_replacement import ROOT,fingerprint

OUT=ROOT/'experiments/qwen-ingestion-estimate'
JOBS=['0292b4b6f8f34a9696c91335cd35f56b','21ebf62dda0a43d787470765fab53d3f']


def main():
    before=json.loads((OUT/'before.json').read_text())
    report={'snapshot':before['project'],'runs':[],'status':'running'}
    def save(): (OUT/'face-only.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    with sqlite3.connect(ROOT/'data/ingestion.sqlite') as db:
        db.row_factory=sqlite3.Row
        jobs=[dict(db.execute('SELECT * FROM jobs WHERE id=?',(j,)).fetchone()) for j in JOBS]
    for job in jobs:
        folder=OUT/job['id'];folder.mkdir(exist_ok=True);stage=folder/'project';prepared=folder/'prepared';prepared.mkdir(exist_ok=True)
        assert not stage.exists(),'Never reuse a partially processed project'
        source=next((ROOT/'data/uploads'/job['id']/'prepared').glob('*.mp4'))
        shutil.copy2(source,prepared/source.name)
        started=time.perf_counter();shutil.copytree(before['project'],stage);copy_seconds=time.perf_counter()-started
        # Feature extractor discovery uses store folders. Only disposable copies
        # lose these assets; archived vector metadata is irrelevant to timing.
        for name in ['mlfoundations','transformers']:
            shutil.rmtree(stage/'store'/name)
        row={'job_id':job['id'],'filename':job['filename'],'duration':job['duration'],
             'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
             'historical_queued_to_ready_seconds':job['updated']-job['created'],
             'copy_seconds':copy_seconds}
        try:
            for phase,command in [('extract',['extract-features',str(prepared),'--project-dir',str(stage),'--num-workers','0','--yes']),
                                  ('index',['create-index','--project-dir',str(stage),'--overwrite'])]:
                started=time.perf_counter()
                with (folder/(phase+'.log')).open('w') as log:
                    proc=subprocess.run([sys.executable,'-u','-m','wise',*command],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'HF_HUB_OFFLINE':'1'},timeout=240)
                row[phase+'_seconds']=time.perf_counter()-started
                if proc.returncode:raise RuntimeError(f'{phase} failed; inspect {folder}')
            with sqlite3.connect(stage/'metadata/internal.db') as db:
                mid=db.execute('SELECT m.id FROM media m JOIN source_collections s ON m.source_collection_id=s.id WHERE s.location=?',(str(prepared),)).fetchone()[0]
                counts=dict(db.execute('SELECT feature_extractor_id,count(*) FROM vectors WHERE media_id=? GROUP BY feature_extractor_id',(mid,)))
                assert counts and all('insightface' in k for k in counts),counts
                row['new_video_vector_counts']=counts
            report['runs'].append(row);save();print(json.dumps(row,ensure_ascii=False),flush=True)
        finally:
            shutil.rmtree(stage)
    report['status']='completed';report['production_unchanged']=fingerprint()==before;save()


if __name__=='__main__':main()
