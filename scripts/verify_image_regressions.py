"""Replay the three user-supplied regression images against the live application.

Usage: python scripts/verify_image_regressions.py SCREENSHOT EXCAVATOR SUNGLASSES
Inputs stay outside the repository; only a local ignored JSON report is written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import requests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('screenshot','excavator','sunglasses'): parser.add_argument(name,type=Path)
    args=parser.parse_args();base='http://127.0.0.1:9670';report={}
    for kind,expected in [('screenshot',{'抖音20261026-905866.mp4'}),
                          ('excavator',{'摔倒01_白天工地.mp4'}),
                          ('sunglasses',{'whistling-OdLV8MmYkKI_000001_000011.mp4','whistling-LqQYSUb_TJI_000052_000062.mp4'})]:
        path=getattr(args,kind);raw=path.read_bytes();started=time.perf_counter()
        endpoint='/mvp-api/trace' if kind=='screenshot' else '/mvp-api/search'
        response=requests.post(base+endpoint,data={} if kind=='screenshot' else {'refine':'true'},
            files={'photo':(path.name,raw,'application/octet-stream')},timeout=180)
        response.raise_for_status();result=response.json();videos=result['video_results'];hits=videos['unmerged_windows']
        ranking=[dict(media_id=h['media_id'],name=videos['videos'][str(h['media_id'])]['filename'],
            score=h['distance'],ts=h['ts'],te=h['te'],method=h.get('trace_method'),original_rank=h.get('original_rank')) for h in hits]
        report[kind]={'sha256':hashlib.sha256(raw).hexdigest(),'seconds':time.perf_counter()-started,
            'ranking':ranking,'refinement':result.get('object_refinement')}
        output=Path(__file__).resolve().parents[1]/'logs/user-image-regression/live-after.json'
        output.parent.mkdir(exist_ok=True,parents=True);output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        assert ranking[0]['name'] in expected,(kind,ranking[:5])
        if kind=='screenshot': assert ranking[0]['method']=='aligned_crop' and ranking[0]['score']>.8
        else: assert result['object_refinement']['applied'] and all(not h.get('bbox') for h in hits)
        print(kind,'PASS',ranking[0],'seconds',round(report[kind]['seconds'],2),flush=True)


if __name__=='__main__': main()
