"""Replay the two object regression images against the live application.

Usage: python scripts/verify_image_regressions.py EXCAVATOR SUNGLASSES
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
    for name in ('excavator','sunglasses'): parser.add_argument(name,type=Path)
    args=parser.parse_args();base='http://127.0.0.1:9670';report={}
    for kind,expected in [('excavator',{'摔倒01_白天工地.mp4'}),
                          ('sunglasses',{'whistling-OdLV8MmYkKI_000001_000011.mp4','whistling-LqQYSUb_TJI_000052_000062.mp4'})]:
        path=getattr(args,kind);raw=path.read_bytes();started=time.perf_counter()
        endpoint='/demo/search'
        params={'search_in':'video','feature_extractor_id':'transformers/owlv2/google/owlv2-base-patch16-ensemble','end':100,'thumbnails_to_send':1,'add_prefix':'false'}
        response=requests.post(base+endpoint,params=params,
            files={'image_file_queries':(path.name,raw,'application/octet-stream')},timeout=180)
        response.raise_for_status();result=response.json();videos=result['video_results'];hits=videos['unmerged_windows']
        ranking=[dict(media_id=h['media_id'],name=videos['videos'][str(h['media_id'])]['filename'],
            score=h['distance'],ts=h['ts'],te=h['te'],original_rank=h.get('original_rank')) for h in hits]
        report[kind]={'sha256':hashlib.sha256(raw).hexdigest(),'seconds':time.perf_counter()-started,
            'ranking':ranking,'refinement':result.get('object_refinement')}
        output=Path(__file__).resolve().parents[1]/'logs/owl-restored-user-images.json'
        output.parent.mkdir(exist_ok=True,parents=True);output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        assert ranking[0]['name'] in expected,(kind,ranking[:5])
        assert any(h.get('bbox') for h in hits) and not result.get('object_refinement')
        print(kind,'PASS',ranking[0],'seconds',round(report[kind]['seconds'],2),flush=True)


if __name__=='__main__': main()
