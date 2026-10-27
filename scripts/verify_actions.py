"""Read-only live integration check; expects the documented demo fixtures."""
import json
import time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:9670'


def verify():
    library=requests.get(BASE+'/mvp-api/videos',timeout=30).json()
    status=requests.get(BASE+'/mvp-api/actions/status',timeout=30).json()
    assert status['status']=='ready',status
    assert status['indexed_videos']==library['count']
    assert status['indexed_segments']==sum(v['action_segments'] for v in library['videos'])
    queries=[('有人在仓库里突然失去平衡倒在地上','动作定位_串接测试24秒.mp4',16),
             ('一个人在室内测量墙面','动作定位_串接测试24秒.mp4',0),
             ('有人越过户外的金属围栏','围栏翻越_户外短片.mp4',0)]
    report={'status':status,'queries':[]}
    for query,expected,start in queries:
        begin=time.perf_counter()
        response=requests.post(BASE+'/mvp-api/search',data={'query':query,'limit':10},timeout=120)
        response.raise_for_status();result=response.json()
        first=result['video_results']['unmerged_windows'][0]
        name=result['video_results']['videos'][str(first['media_id'])]['filename']
        assert name==expected,(query,name,first)
        assert first['ts']==start,(query,first)
        for h in result['video_results']['unmerged_windows']:
            duration=result['video_results']['videos'][str(h['media_id'])]['duration']
            assert 0<=h['ts']<h['te']<=duration+.001
        assert requests.get(BASE+first['thumbnail'],timeout=30).headers['content-type'].startswith('image/')
        media=requests.get(BASE+first['link'].split('#')[0],headers={'Range':'bytes=0-127'},timeout=30)
        assert media.status_code==206 and len(media.content)==128
        report['queries'].append({'query':query,'elapsed_seconds':time.perf_counter()-begin,
            'encode_seconds':result['encode_seconds'],'top':first,'filename':name})
    for query,code in [('   ',400),('测'*1001,422)]:
        response=requests.post(BASE+'/mvp-api/search',data={'query':query},timeout=30)
        assert response.status_code==code
    response=requests.post(BASE+'/mvp-api/search',data={'query':'摔倒','limit':51},timeout=30)
    assert response.status_code==422
    path=ROOT/'logs/live-action-verification.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('PASS: complete index coverage, arbitrary Chinese, distinct time windows, playback range and input validation')


if __name__=='__main__': verify()
