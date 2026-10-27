"""Check the real uploaded fixture, hot-loaded indexes, playback and failure isolation."""
import json
import time
from pathlib import Path

import requests

ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:9670'
report={}
job=json.loads((ROOT/'logs/upload-test-job.json').read_text())['job']
current=next(j for j in requests.get(BASE+'/mvp-api/jobs').json()['jobs'] if j['id']==job['id'])
assert current['status']=='completed',current
mid=str(current['media_id'])
videos=requests.get(BASE+'/mvp-api/videos').json()['videos']
video=next(v for v in videos if str(v['id'])==mid)
assert video['action_segments']>0
assert any('owlv2' in model for model in video['features'])
assert any('insightface' in model for model in video['features'])
report['video']=video
report['processing_seconds']=round(current['updated']-current['created'],2)
image=(ROOT/'logs/test-fixtures/new-video-query.jpg').read_bytes()
for kind in ['semantic','image','face']:
    if kind in ['face','image']:
        r=requests.post(BASE+'/demo/search',params={'search_in':'video',
            'feature_extractor_id':('deepinsight/insightface/buffalo_l/_unknown' if kind=='face' else 'transformers/owlv2/google/owlv2-base-patch16-ensemble'),
            'end':100,'thumbnails_to_send':1,'add_prefix':'false'},
            files={'image_file_queries':('query.jpg',image,'image/jpeg')},timeout=120)
    else:
        r=requests.post(BASE+'/mvp-api/search',
            data={'query':'穿橙色衣服的小孩坐在地毯上'} if kind=='semantic' else {},
            files={'photo':('query.jpg',image,'image/jpeg')} if kind=='image' else None,timeout=120)
    r.raise_for_status()
    hits=r.json()['video_results']['unmerged_windows']
    assert mid in [str(h['media_id']) for h in hits],(kind,'uploaded video missing')
    report[kind]={'top_media_id':hits[0]['media_id'],'uploaded_hit_rank':next(i+1 for i,h in enumerate(hits) if str(h['media_id'])==mid)}
r=requests.post(BASE+'/mvp-api/trace',files={'photo':('query.jpg',image,'image/jpeg')},timeout=30)
r.raise_for_status()
top=r.json()['video_results']['unmerged_windows'][0]
assert str(top['media_id'])==mid,top
report['trace']={'media_id':top['media_id'],'time':top['thumbnail_ts']}
for endpoint in [video['url'],video['thumbnail']]:
    r=requests.get(BASE+endpoint,headers={'Range':'bytes=0-1023'},timeout=10)
    assert r.status_code in (200,206)
    assert r.content
report['playback']='passed'
# Identical bytes must reuse the same task, including when the filename changes.
original=(ROOT/'logs/test-fixtures/新增人物检索验证.mp4').read_bytes()
r=requests.post(BASE+'/mvp-api/uploads',files={'video':('重复上传验证.mp4',original,'video/mp4')},timeout=10)
r.raise_for_status()
assert r.json()['duplicate'] and r.json()['job']['id']==job['id']
count_before=len(videos)
# Invalid .mp4 must visibly fail and never publish a new project version.
pointer=(ROOT/'projects/demo').resolve()
r=requests.post(BASE+'/mvp-api/uploads',files={'video':('无效文件测试.mp4',b'not an actual video: upload failure test','video/mp4')},timeout=10)
r.raise_for_status()
bad=r.json()['job']['id']
for _ in range(20):
    state=next(j for j in requests.get(BASE+'/mvp-api/jobs').json()['jobs'] if j['id']==bad)
    if state['status']=='failed':break
    time.sleep(.5)
assert state['status']=='failed',state
assert (ROOT/'projects/demo').resolve()==pointer
assert requests.get(BASE+'/mvp-api/videos').json()['count']==count_before
report['invalid_upload']={'status':state['status'],'message':state['message'],'library_unchanged':True}
report['duplicate']='same task, no extra video'
(ROOT/'logs/upload-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
