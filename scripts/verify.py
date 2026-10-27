"""Exercise actual inference and API. Writes measured ranks; never changes ranking."""
import json
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:9670'
report = []

def record(label, response, elapsed, expected):
    response.raise_for_status()
    result = response.json()
    r = result.get('video_results') or {}
    seen = set()
    matches = []
    for hit in r.get('unmerged_windows', []):
        if hit['media_id'] in seen:
            continue
        seen.add(hit['media_id'])
        video = r['videos'][str(hit['media_id'])]
        matches.append({'video':video['filename'],'time':hit.get('thumbnail_ts',hit['ts']),
                        'score':hit['distance'],'bbox':hit.get('bbox'),'link':hit['link']})
    rank = next((i+1 for i,x in enumerate(matches) if expected in x['video']),None)
    row = {'test':label,'seconds':round(elapsed,3),'expected':expected,'rank':rank,'top5':matches[:5]}
    report.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)
    (ROOT/'logs/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return matches

def search(label, expected, text=None, image=None):
    start=time.perf_counter()
    if image:
        r=requests.post(BASE+'/demo/search',params={'search_in':'video',
            'feature_extractor_id':'transformers/owlv2/google/owlv2-base-patch16-ensemble',
            'end':100,'thumbnails_to_send':1,'add_prefix':'false'},
            files={'image_file_queries':(image,(ROOT/'data/queries'/image).read_bytes(),'image/jpeg')},timeout=240)
    else:
        r=requests.post(BASE+'/mvp-api/search',data={'query':text},timeout=240)
        r.raise_for_status()
        assert r.json()['score_kind']=='action'
    r.raise_for_status()
    return record(label,r,time.perf_counter()-start,expected)

info=requests.get(BASE+'/demo/info',timeout=240)
info.raise_for_status()
assert info.json()['num_thumbnails'] > 0, 'Missing thumbnails'
(ROOT/'logs/project-info.json').write_text(json.dumps(info.json(),ensure_ascii=False,indent=2))
for query,expected in [('一个人背着背包走路','backpack'),('一个人在跑步','jogging'),('锅里正在炒蔬菜','frying-vegetables')]:
    search('中文：'+query,expected,text=query)
search('独立背包图片','backpack',image='backpack-independent.jpg')
names=requests.get(BASE+'/mvp-api/people').json()
if '演示人物A' not in names:
    r=requests.post(BASE+'/mvp-api/people',data={'name':'演示人物A'},files={'photo':('face.jpg',(ROOT/'data/queries/coughing-6XvsLPDioVA_000000_000010-preview.jpg').read_bytes(),'image/jpeg')},timeout=120)
    r.raise_for_status()
start=time.perf_counter()
r=requests.post(BASE+'/mvp-api/people/演示人物A/search',timeout=120)
matches=record('姓名检索：演示人物A',r,time.perf_counter()-start,'coughing-6XvsLPDioVA')
assert requests.post(BASE+'/mvp-api/people/未登记的人物/search').status_code==404
# A bag must not silently register as a person.
r=requests.post(BASE+'/mvp-api/people',data={'name':'测试无脸图片'},files={'photo':('bag.jpg',(ROOT/'data/queries/backpack-independent.jpg').read_bytes(),'image/jpeg')},timeout=120)
assert r.status_code==400,(r.status_code,r.text)
if matches:
    url=requests.compat.urljoin(BASE+'/demo/',matches[0]['link'].split('#')[0])
    r=requests.get(url,headers={'Range':'bytes=0-1023'},timeout=10)
    assert r.status_code==206,('video range playback',r.status_code)
assert all(row['rank'] and row['rank']<=5 for row in report),'Some expected videos are outside Top-5; inspect report'
print('PASS: retrieval, enrollment validation, unknown name, video byte ranges',flush=True)
