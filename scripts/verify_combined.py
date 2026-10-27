"""Real endpoint regression for combination queries. Does not add/delete media."""
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:9670'


def query(text, image=False):
    data = dict(query=text)
    files = {'photo': ('bag.jpg', (ROOT/'data/queries/backpack-independent.jpg').read_bytes(), 'image/jpeg')} if image else None
    response = requests.post(BASE+'/mvp-api/search', data=data, files=files, timeout=120)
    response.raise_for_status()
    data = response.json()['video_results']
    seen, hits = set(), []
    for hit in data['unmerged_windows']:
        if hit['media_id'] not in seen:
            hits.append(dict(video=data['videos'][str(hit['media_id'])]['filename'], score=hit['distance'], time=hit['ts']))
            seen.add(hit['media_id'])
    return hits[:5]


if __name__ == '__main__':
    report = {'text': query('一个人背着背包走路')}
    for endpoint in ['/mvp-api/search', '/mvp-api/library/search']:
        response = requests.post(BASE+endpoint, data={'query':'在街道上'},
            files={'photo':('bag.jpg',(ROOT/'data/queries/backpack-independent.jpg').read_bytes(),'image/jpeg')},timeout=30)
        assert response.status_code == 400, response.text
    report['mixed_query'] = 'rejected at both public endpoints'
    response = requests.post(BASE+'/mvp-api/people/演示人物A/search',
                             data=dict(scene='坐在汽车里', threshold=.45), timeout=30)
    response.raise_for_status()
    data = response.json()
    hits = data['video_results']['unmerged_windows']
    assert data['score_kind'] == 'fusion' and data['scene_scope']=='containing_segment'
    assert hits and all(h['face_similarity'] >= .45 for h in hits)
    assert all(h['scene_ts']<=h['thumbnail_ts']<h['scene_te'] for h in hits)
    assert all('coughing-6XvsLPDioVA' in data['video_results']['videos'][h['media_id']]['filename'] or
               data['video_results']['videos'][h['media_id']]['filename']=='Qwen人脸入库回归.mp4' for h in hits)
    report['person_scene'] = dict(hits=len(hits), first={k:v for k,v in hits[0].items() if k!='thumbnail'})
    response = requests.post(BASE+'/mvp-api/people/演示人物A/search',
                             data=dict(scene='汽车', threshold=1), timeout=30)
    response.raise_for_status()
    assert not response.json()['video_results']
    report['strict_person_threshold'] = 'empty as expected'
    (ROOT/'logs/combined-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('PASS: mixed query rejected, person + scene and identity threshold')
