"""Read-only diagnostic: compound results are not a subset of action results.

An assertion failure is the observed symptom, not a claimed code regression.
No server/index mutations. Retain here as a diagnostic artifact.
"""
import json
from pathlib import Path
import httpx

HERE = Path(__file__).resolve().parent

def main():
    queries = ['红色安全帽骑自行车', '红色安全帽写字', '骑自行车']
    runs = []
    with httpx.Client(timeout=45) as client:
        for _ in range(2):
            rows = []
            for query in queries:
                response = client.post('http://127.0.0.1:9670/mvp-api/library/search', data={'query':query, 'limit':'100'})
                response.raise_for_status()
                rows.append({'query':query, 'items':[{k:i.get(k) for k in ['id','name','score','start','end']} for i in response.json()['items']]})
            runs.append(rows)
    (HERE/'reproduction.json').write_text(json.dumps(runs,ensure_ascii=False,indent=2)+'\n')
    assert runs[0] == runs[1], 'Results changed across repeated requests'
    ids = [{i['id'] for i in row['items']} for row in runs[0]]
    print('Repeated twice:', dict(zip(queries,[sorted(v) for v in ids])), flush=True)
    assert ids[0] <= ids[2], f'Compound-only candidates absent from action query: {sorted(ids[0]-ids[2])}'

if __name__ == '__main__':
    main()
