"""Read-only baseline collection for automatic semantic result filtering."""
import json, time, urllib.request, urllib.parse
from pathlib import Path
OUT=Path(__file__).resolve().parent
CASES=[
('calibration','戴红色安全帽的人坐着写字',[27]),
('calibration','锅里正在炒蔬菜',[5,6]),
('calibration','有人翻越围栏',[18]),
('calibration','戴安全帽的工人在吸烟',[19]),
('calibration','现场出现明火和黑烟',[20]),
('calibration','有人在仓库里突然失去平衡倒在地上',[17]),
('calibration','一个人背着背包在街道上走路',[1]),
('calibration','有人在雪地上滑倒',[26]),
('calibration','潜水员在海底珊瑚礁旁游泳',[]),
('calibration','宇航员在太空舱外维修卫星',[]),
('calibration','厨师在厨房揉面制作披萨',[]),
('calibration','一只大象在草原上喝水',[]),
('holdout','工人坐在桌前低头书写',[27]),
('holdout','有人在走廊里抽香烟',[25]),
('holdout','挖掘机旁边有人跌倒',[21]),
('holdout','有人在户外慢跑',[]),
('holdout','医生在手术室给病人做手术',[]),
('holdout','企鹅在冰山上跳入海水',[]),
('holdout','戴红色安全帽的人骑自行车',[]),
]
if (OUT/'baseline.json').exists():
    raise SystemExit('Baseline already exists; do not overwrite pre-filter evidence.')
rows=[]
for split,q,targets in CASES:
    req=urllib.request.Request('http://127.0.0.1:9670/mvp-api/library/search',data=urllib.parse.urlencode({'query':q,'limit':100}).encode())
    t=time.perf_counter()
    data=json.load(urllib.request.urlopen(req,timeout=180))
    rows.append(dict(split=split,query=q,known_positive_media=targets,expected_empty=not targets,seconds=time.perf_counter()-t,items=data['items']))
    (OUT/'baseline.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print(q, 'top:',[(i['id'],round(i['score'],3)) for i in data['items'][:4]],'known:',[(i['id'],round(i['score'],3)) for i in data['items'] if i['id'] in targets],flush=True)
