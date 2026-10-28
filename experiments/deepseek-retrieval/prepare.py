"""Freeze retrieval comparison before seeing new parser/retrieval outputs."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def main():
    output = HERE / 'design.json'
    if output.exists():
        raise SystemExit('Preserve frozen design')
    old = json.loads((ROOT / 'experiments/compound-filter/design.json').read_text())
    parser = json.loads((ROOT / 'experiments/text-query-parser/design.json').read_text())
    cases = [dict(c, cohort='original26') for c in old['cases']]
    by_id = {c['id']: c for c in cases}
    extra = [
        ('paragraph_write', 'write', '帮我找一个人戴着红色安全帽坐着写字的片段。只需要这些画面，不限定地点。'),
        ('paragraph_indoor_run', 'indoor_run', '我想找有人在室内慢跑的视频。衣服颜色不限，具体是哪一个人也没有要求。'),
        ('paragraph_smoke', 'smoke', '请找一个工人戴着安全帽吸烟的画面。戴帽子和吸烟必须是同一个人，不需要限定帽子的颜色。'),
        ('paragraph_bag', 'outdoor_bag', '想找户外有人背着背包行走的片段。背包颜色和衣服颜色都不限，不用限定具体人物。'),
        ('paragraph_fire', 'warehouse_fire', '我要找仓库里的火情画面，货架旁边起火并冒出黑烟。不要限制具体是哪一家仓库。'),
        ('paragraph_helmet_bike', 'helmet_bike', '帮我找有人戴着红色安全帽骑自行车的视频。要求骑车的人本人戴着这顶帽子，地点没有限制。'),
    ]
    for cid, source, query in extra:
        c = dict(by_id[source], id=cid, query=query, cohort='paragraph6', source_case=source, split='paraphrase')
        cases.append(c)
    cases.extend([
        {'id': 'logic_or', 'cohort': 'logic3', 'query': '有人在室内慢跑，或者有人在雪地滑倒，找到其中任意一类都可以。', 'positive_media': [7,8,26], 'negative_media': [], 'conditions': [], 'split': 'logic'},
        {'id': 'logic_no_helmet', 'cohort': 'logic3', 'query': '有人未佩戴安全帽', 'positive_media': [17,27], 'negative_media': [], 'conditions': [], 'split': 'logic', 'label_note': '既有视觉核验：17在16–24秒可见裸露头部；27摘帽后存在未戴帽状态。仅视频级标注。'},
        {'id': 'logic_order', 'cohort': 'logic3', 'query': '同一个工人先摘下安全帽，再坐着写字。', 'positive_media': [], 'negative_media': [], 'conditions': [], 'split': 'logic', 'label_note': '不预设库内存在满足顺序的正例；仅检查不支持条件的处理，不计正确率。'},
    ])
    d = {'cases': cases, 'system_prompt': parser['system_prompt'], 'warmup': parser['warmup'], 'repeats': [],
         'rule': {'baseline_min_best': .52, 'baseline_min_result': .45, 'baseline_gap': .08, 'condition_min': .35, 'condition_gap': .15},
         'frozen_plan': '主方案仅用 DeepSeek 自动 must 条件，固定规则替换人物A/B为有人、工人A/B为工人后编码；原样条件作为预声明敏感性对照。单组同片段AND、多组OR；先整句基线门槛，再条件过滤，再每视频保留最高原句分片段。无参数重选。',
         'unsupported': 'must_not、uncertain、负向must、先后顺序、显式人数/多人或不同人物关系：整条查询退回原基线并报告未验证；prefer不作硬门槛。同人属性采用同片段共现近似，不能验证同人身份。',
         'labels': '复用之前复核的查询—视频级部分标注，段落继承等义短句标注。不是穷尽标注；未知候选不计正确或误报。新长句非独立事件样本。'}
    output.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')
    print('Frozen', len(cases), 'queries')

if __name__ == '__main__':
    main()
