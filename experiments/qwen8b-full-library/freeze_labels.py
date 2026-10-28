"""Freeze pre-ranking, visually reviewed relevance judgements for all 68 x 129 pairs."""
from pathlib import Path
import csv
import hashlib
import json
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def save(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))

def main():
    assert not (HERE/'2b-fp16.json').exists() and not (HERE/'8b-q8.json').exists(), 'Freeze before running either model'
    data=json.loads((HERE/'dataset-unlabeled.json').read_text())
    draft=json.loads((ROOT/'.scratch/qwen8b-full-library/queries-draft.json').read_text())
    clips=data['clips']; queries=draft['queries']; labels={q['id']:{c['id']:0 for c in clips} for q in queries}
    evidence={q['id']:[] for q in queries}
    def mark(qid,mid,starts=None,grade=2,why='画面满足查询，见对应视频的时间标注与预览'):
        selected=[c for c in clips if c['media_id']==mid and (starts is None or c['ts'] in starts)]
        assert selected,(qid,mid,starts)
        for c in selected: labels[qid][c['id']]=grade
        evidence[qid].append({'media_id':mid,'segments':[c['id'] for c in selected],'grade':grade,'reason':why})
    simple={'B01':[1],'B03':[3,34],'B04':[4],'B05':[5],'B06':[6],'B07':[7],'B08':[8],
        'B09':[9],'B10':[10],'B11':[11],'B12':[12],'B13':[13],'B14':[14,39],'B15':[15,16],
        'B16':[19,20],'B19':[18],'B20':[19],'B21':[20],'B22':[21],'B25':[24],'B26':[25],
        'B27':[26],'B29':[28],'B30':[29],'B31':[30],'B33':[32],'B34':[33],'B36':[36],
        'B37':[37],'B38':[38],'B39':[40],'H01':[27],'H04':[19],'H05':[21],'H07':[28],
        'H08':[29],'H09':[37],'H12':[20],'H13':[6],'H14':[36],'H15':[40],'H16':[15,16]}
    for qid,mids in simple.items():
        for mid in mids: mark(qid,mid)
    for qid in ['B02','B17','H10']:
        mark(qid,2);mark(qid,17,[4,8,12],why='8–16秒同源背包行走；0–8窗口不含该动作')
    mark('B18',17,[12,16],why='17.75–19秒失衡、倒地过程，两窗口均包含')
    mark('B23',22,[0,4,8,12,16,20],why='白衣人物桌前讲话截至约22.5秒，随后为图片和片尾')
    mark('B24',23,list(range(4,48,4)),why='窗口内有明亮练习室多人舞蹈，包含穿插的排练镜头')
    mark('B24',23,[0,48],grade=1,why='仅有独舞/录音室动作，没有同时出现多人群舞')
    for qid in ['B28','H02']:
        mark(qid,27,[0,4,8],why='坐者戴红帽低头在膝上垫纸写字，站者蓝工服')
        mark(qid,27,[12],grade=1,why='剪裁近景只见站者和坐者帽顶，未见写字动作')
    mark('B32',31,[0,4,28,32,36,40,44],why='0–7、34–48秒舞台舞蹈；其他时段含后台、持麦和拥抱')
    mark('B32',31,[8,12,16,20,24,48],grade=1,why='表演/后台/相拥，同一人物但没有清楚街舞过程')
    mark('B35',35,[0,4,8],why='0–10.5秒人物转身姿态；之后变为封面静态图')
    mark('B35',35,[12],grade=1,why='仅静态黑裙封面和片尾，没有转身')
    mark('B38',35,[8,12],grade=1,why='同样有杂志封面，但只有单张封面，不是多张图片轮换')
    mark('B40',40,[20,24,28],why='约27秒起目标圆环入镜，石头滑入；后段有石头近景')
    mark('B40',40,[0,4,8,12,16],grade=1,why='可见冰壶运动/滑石，但尚未看到滑向目标区的完整证据')
    mark('H03',27,[0,4],why='约5.0–5.25秒从戴帽转为脱帽；16帧输入含动作前后')
    mark('H03',27,[8,12],grade=1,why='已脱帽状态可见，但不含摘帽动作过程')
    mark('H06',26,[4,8],why='约8.5–9秒由躺卧撑地坐起，试图起身')
    mark('H06',26,[0],grade=1,why='窗口截至7.96秒，含跌倒和躺卧，缺少后续坐起过程')
    mark('H11',23,list(range(12,48,4)),why='群体舞蹈出现在明亮练习空间')
    mark('H11',23,[0,4,8,48],grade=1,why='只有独舞或双人主画面，不满足很多人一起跳舞')
    pairs={'P01':'B33','P02':'B17','P03':'B21','P04':'B30','P05':'B37','P06':'B39','P07':'B27','P08':'H01'}
    for qid,base in pairs.items():
        labels[qid]=dict(labels[base]);evidence[qid]=[dict(x) for x in evidence[base]]
    edits={
        'B28':('蓝色工服的人员在门岗旁，一人坐着写字、一人站着','实际在膝上垫纸写字，没有可确认桌面'),
        'H01':('站在门岗旁的蓝色工服人员头上没有戴安全帽','地点改为可见的门岗，不再假定桌子'),
        'H02':('戴红色安全帽的蓝色工服人员坐着写字','删除不成立的桌前条件'),
        'P08':('穿蓝工服站在门岗边的那个人没戴安全帽','与H01一致修正位置描述'),
        'P01':('找那段下雨天大家打伞在街上走的视频','原片支持沿街行走，不能确认横穿马路'),
        'N01':('有人骑自行车从室内楼梯上往下骑','原题路口背景存在类型不清的两轮车；按方案预先换成全库未见的明确动作，避免把不确定骑行者标为负例')}
    changes=[]
    for q in queries:
        original=q['query'];text=edits.get(q['id'],(original,''))[0]
        if text!=original: changes.append({'id':q['id'],'original':original,'revised':text,'reason':edits[q['id']][1]})
        row={'id':q['id'],'text':text,'original_text':original,'group':q['group'],
            'language':'zh','expected':[cid for cid,g in labels[q['id']].items() if g==2],
            'relevance':labels[q['id']], 'evidence':evidence[q['id']],
            'answer_status':'no_visible_match' if q['id'].startswith('N') else 'answerable',
            'paired_with':pairs.get(q['id'])}
        if not q['id'].startswith('N'): assert row['expected'],q['id']
        data['queries'].append(row)
    assert len(data['queries'])==68 and len(clips)==129
    data['label_protocol']={'reviewer':'Codex, single visual annotator; no independent human adjudication',
        'blind_to_rankings':True,'grade_2':'All query constraints visibly supported in a subinterval of the candidate window; process queries require before/after evidence',
        'grade_1':'Related content with incomplete action/state evidence; not a strict hit',
        'grade_0':'Different subject/action/place or no visual evidence. Shared words/background alone are insufficient.',
        'binary_hit_threshold':2,'no_answer_note':'No supported visible match in reviewed corpus. This is not proof of absence in every original video frame.',
        'coverage':'All 40 source storyboards across all 129 windows; dense and actual-input checks for ambiguous transitions',
        'main_queries':'56 B/H; 8 P separately; 4 N diagnostic only',
        'source_groups':{'3/34':'same original','14/39':'same original','15/16':'same original','2/17-backpack':'same original; query-specific grouping of mixed windows'},
        'frozen_at_unix':time.time()}
    save('query-amendments.json',changes);save('dataset.json',data)
    with (HERE/'labels.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['query_id','query','segment_id','media_id','start','end','relevance'])
        for q in data['queries']:
            for c in clips:w.writerow([q['id'],q['text'],c['id'],c['media_id'],c['ts'],c['te'],q['relevance'][c['id']]])
    save('labels-freeze.json',{'created_at':time.time(),'query_count':68,'segment_count':129,'pair_count':68*129,
        'files':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['dataset.json','labels.csv','query-amendments.json','visual-review-notes.json']}})
    print('Frozen 68 queries / 129 segments / 8772 judgements; 56 main + 8 paraphrases + 4 no-answer diagnostics')

if __name__=='__main__': main()
