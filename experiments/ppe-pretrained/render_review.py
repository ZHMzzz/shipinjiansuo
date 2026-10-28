"""Render raw fixed-threshold predictions; never changes labels or runs models."""
from pathlib import Path
import hashlib
import json
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
SELECTED=[
    'library21_0008',
    'hat_before_0004',
    'hat_transition_0002',
    'hat_after_0008',
    'library30_0013',
    'fixture_worker_barehead_0002',
    'library41_0008',
    'head_out_of_frame_0000',
]
EXTRA=['hat_after_0004']
COLORS={'person':(40,110,240),'helmet':(0,170,70),'no_helmet':(230,45,45)}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    frozen=json.loads((HERE/'labels-freeze.json').read_text())
    assert sha(HERE/'labels.json')==frozen['labels_sha256']
    assert sha(HERE/'dataset.json')==frozen['dataset_sha256']
    dataset=json.loads((HERE/'dataset.json').read_text())
    labels={r['frame_id']:r for r in json.loads((HERE/'labels.json').read_text())['frames']}
    frames={f['id']:f for c in dataset['cases'] for f in c['frames']}
    models=[]
    for name in ['yolo11-ppe','yolov8-ppe']:
        data=json.loads((HERE/f'{name}-mps-fp32.json').read_text())
        assert len(data['rounds'])==3 and 'all_phases_resources' in data,'Run is still incomplete'
        assert data['labels_sha256']==frozen['labels_sha256']
        models.append((name,data,{r['frame_id']:r for r in data['rounds'][0]['records']}))
    fontpath='/System/Library/Fonts/Supplemental/Arial.ttf'
    font=ImageFont.truetype(fontpath,21)
    small=ImageFont.truetype(fontpath,17)
    cellw=640
    def row_height(fid):
        f=frames[fid]
        return int(min(600, f['height']*min(1,624/f['width'])))+70
    whole=Image.new('RGB',(cellw*3,sum(row_height(fid) for fid in SELECTED)),(248,248,248))
    rowsdir=HERE/'review-rows';rowsdir.mkdir(exist_ok=True)
    selections=[]
    whole_y=0
    for ri,fid in enumerate(SELECTED+EXTRA):
        f=frames[fid];source=Image.open(f['path']).convert('RGB')
        cellh=row_height(fid)
        row=Image.new('RGB',(cellw*3,cellh),(248,248,248));rd=ImageDraw.Draw(row)
        for col in range(3):
            panel=source.copy();pd=ImageDraw.Draw(panel)
            head=f'{fid} | t={f["timestamp"]:.3f}s' if col==0 else models[col-1][0]
            subtitle=f'GT: {labels[fid]["presence"]}' if col==0 else 'conf >= 0.25 | blue=person green=helmet red=no-helmet'
            if col:
                name,data,records=models[col-1]
                mapping={int(i):k for k,ids in data['class_mapping'].items() for i in ids}
                for det in records[fid]['detections']:
                    group=mapping.get(det['class_id'])
                    if group is None:continue
                    box=det['xyxy'];color=COLORS[group]
                    pd.rectangle(box,outline=color,width=2)
                    txt=f'{group} {det["confidence"]:.2f}'
                    # Keep compact labels readable at native pixels before thumbnail placement.
                    label_font=ImageFont.truetype(fontpath,12)
                    x=max(0,min(box[0],panel.width-110));y=max(0,box[1]-15)
                    bb=pd.textbbox((x,y),txt,font=label_font)
                    pd.rectangle(bb,fill=color);pd.text((x,y),txt,font=label_font,fill='white')
            panel.thumbnail((cellw-16,cellh-70))
            x=col*cellw+(cellw-panel.width)//2;y=65
            row.paste(panel,(x,y));rd.text((col*cellw+8,8),head,font=font,fill='black')
            rd.text((col*cellw+8,36),subtitle,font=small,fill=(45,45,45))
            if col:rd.line((col*cellw,0,col*cellw,cellh),fill=(190,190,190),width=2)
        out=rowsdir/f'{fid}.jpg';row.save(out,quality=95)
        if fid in SELECTED:
            whole.paste(row,(0,whole_y))
            whole_y+=cellh
        selections.append({'frame_id':fid,'timestamp':f['timestamp'],'label':labels[fid]['presence'],'row_path':str(out),'models':{name:rec[fid]['detections'] for name,_,rec in models}})
    whole.save(HERE/'comparison.jpg',quality=95)
    (HERE/'visual-selections.json').write_text(json.dumps(selections,ensure_ascii=False,indent=2))
    print(json.dumps({'rows':len(SELECTED),'comparison':str(HERE/'comparison.jpg'),'labels_unchanged':sha(HERE/'labels.json')==frozen['labels_sha256']}))

if __name__=='__main__':main()
