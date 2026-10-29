"""Explain full-gallery ranking changes and same-frame box consistency."""
import json
from pathlib import Path
import sqlite3

import bench
from evaluate import iou
from owl_sdpa_full import OUT


def main():
    report = json.loads((OUT/'retrieval.json').read_text())
    manifest = json.loads((OUT/'manifest.json').read_text())
    canonical = {Path(m['path']).name: m['id'] for m in manifest['media']}
    with sqlite3.connect((OUT/'sdpa/project/metadata/internal.db').as_uri()+'?mode=ro', uri=True) as db:
        local_ids = {canonical[Path(path).name]: mid for mid, path in db.execute('select id,path from media')}
        rows = []
        for query in report['queries']:
            a, b = query['baseline']['top_videos'], query['variant']['top_videos']
            row = {'query_id': query['query_id'], 'origin': query['origin'],
                   'baseline_video_top5': [x['media_id'] for x in a[:5]],
                   'sdpa_video_top5': [x['media_id'] for x in b[:5]],
                   'top1_video_same': query['top1_video_same'],
                   'baseline_known_class': query['baseline']['video_known_class'],
                   'sdpa_known_class': query['variant']['video_known_class'],
                   'baseline_source': query['baseline']['video_source'],
                   'sdpa_source': query['variant']['video_source'],
                   'matches': []}
            other = {x['media_id']: x for x in b}
            for old in a[:5]:
                new = other.get(old['media_id'])
                boxes = db.execute('SELECT b.bbox_x,b.bbox_y,b.bbox_w,b.bbox_h FROM vectors v '
                    'JOIN vector_metadata_owlv2 b ON b.vector_id=v.id '
                    'WHERE v.media_id=? AND abs(v.timestamp-?)<0.00001', (local_ids[old['media_id']], old['timestamp'])).fetchall()
                closest = max(boxes, key=lambda box: iou(old['bbox'], box)) if boxes else None
                row['matches'].append({'media_id': old['media_id'], 'filename': old['filename'],
                    'baseline_time': old['timestamp'], 'sdpa_best_time': new['timestamp'] if new else None,
                    'best_time_shift_seconds': new['timestamp']-old['timestamp'] if new else None,
                    'baseline_score': old['logit'], 'sdpa_best_score': new['logit'] if new else None,
                    'baseline_box': old['bbox'], 'sdpa_box_at_baseline_time': closest,
                    'same_frame_best_iou': iou(old['bbox'], closest) if closest else None,
                    'note': 'Best geometric match in same frame, not a human object-identity label; time shift compares best query hit in this video.'})
            rows.append(row)
    result = {'scope': 'All 44 re-extracted video galleries; unchanged frozen 9 reference-image queries',
              'changed_top5_video_order': [r['query_id'] for r in rows if r['baseline_video_top5'] != r['sdpa_video_top5']],
              'changed_top5_video_membership': [r['query_id'] for r in rows if set(r['baseline_video_top5']) != set(r['sdpa_video_top5'])],
              'changed_top1_video': [r['query_id'] for r in rows if not r['top1_video_same']],
              'queries': rows, 'faces': report['faces'], 'summary': report['summary']}
    bench.dump(OUT/'review.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('queries','faces','summary')}, ensure_ascii=False))
    print(json.dumps(report['summary'], ensure_ascii=False))


if __name__ == '__main__': main()
