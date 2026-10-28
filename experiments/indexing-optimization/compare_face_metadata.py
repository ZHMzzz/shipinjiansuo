#!/usr/bin/env python3
"""Compare existing face metadata using SQLite only; no models or FAISS imports."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


HERE = Path(__file__).resolve().parent
FACE = 'deepinsight/insightface/buffalo_l/_unknown'
SQL = '''SELECT v.id,m.path,v.timestamp,f.bbox_x,f.bbox_y,f.bbox_w,f.bbox_h,
                f.age,f.is_male,f.detection_score
         FROM vectors v JOIN media m ON m.id=v.media_id
         JOIN vector_metadata_insightface f ON f.vector_id=v.id
         WHERE v.feature_extractor_id=? AND v.modality=? ORDER BY v.id'''


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(project):
    path = (project / 'metadata/internal.db').resolve()
    wal = Path(str(path) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f'Require a completed snapshot without a nonempty WAL: {path}')
    before = sha256(path)
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        names = list(db.execute('SELECT path FROM media'))
        if len({Path(row[0]).name for row in names}) != len(names):
            raise ValueError('Media basenames must be unique within each project')
        rows = db.execute(SQL, (FACE, 'VIDEO')).fetchall()
    after = sha256(path)
    if before != after:
        raise ValueError(f'Database changed during read: {path}')
    records = [dict(identifier=row[0], key=(Path(row[1]).name, round(row[2], 4)),
                    bbox=row[3:7], age=row[7], is_male=row[8], score=row[9])
               for row in rows]
    return records, {'database': str(path), 'sha256': before,
                     'unchanged_during_read': True, 'nonempty_wal': False}


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = (max(0, min(ax + aw, bx + bw) - max(ax, bx))
                    * max(0, min(ay + ah, by + bh) - max(ay, by)))
    return intersection / max(aw * ah + bw * bh - intersection, 1e-12)


def compare(baseline, variant):
    left, left_source = load(baseline)
    right, right_source = load(variant)
    groups = defaultdict(list)
    for row in right:
        groups[row['key']].append(row)
    used, pairs = set(), []
    for row in left:
        candidates = [r for r in groups[row['key']] if r['identifier'] not in used]
        if not candidates:
            continue
        match = max(candidates, key=lambda r: iou(row['bbox'], r['bbox']))
        if iou(row['bbox'], match['bbox']) < .9:
            continue
        used.add(match['identifier'])
        pairs.append((row, match))
    age_deltas = [b['age'] - a['age'] for a, b in pairs]
    return {
        'schema': 1,
        'sources': {'baseline': left_source, 'variant': right_source},
        'method': {
            'access': 'SQLite mode=ro; existing VIDEO metadata only; no inference or index loading',
            'model': FACE, 'sql': SQL,
            'alignment': 'same unique media basename and timestamp rounded to 4 decimals; '
                         'greedy maximum bbox IoU >= 0.9; process vector IDs ascending',
            'bbox_format': 'x,y,width,height', 'age_delta': 'variant minus baseline',
            'script_sha256': sha256(Path(__file__).resolve()),
        },
        'baseline_faces': len(left), 'variant_faces': len(right),
        'aligned_faces': len(pairs),
        'baseline_unmatched': len(left) - len(pairs),
        'variant_unmatched': len(right) - len(pairs),
        'age_same_count': sum(d == 0 for d in age_deltas),
        'age_changed_count': sum(d != 0 for d in age_deltas),
        'age_delta_histogram': {str(d): n for d, n in sorted(Counter(age_deltas).items())},
        'age_max_absolute_change': max(map(abs, age_deltas), default=None),
        'gender_same_count': sum(a['is_male'] == b['is_male'] for a, b in pairs),
        'bbox_iou_min': min((iou(a['bbox'], b['bbox']) for a, b in pairs), default=None),
        'bbox_exact_count': sum(a['bbox'] == b['bbox'] for a, b in pairs),
        'detection_score_max_absolute_change': max(
            (abs(a['score'] - b['score']) for a, b in pairs), default=None),
        'limitations': [
            'Aggregated metadata consistency, not demographic or identity accuracy.',
            'Does not read registration names, photos, or embedding vectors.',
            'The full-batch +1 year result and the separate preview-image smoke -1 year '
            'result concern different inputs; do not combine their denominators or directions.',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=HERE / 'runs/baseline-r1/project')
    parser.add_argument('--variant', type=Path, default=HERE / 'runs/fast_fp16_coreml-r1/project')
    parser.add_argument('--output', type=Path, default=HERE / 'face-metadata-comparison.json')
    args = parser.parse_args()
    result = compare(args.baseline, args.variant)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('aligned_faces', 'age_delta_histogram',
                                            'gender_same_count', 'bbox_iou_min')}))
