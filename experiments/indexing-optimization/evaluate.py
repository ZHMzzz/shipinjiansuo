"""Frozen-query CPU evaluation of real WISE ingestion variants.

prepare: freeze existing query images, labels, and media identity (no inference).
freeze: encode images ONCE through the unmodified WISE EmbeddingService.
compare: compare actual FAISS rankings, optionally replacing sampled media in a
         fixed production background. No query/gallery normalization is added.

Only ``freeze`` loads a model. ``prepare`` and ``compare`` are CPU-only.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OWL = 'transformers/owlv2/google/owlv2-base-patch16-ensemble'
FACE = 'deepinsight/insightface/buffalo_l/_unknown'
DEFAULT_MEDIA = [1, 2, 3, 5, 9, 13, 14, 19, 21, 40, 41]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def readonly_db(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def prepare(project, output, media_ids=DEFAULT_MEDIA):
    project = Path(project).resolve()
    selected = set(media_ids)
    with readonly_db(project / 'metadata/internal.db') as db:
        media = [dict(zip(('id', 'path', 'duration', 'location'), row))
                 for row in db.execute('SELECT m.id,m.path,m.duration,s.location FROM media m '
                                       'JOIN source_collections s ON s.id=m.source_collection_id')
                 if row[0] in selected]
    if {m['id'] for m in media} != selected:
        raise ValueError('Requested media IDs are missing')
    if len({Path(m['path']).name for m in media}) != len(media):
        raise ValueError('Ambiguous media basenames')
    for item in media:
        source = Path(item['location']) / item['path']
        if not source.is_absolute():
            source = ROOT / source
        item['sha256'] = sha(source)
    design_path = ROOT / 'experiments/owl-qwen-comparison/design.json'
    design = json.loads(design_path.read_text())
    queries = []
    for query in design['queries']:
        positives = set(query['known_class_media']) & selected
        sources = set(query['source_media']) & selected
        # Exclude helmet_large whose source clip is absent, and the toy query.
        # This leaves eight same-source crops and one independent image.
        if not positives or (query['source_media'] and not sources):
            continue
        queries.append({**query, 'image_sha256': sha(ROOT / query['image']),
                        'subset_source_media': sorted(sources),
                        'subset_known_class_media': sorted(positives),
                        'origin': 'same_source_crop' if query['source_media'] else 'independent_image'})
    result = {'schema': 1, 'model': OWL, 'production_project': str(project),
              'media': media, 'queries': queries, 'source_design_sha256': sha(design_path),
              'limitations': [
                  'Only independent_bag is an independent query image; other queries are same-source crops.',
                  'Known-class positives are inherited annotations, not exhaustive labels of the full background.',
                  'Region Top-k follows WISE raw FAISS ordering; video Top-k is a separate best-region-per-video metric.',
                  'This is regression evidence on a fixed small sample, not a general accuracy estimate.']}
    output = Path(output)
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise FileExistsError('Refusing to replace a different frozen query manifest')
    else:
        write_json(output, result)
    return result


def verify_queries(manifest):
    for query in manifest['queries']:
        if sha(ROOT / query['image']) != query['image_sha256']:
            raise ValueError(f"Query image changed: {query['id']}")


def norm_summary(norms):
    norms = np.asarray(norms, dtype='float64')
    return {'count': len(norms), 'norm_min': float(norms.min()) if len(norms) else None,
            'norm_median': float(np.median(norms)) if len(norms) else None,
            'norm_max': float(norms.max()) if len(norms) else None,
            'zero_count': int(np.count_nonzero(norms == 0))}


def query_integrity(vector, dimension, label):
    """Check the actual FP32 search input without changing its scale."""
    vector = np.asarray(vector, dtype='float32')
    if vector.shape != (1, dimension):
        raise ValueError(f'{label}: expected query shape (1, {dimension}), got {vector.shape}')
    if not np.isfinite(vector).all():
        raise ValueError(f'{label}: query contains non-finite values')
    stats = {'all_finite': True, 'dimension': dimension,
             **norm_summary(np.linalg.norm(vector.astype('float64'), axis=1))}
    if stats['zero_count']:
        raise ValueError(f'{label}: query has zero norm')
    return stats


def index_integrity(index, label, chunk_size=4096):
    """Scan every underlying flat-index vector once, with bounded temporary RAM.

    Reconstruction is by physical row of IndexFlatIP, not the external IDs held
    by IndexIDMap2. Float64 norm accumulation avoids FP32 square overflow. No
    reconstructed values are written back or normalized.
    """
    import faiss
    flat = faiss.downcast_index(index.index)
    if flat.ntotal != index.ntotal or flat.d != index.d:
        raise ValueError(f'{label}: wrapped and underlying FAISS shapes differ')
    norms = np.empty(flat.ntotal, dtype='float64')
    bad_rows = 0
    bad_values = 0
    for start in range(0, flat.ntotal, chunk_size):
        count = min(chunk_size, flat.ntotal - start)
        vectors = flat.reconstruct_n(start, count)
        finite = np.isfinite(vectors)
        bad_rows += int(np.count_nonzero(~finite.all(axis=1)))
        bad_values += int(np.count_nonzero(~finite))
        if finite.all():
            norms[start:start + count] = np.linalg.norm(vectors.astype('float64'), axis=1)
    if bad_rows:
        raise ValueError(f'{label}: FAISS contains {bad_rows} non-finite vectors '
                         f'({bad_values} non-finite values out of {flat.ntotal} vectors)')
    return {'all_finite': True, 'nonfinite_vector_count': 0, 'nonfinite_value_count': 0,
            'dimension': flat.d, 'scan': 'all underlying IndexFlatIP rows, once at gallery load',
            **norm_summary(norms)}


def freeze(manifest_path, output, device='mps', extractor=None, variant='baseline', processor_device='cpu'):
    """Root invokes this once, serially; this function performs model inference."""
    from owl_variants import install_variant
    variant_settings = install_variant(variant, processor_device=processor_device)
    from wise.api.common import MediaQueryTerm
    from wise.api.services.embedding._embedding import EmbeddingConfig, EmbeddingService
    from wise.feature.transformers_owlv2 import TransformersOWLv2FeatureExtractor as Extractor
    manifest = json.loads(Path(manifest_path).read_text())
    verify_queries(manifest)
    output = Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Frozen queries already exist; use a new experiment output path')
    if extractor is None:
        extractor = Extractor(OWL, device=device, warmup=False,
                              config=Extractor.Config(device=device, compile=False, warmup=False))
    service = EmbeddingService({OWL: extractor})
    config = EmbeddingConfig(query_prefix='')
    vectors = {}
    for query in manifest['queries']:
        term = MediaQueryTerm(term_id=query['id'], is_negative=False, qtype='visual',
                              src=(ROOT / query['image']).read_bytes())
        vector = service.embed(OWL, config, [term])
        vector = np.ascontiguousarray(vector, dtype='float32')
        if vector.shape[0] != 1 or not np.isfinite(vector).all():
            raise ValueError('Invalid query vector')
        vectors[query['id']] = vector
        print('FROZEN_QUERY', query['id'], list(vector.shape), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **vectors)
    write_json(output.with_suffix('.json'), {
        'manifest_sha256': sha(manifest_path), 'vectors_sha256': sha(output),
        'model': OWL, 'device': device, 'variant': variant_settings,
        'path': 'WISE EmbeddingService.embed, external-image hook and final weighted-average normalization',
        'query_ids': list(vectors),
        'source_sha256': {str(path.relative_to(ROOT)): sha(path) for path in (
            ROOT / 'wise/src/wise/api/services/embedding/_embedding.py',
            ROOT / 'wise/src/wise/feature/transformers_owlv2.py')}})


class Gallery:
    def __init__(self, project, canonical, model=OWL):
        import faiss
        self.project = Path(project).resolve()
        self.index_path = self.project / 'store' / model / 'index/video-IndexFlatIP.faiss'
        self.index = faiss.read_index(str(self.index_path))
        self.records = {}
        table = 'vector_metadata_owlv2' if model == OWL else 'vector_metadata_insightface'
        with readonly_db(self.project / 'metadata/internal.db') as db:
            names = dict(db.execute('SELECT id,path FROM media'))
            if len({Path(p).name for p in names.values()}) != len(names):
                raise ValueError('Ambiguous media basenames in project')
            self.media = {mid: canonical.get(Path(path).name, mid) for mid, path in names.items()}
            for row in db.execute(f'SELECT v.id,v.media_id,v.timestamp,b.bbox_x,b.bbox_y,b.bbox_w,b.bbox_h '
                                  f'FROM vectors v JOIN {table} b ON b.vector_id=v.id '
                                  'WHERE v.feature_extractor_id=? AND v.modality=?', (model, 'VIDEO')):
                vector_id, local_mid, timestamp, *bbox = row
                self.records[vector_id] = {
                    'vector_id': vector_id, 'local_media_id': local_mid,
                    'media_id': self.media[local_mid], 'filename': Path(names[local_mid]).name,
                    'timestamp': timestamp, 'bbox': bbox}
        if self.index.ntotal != len(self.records):
            raise ValueError('FAISS and metadata vector counts differ')
        index_ids = set(faiss.vector_to_array(self.index.id_map).tolist())
        if index_ids != set(self.records):
            raise ValueError('FAISS IDs and database metadata differ')
        self.numerical_integrity = index_integrity(self.index, str(self.index_path))

    def ranked(self, vector, exclude_names=()):
        vector = np.ascontiguousarray(vector, dtype='float32').reshape(1, -1)
        if vector.shape[1] != self.index.d:
            raise ValueError('Query/gallery dimensions differ')
        if not np.isfinite(vector).all():
            raise ValueError('Query contains non-finite values')
        scores, ids = self.index.search(vector, self.index.ntotal)
        return [{**self.records[int(i)], 'logit': float(score),
                 'score': float(1.0 / (1.0 + np.exp(-float(score))))}
                for score, i in zip(scores[0], ids[0])
                if i != -1 and self.records[int(i)]['filename'] not in exclude_names]


def top_videos(ranking, count=10):
    seen, result = set(), []
    for row in ranking:
        if row['media_id'] in seen:
            continue
        seen.add(row['media_id'])
        result.append(row)
        if len(result) == count:
            break
    return result


def metrics(ranking, positive_ids):
    if not positive_ids:
        return None
    rank = next((i + 1 for i, row in enumerate(ranking) if row['media_id'] in positive_ids), None)
    return {'first_positive_rank': rank, 'hit_at_1': rank == 1,
            'hit_at_5': rank is not None and rank <= 5}


def overlap(a, b, key, count=5):
    left, right = {key(x) for x in a[:count]}, {key(x) for x in b[:count]}
    return len(left & right) / max(1, min(len(left), len(right)))


def iou(left, right):
    ax, ay, aw, ah = left
    bx, by, bw, bh = right
    area = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return area / max(aw * ah + bw * bh - area, 1e-12)


def compare_faces(baseline, variant, manifest, people_db, threshold=.45):
    canonical = {Path(m['path']).name: m['id'] for m in manifest['media']}
    left, right = Gallery(baseline, canonical, FACE), Gallery(variant, canonical, FACE)
    # Match detections within the same media/time, then by maximal bbox overlap.
    right_groups = {}
    for identifier, row in right.records.items():
        key = row['media_id'], round(row['timestamp'], 4)
        right_groups.setdefault(key, []).append(identifier)
    pairs, used, cosines = [], set(), []
    undefined_cosines = 0
    for identifier, row in left.records.items():
        key = row['media_id'], round(row['timestamp'], 4)
        candidates = [x for x in right_groups.get(key, []) if x not in used]
        if not candidates:
            continue
        match = max(candidates, key=lambda x: iou(row['bbox'], right.records[x]['bbox']))
        if iou(row['bbox'], right.records[match]['bbox']) < .9:
            continue
        used.add(match)
        a, b = left.index.reconstruct(identifier), right.index.reconstruct(match)
        a64, b64 = a.astype('float64'), b.astype('float64')
        denominator = np.linalg.norm(a64) * np.linalg.norm(b64)
        if denominator:
            cosines.append(float(np.dot(a64, b64) / denominator))
        else:
            undefined_cosines += 1
        pairs.append((identifier, match))
    with readonly_db(people_db) as db:
        # Deliberately exclude names and photos from the query and output.
        registered = list(db.execute('SELECT vector,model FROM people ORDER BY rowid'))
    queries = []
    mapping = {b: a for a, b in pairs}
    for number, (raw, model) in enumerate(registered, 1):
        if model != FACE:
            continue
        query = np.frombuffer(raw, dtype='float32').reshape(1, -1)
        query_stats = query_integrity(query, left.index.d, f'registered query {number}')
        if right.index.d != left.index.d:
            raise ValueError('Baseline and variant face dimensions differ')
        ls, li = left.index.search(query, left.index.ntotal)
        rs, ri = right.index.search(query, right.index.ntotal)
        lp = {int(i) for score, i in zip(ls[0], li[0]) if i != -1 and score >= threshold}
        rp = {mapping.get(int(i), f'variant-only-{i}') for score, i in zip(rs[0], ri[0]) if i != -1 and score >= threshold}
        queries.append({'query_number': number, 'numerical_integrity': query_stats,
                        'baseline_candidates': len(lp),
                        'variant_candidates': len(rp), 'retained_candidates': len(lp & rp),
                        'removed_candidates': len(lp - rp), 'added_candidates': len(rp - lp),
                        'candidate_jaccard': len(lp & rp) / len(lp | rp) if lp | rp else 1.0,
                        'candidate_sets_identical': lp == rp})
    return {'threshold': threshold, 'alignment': 'same media/time; greedy bbox IoU >= 0.9',
            'gallery_numerical_integrity': {'baseline': left.numerical_integrity,
                                            'variant': right.numerical_integrity},
            'baseline_faces': left.index.ntotal, 'variant_faces': right.index.ntotal,
            'aligned_faces': len(pairs), 'baseline_unmatched': left.index.ntotal - len(pairs),
            'variant_unmatched': right.index.ntotal - len(pairs),
            'aligned_vector_cosine_min': min(cosines) if cosines else None,
            'aligned_vector_cosine_median': float(np.median(cosines)) if cosines else None,
            'aligned_vector_cosine_undefined_count': undefined_cosines,
            'registered_queries': queries,
            'limitation': 'Candidate consistency against existing registrations; not a new identity accuracy evaluation.'}


def compare(baseline, variant, manifest_path, vectors_path, output, background=None, people_db=None, variant_vectors=None):
    import faiss
    faiss.omp_set_num_threads(1)
    manifest = json.loads(Path(manifest_path).read_text())
    verify_queries(manifest)
    provenance = json.loads(Path(vectors_path).with_suffix('.json').read_text())
    if provenance['manifest_sha256'] != sha(manifest_path) or provenance['vectors_sha256'] != sha(vectors_path):
        raise ValueError('Frozen query provenance mismatch')
    canonical = {Path(m['path']).name: m['id'] for m in manifest['media']}
    left, right = Gallery(baseline, canonical), Gallery(variant, canonical)
    expected_names = set(canonical)
    for gallery in (left, right):
        if {r['filename'] for r in gallery.records.values()} != expected_names:
            raise ValueError('Evaluation requires exactly the frozen subset in both projects')
    bg = Gallery(background, canonical) if background else None
    vectors = np.load(vectors_path, allow_pickle=False)
    variant_vectors_path = Path(variant_vectors) if variant_vectors else Path(vectors_path)
    variant_provenance = json.loads(variant_vectors_path.with_suffix('.json').read_text())
    if (variant_provenance['manifest_sha256'] != sha(manifest_path)
            or variant_provenance['vectors_sha256'] != sha(variant_vectors_path)):
        raise ValueError('Variant query provenance mismatch')
    variant_queries = np.load(variant_vectors_path, allow_pickle=False)
    if right.index.d != left.index.d or (bg and bg.index.d != left.index.d):
        raise ValueError('Baseline, variant and background gallery dimensions differ')
    query_stats = {}
    for query in manifest['queries']:
        key = query['id']
        query_stats[key] = {
            'baseline': query_integrity(vectors[key], left.index.d, f'baseline query {key}'),
            'variant': query_integrity(variant_queries[key], right.index.d, f'variant query {key}')}
    results = []
    for query in manifest['queries']:
        vector = vectors[query['id']]
        variant_vector = variant_queries[query['id']]
        a, b = left.ranked(vector), right.ranked(variant_vector)
        modes = [('subset', a, b)]
        if bg:
            shared = bg.ranked(vector, exclude_names=expected_names)
            variant_shared = (bg.ranked(variant_vector, exclude_names=expected_names)
                              if variant_vectors else shared)
            # Stable merge of true FAISS scores. Gallery/query vectors are unchanged.
            modes.append(('production_background', sorted(shared + a, key=lambda x: -x['logit']),
                          sorted(variant_shared + b, key=lambda x: -x['logit'])))
        for scope, ar, br in modes:
            av, bv = top_videos(ar, count=None), top_videos(br, count=None)
            source = query['source_media'] if bg and scope == 'production_background' else query['subset_source_media']
            known = query['known_class_media'] if bg and scope == 'production_background' else query['subset_known_class_media']
            results.append({'query_id': query['id'], 'origin': query['origin'], 'scope': scope,
                            'baseline': {'region_source': metrics(ar, source), 'region_known_class': metrics(ar, known),
                                         'video_source': metrics(av, source), 'video_known_class': metrics(av, known),
                                         'top_regions': ar[:20], 'top_videos': av[:10]},
                            'variant': {'region_source': metrics(br, source), 'region_known_class': metrics(br, known),
                                        'video_source': metrics(bv, source), 'video_known_class': metrics(bv, known),
                                        'top_regions': br[:20], 'top_videos': bv[:10]},
                            'top1_video_same': av[0]['media_id'] == bv[0]['media_id'],
                            'top5_video_overlap': overlap(av, bv, lambda r: r['media_id']),
                            'top5_frame_overlap': overlap(ar, br, lambda r: (r['media_id'], round(r['timestamp'], 4)))})
    summary = []
    for scope in sorted({r['scope'] for r in results}):
        for origin in ('all', 'same_source_crop', 'independent_image'):
            rows = [r for r in results if r['scope'] == scope and (origin == 'all' or r['origin'] == origin)]
            if not rows:
                continue
            item = {'scope': scope, 'origin': origin, 'count': len(rows),
                    'top1_video_same_count': sum(r['top1_video_same'] for r in rows),
                    'mean_top5_video_overlap': float(np.mean([r['top5_video_overlap'] for r in rows]))}
            for side in ('baseline', 'variant'):
                item[side] = {}
                for metric in ('region_source', 'region_known_class', 'video_source', 'video_known_class'):
                    valid = [r[side][metric] for r in rows if r[side][metric] is not None]
                    item[side][metric] = {'count': len(valid),
                                          'top1': sum(r['hit_at_1'] for r in valid),
                                          'top5': sum(r['hit_at_5'] for r in valid)}
            summary.append(item)
    report = {'schema': 1, 'baseline_project': str(left.project), 'variant_project': str(right.project),
              'background_project': str(bg.project) if bg else None, 'query_manifest_sha256': sha(manifest_path),
              'query_vectors_sha256': sha(vectors_path),
              'variant_query_vectors_sha256': sha(variant_vectors_path),
              'query_mode': 'matched_variant_queries' if variant_vectors else 'frozen_baseline_queries',
              'background_mode': ('unchanged background gallery, variant query applied to all candidates'
                                  if background and variant_vectors else 'unchanged background gallery'),
              'baseline_vectors': left.index.ntotal,
              'variant_vectors': right.index.ntotal, 'background_vectors': bg.index.ntotal if bg else 0,
              'gallery_numerical_integrity': {'baseline': left.numerical_integrity,
                                              'variant': right.numerical_integrity,
                                              'background': bg.numerical_integrity if bg else None},
              'query_numerical_integrity': query_stats,
              'summary': summary, 'queries': results, 'limitations': manifest['limitations']}
    if people_db:
        report['faces'] = compare_faces(baseline, variant, manifest, people_db)
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--project', type=Path, default=ROOT / 'projects/demo')
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'queries.json')
    p.add_argument('--media-ids', type=int, nargs='+', default=DEFAULT_MEDIA)
    p = sub.add_parser('freeze')
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--variant', choices=('baseline', 'fast', 'fp16', 'fast_fp16', 'fp16_weights', 'fast_fp16_weights'), default='baseline')
    p.add_argument('--processor-device', choices=('cpu', 'mps'), default='cpu')
    p = sub.add_parser('compare')
    for name in ('baseline', 'variant', 'queries', 'vectors', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--background', type=Path)
    p.add_argument('--people-db', type=Path)
    p.add_argument('--variant-vectors', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.project, args.output, args.media_ids)
        print(json.dumps({'media': len(result['media']), 'queries': [q['id'] for q in result['queries']]}))
    elif args.command == 'freeze':
        freeze(args.queries, args.output, args.device, variant=args.variant, processor_device=args.processor_device)
    else:
        result = compare(args.baseline, args.variant, args.queries, args.vectors, args.output,
                         args.background, args.people_db, args.variant_vectors)
        print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
