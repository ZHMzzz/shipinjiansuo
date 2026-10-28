"""Isolated CPU/CoreML face smoke test and repeated-session cache benchmark.

Run --inspect first (no sessions/inference). Actual inference must be scheduled
by the parent benchmark so that it does not compete with OWLv2 or Qwen.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import statistics
import time

from face_variants import (DEFAULT_MODELS, EXPERIMENT, VARIANTS,
                           experiment_path, inspect_models, install_variant, sha256)


def face_records(faces):
    return [{'bbox': face.bbox.tolist(), 'kps': face.kps.tolist(),
             'embedding': face.embedding.tolist(), 'det_score': float(face.det_score),
             'age': int(face.age), 'gender': int(face.gender)} for face in faces]


def iou(left, right):
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    a = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    b = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / max(a + b - intersection, 1e-12)


def compare_faces(reference, candidate, gallery=None, threshold=.45):
    """Match detections one-to-one by IoU, then compare identity gate decisions."""
    import numpy as np
    if gallery is None:
        gallery = [face['embedding'] for item in reference for face in item['faces']]
    gallery = np.asarray(gallery, dtype='float64')
    if gallery.size:
        gallery = gallery.reshape(-1, 512)
        gallery /= np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    reference_by_path = {item['image']: item for item in reference}
    candidate_by_path = {item['image']: item for item in candidate}
    results, matched_total, gate_changes = [], 0, 0
    for path in sorted(set(reference_by_path) | set(candidate_by_path)):
        left = reference_by_path.get(path, {}).get('faces', [])
        right = candidate_by_path.get(path, {}).get('faces', [])
        pairs = sorted(((iou(a['bbox'], b['bbox']), i, j)
                        for i, a in enumerate(left) for j, b in enumerate(right)), reverse=True)
        used_a, used_b, matched = set(), set(), []
        for overlap, i, j in pairs:
            if overlap < .5 or i in used_a or j in used_b:
                continue
            used_a.add(i); used_b.add(j)
            a = np.asarray(left[i]['embedding'], dtype='float64')
            b = np.asarray(right[j]['embedding'], dtype='float64')
            a /= max(np.linalg.norm(a), 1e-12)
            b /= max(np.linalg.norm(b), 1e-12)
            changes = int(np.count_nonzero((gallery @ a >= threshold) !=
                                          (gallery @ b >= threshold))) if gallery.size else 0
            gate_changes += changes
            matched.append({'reference_face': i, 'candidate_face': j, 'bbox_iou': overlap,
                            'embedding_cosine': float(a @ b), 'gate_changes': changes,
                            'age_equal': left[i]['age'] == right[j]['age'],
                            'gender_equal': left[i]['gender'] == right[j]['gender']})
        matched_total += len(matched)
        results.append({'image': path, 'reference_faces': len(left), 'candidate_faces': len(right),
                        'unmatched_reference': len(left) - len(matched),
                        'unmatched_candidate': len(right) - len(matched), 'matches': matched})
    unmatched = sum(x['unmatched_reference'] + x['unmatched_candidate'] for x in results)
    cosines = [m['embedding_cosine'] for item in results for m in item['matches']]
    return {'images': results, 'matched_faces': matched_total, 'unmatched_faces': unmatched,
            'minimum_embedding_cosine': min(cosines) if cosines else None,
            'identity_gate': threshold, 'gallery_vectors': len(gallery),
            'gate_changes': gate_changes,
            'sample_gate_consistent': unmatched == 0 and matched_total > 0 and gate_changes == 0,
            'meaningful_face_comparison': matched_total > 0,
            'note': 'This checks only supplied samples/gallery; it is not a population accuracy claim.'}


def run_probe(args):
    import cv2
    import numpy as np
    import insightface
    import onnxruntime as ort
    sources_before = inspect_models(args.source_dir)
    if Path(args.source_dir).name != 'buffalo_l':
        raise ValueError('--source-dir must point to the existing buffalo_l directory')
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    gallery = np.load(args.gallery_npy, allow_pickle=False) if args.gallery_npy else None
    cache_dir = args.cache_dir or EXPERIMENT / 'cache/face'
    profile_dir = (EXPERIMENT / 'profiles/face') if args.profile else None
    images = []
    for name in args.images:
        path = Path(name).expanduser().resolve()
        pixels = cv2.imread(str(path))
        if pixels is None:
            raise ValueError(f'Cannot decode image: {path}')
        images.append((path, pixels))
    report = {'variant': args.variant, 'insightface_version': insightface.__version__,
              'onnxruntime_version': ort.__version__, 'sources_before': sources_before,
              'configuration': {'det_size': [640, 640], 'det_thresh': .5,
                                'allowed_modules': ['detection', 'recognition', 'genderage'],
                                'cache_dir': str(cache_dir), 'repeats': args.repeats,
                                'profiling': args.profile}, 'runs': []}
    for session_number in range(args.sessions):
        controller = install_variant(args.variant, source_dir=args.source_dir,
                                     cache_dir=cache_dir, profile_dir=profile_dir,
                                     quiet=not args.verbose, profile_compute_plan=args.compute_plan)
        app = None
        try:
            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                app = insightface.app.FaceAnalysis(name='buffalo_l',
                    root=str(Path(args.source_dir).expanduser().resolve().parent.parent),
                    allowed_modules=['detection', 'recognition', 'genderage'])
                app.prepare(ctx_id=0, det_thresh=.5, det_size=(640, 640))
            initialized = time.perf_counter() - started
            items = []
            for path, pixels in images:
                timings, first_faces = [], None
                for repeat in range(args.repeats):
                    started = time.perf_counter()
                    faces = app.get(pixels)
                    timings.append(time.perf_counter() - started)
                    if repeat == 0:
                        first_faces = face_records(faces)
                items.append({'image': str(path), 'image_sha256': sha256(path),
                              'first_inference_s': timings[0], 'inference_s': timings,
                              'warm_median_s': statistics.median(timings[1:]) if len(timings) > 1 else None,
                              'faces': first_faces})
            run = {'session_number': session_number, 'initialize_s': initialized,
                   'images': items, 'runtime': controller.report(finish_profiling=True)}
            if reference:
                run['comparison'] = compare_faces(reference['runs'][0]['images'], items, gallery)
                run['comparison']['gallery_source'] = 'provided_npy' if gallery is not None else 'baseline_sample_faces'
            report['runs'].append(run)
        except Exception as exc:
            report['runs'].append({'session_number': session_number,
                                   'error': f'{type(exc).__name__}: {exc}',
                                   'runtime': controller.report(finish_profiling=True)})
            report['error'] = f'{type(exc).__name__}: {exc}'
            break
        finally:
            controller.uninstall()
            del app
            gc.collect()
    report['sources_after'] = inspect_models(args.source_dir)
    report['source_models_unchanged'] = report['sources_before'] == report['sources_after']
    if not report['source_models_unchanged']:
        raise RuntimeError('Original model files changed during the probe')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true', help='Read model shapes/hashes; no sessions')
    parser.add_argument('--variant', choices=VARIANTS, default='cpu')
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_MODELS)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--images', nargs='+')
    parser.add_argument('--reference', type=Path, help='CPU probe JSON to compare')
    parser.add_argument('--gallery-npy', type=Path, help='Optional existing Nx512 identity gallery')
    parser.add_argument('--sessions', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--profile', action='store_true', help='Separate diagnostic run; profiling adds overhead')
    parser.add_argument('--compute-plan', action='store_true', help='Log CoreML hardware plan diagnostics')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.inspect:
        report = {'models': inspect_models(args.source_dir), 'inference_executed': False}
    else:
        if not args.images or args.sessions < 1 or args.repeats < 1:
            parser.error('Supply --images, and positive --sessions and --repeats')
        report = run_probe(args)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        output = experiment_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        print(json.dumps({'report': str(output), 'variant': args.variant,
                          'inference_executed': not args.inspect}, ensure_ascii=False))
    else:
        print(text)
    if report.get('error'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
