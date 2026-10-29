"""Frozen pixels, production weights, independent official SDPA/compile trials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import bench

OUT = bench.OUT/'owl-official-20261028'


def prepare():
    import cv2
    import numpy as np
    import torch
    from transformers import Owlv2Processor
    from action_search import media_catalog
    from evaluate import verify_queries
    OUT.mkdir(exist_ok=False)
    bench.dump(OUT/'production-before.json', bench.fingerprint())
    queries = json.loads((bench.OUT/'queries.json').read_text())
    verify_queries(queries)
    assert bench.sha(bench.OUT/'query-baseline.npz') == json.loads((bench.OUT/'query-baseline.json').read_text())['vectors_sha256']
    processor = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble', use_fast=True, local_files_only=True)
    rows = []
    for media in media_catalog(bench.ROOT/'projects/demo'):
        if media['id'] not in bench.MEDIA_IDS:
            continue
        source = Path(media['location'])/media['path']
        if not source.is_absolute():
            source = bench.ROOT/source
        cap = cv2.VideoCapture(str(source))
        for ratio in (0., .25, .5, .75):
            sec = float(media['duration'])*ratio
            cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
            ok, bgr = cap.read()
            assert ok, (source, sec)
            x = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0)
            pixels = processor(images=x, return_tensors='pt', device='cpu')['pixel_values']
            file = f'pixels-{len(rows):02d}.npy'
            np.save(OUT/file, pixels.numpy())
            rows.append({'media_id': media['id'], 'ts': sec, 'pixels': file, 'shape': list(pixels.shape),
                         'pixels_sha256': bench.sha(OUT/file), 'source_sha256': bench.sha(source)})
        cap.release()
    assert len(rows) == len(bench.MEDIA_IDS)*4
    bench.dump(OUT/'design.json', {'frames': rows, 'query_ids': list(np.load(bench.OUT/'query-baseline.npz').keys()),
        'scope': '44 fixed preprocessed frames across 11 videos; resident model and postprocessing only; no ingestion, decoder, preprocessing, write or publication',
        'query_sha256': bench.sha(bench.OUT/'query-baseline.npz')})
    print('Prepared', len(rows), 'frames', flush=True)


def run(label, variants, rounds):
    import numpy as np
    import torch
    import transformers
    from transformers import Owlv2ForObjectDetection
    from wise.feature.transformers_owlv2 import get_object_features, sort_by_objectness
    from resource_metrics import ResourceSampler
    torch.set_num_threads(1)
    design = json.loads((OUT/'design.json').read_text())
    target = OUT/label
    target.mkdir(exist_ok=False)
    pixels = []
    for row in design['frames']:
        assert bench.sha(OUT/row['pixels']) == row['pixels_sha256']
        pixels.append(torch.from_numpy(np.load(OUT/row['pixels'])))
    loaded = time.perf_counter()
    net, info = Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble',
        local_files_only=True, torch_dtype=torch.float16, attn_implementation='eager', output_loading_info=True)
    assert not info.get('missing_keys') and not info.get('unexpected_keys'), info
    net = net.eval().to('mps')
    assert {p.dtype for p in net.parameters() if p.is_floating_point()} == {torch.float16}
    weights = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        weights.update(name.encode()); weights.update(value.detach().cpu().contiguous().numpy().tobytes())
    report = {'status': 'running', 'torch': torch.__version__, 'transformers': transformers.__version__,
              'weights_sha256': weights.hexdigest(), 'load_seconds': time.perf_counter()-loaded,
              'scope': design['scope'], 'variants': {}, 'rows': []}
    original = net.owlv2.vision_model.forward
    methods = {'eager': original, 'sdpa': original}
    current = None

    def select(variant):
        nonlocal current
        if variant == current:
            return
        if hasattr(net, 'set_attn_implementation'):
            net.set_attn_implementation('sdpa' if variant == 'sdpa' else 'eager')
        elif variant == 'sdpa':
            raise RuntimeError('Installed official model has no attention backend switch')
        net.owlv2.vision_model.forward = methods[variant]
        current = variant

    def infer(x):
        raw = get_object_features(net, x.to('mps'))
        scores, vectors, boxes = sort_by_objectness(*(v.float() for v in raw))
        # Same production order: sort on MPS, copy all rows, threshold on CPU.
        vectors = vectors.cpu().numpy(); scores = scores.cpu().numpy(); boxes = boxes.cpu().numpy()
        keep = scores[0] >= .02
        return vectors[0, keep], scores[0, keep], boxes[0, keep], raw

    successful = []
    try:
        with torch.inference_mode():
            # Prime common MPS kernels before comparing backend changes.
            select('eager')
            infer(pixels[0]); torch.mps.synchronize()
            for variant in variants:
                data = {'status': 'warming'}
                report['variants'][variant] = data
                bench.dump(target/'result.json', report)
                started = time.perf_counter()
                try:
                    if variant == 'compiled':
                        torch._dynamo.config.suppress_errors = False
                        torch._dynamo.utils.counters.clear()
                        methods[variant] = torch.compile(original, backend='inductor', fullgraph=True, dynamic=False)
                    select(variant)
                    with ResourceSampler() as resources:
                        for x in pixels[:2]:
                            infer(x)
                        torch.mps.synchronize()
                    data.update(status='ready', first_two_frames_seconds=time.perf_counter()-started,
                                warmup_resources=resources.result)
                    if variant == 'compiled':
                        data['dynamo_counters'] = {k: dict(v) for k, v in torch._dynamo.utils.counters.items()}
                        assert torch._dynamo.utils.counters['stats']['unique_graphs'] > 0, 'No compiled graph executed'
                    successful.append(variant)
                except Exception as exc:
                    data.update(status='failed', first_two_frames_seconds=time.perf_counter()-started,
                                error=repr(exc), traceback=traceback.format_exc())
                    print('FAILED', label, variant, repr(exc)[:400], flush=True)
                    net.owlv2.vision_model.forward = original
                    current = None
                bench.dump(target/'result.json', report)
            for repeat in range(rounds):
                for variant in (successful if repeat % 2 == 0 else list(reversed(successful))):
                    select(variant)
                    with ResourceSampler() as resources:
                        torch.mps.synchronize(); started = time.perf_counter()
                        values = [infer(x)[:3] for x in pixels]
                        torch.mps.synchronize(); seconds = time.perf_counter()-started
                    report['rows'].append({'variant': variant, 'repeat': repeat, 'seconds': seconds,
                        'regions': sum(len(x[0]) for x in values), 'resources': resources.result})
                    bench.dump(target/'result.json', report)
                    print(label, variant, repeat, round(seconds, 3), flush=True)
            # Separate memory probe: drop unused allocator caches between variants.
            # These samples are outside the steady-state timing rows above.
            import gc
            for variant in successful:
                select(variant)
                gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()
                with ResourceSampler() as resources:
                    for x in pixels[:4]:
                        infer(x)
                    torch.mps.synchronize()
                report['variants'][variant]['memory_probe'] = resources.result
            # Raw patch alignment, unlike region ordering, survives threshold crossings.
            for variant in successful:
                select(variant); output = {}
                for i, x in enumerate(pixels):
                    vectors, scores, boxes, raw = infer(x)
                    output[f'vectors_{i}'] = vectors
                    output[f'scores_{i}'] = scores
                    output[f'boxes_{i}'] = boxes
                    for name, value in zip(('raw_scores', 'raw_vectors', 'raw_boxes'), raw):
                        output[f'{name}_{i}'] = value.float().cpu().numpy()
                np.savez(target/(variant+'.npz'), **output)
                report['variants'][variant]['output_sha256'] = bench.sha(target/(variant+'.npz'))
                report['variants'][variant]['mean_seconds'] = statistics.mean(r['seconds'] for r in report['rows'] if r['variant'] == variant)
            report['status'] = 'completed'
    finally:
        net.owlv2.vision_model.forward = original
        report['production_unchanged'] = bench.fingerprint() == json.loads((OUT/'production-before.json').read_text())
        bench.dump(target/'result.json', report)
        assert report['production_unchanged']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--label')
    parser.add_argument('--variants', default='eager,sdpa')
    parser.add_argument('--rounds', type=int, default=4)
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else run(args.label, args.variants.split(','), args.rounds)
