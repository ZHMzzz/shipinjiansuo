"""Resident OWLv2 component timings; synchronized hooks are diagnostic only."""
import hashlib
import json
from pathlib import Path
import statistics
import time

import bench


def main():
    import cv2
    import torch
    from frame_acceleration import enable
    from action_search import media_catalog
    from wise.feature.feature_extractor_factory import FeatureExtractorFactory

    out = bench.OUT/'owl-backbone-20261028.json'
    if out.exists():
        raise RuntimeError('Do not overwrite previous evidence')
    before = bench.fingerprint()
    torch.set_num_threads(1)
    enable()
    media = next(m for m in media_catalog(Path(before['project'])) if m['id'] == 40)
    source = Path(media['location'])/media['path']
    if not source.is_absolute():
        source = bench.ROOT/source
    cap = cv2.VideoCapture(str(source))
    frames = []
    for sec in (0, 5, 15, 30):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
        ok, bgr = cap.read()
        assert ok
        frames.append(torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0))
    cap.release()
    ext = FeatureExtractorFactory(bench.OWL, {'default': {'device': 'mps', 'compile': False, 'warmup': True}})
    net = ext.model.model
    vision = net.owlv2.vision_model
    timings = {}
    originals = []

    def wrap(module, key):
        original = module.forward
        def timed(*args, **kwargs):
            torch.mps.synchronize()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            torch.mps.synchronize()
            timings[key] = timings.get(key, 0.) + time.perf_counter()-start
            return result
        originals.append((module, original))
        module.forward = timed

    def signature(result):
        return [{'vectors': hashlib.sha256(f.vectors.tobytes()).hexdigest(),
                 'metadata': hashlib.sha256(repr(f.metadata).encode()).hexdigest(),
                 'regions': len(f.vectors)} for values in result for f in values]

    def run():
        torch.mps.synchronize()
        start = time.perf_counter()
        values = [ext.extract_image_features(x) for x in frames]
        torch.mps.synchronize()
        return time.perf_counter()-start, signature(values)

    with torch.inference_mode():
        run()
        plain_before = [run() for _ in range(2)]
        wrap(vision, 'vision_total_inclusive')
        wrap(vision.embeddings, 'embeddings')
        wrap(vision.embeddings.patch_embedding, 'patch_conv_inclusive_in_embeddings')
        for layer in vision.encoder.layers:
            wrap(layer.self_attn, 'attention_including_qkv')
            wrap(layer.mlp, 'mlp')
            wrap(layer.layer_norm1, 'encoder_norms')
            wrap(layer.layer_norm2, 'encoder_norms')
        try:
            instrumented = []
            for _ in range(2):
                timings.clear()
                seconds, sig = run()
                instrumented.append({'seconds': seconds, 'parts': dict(timings), 'signature': sig})
        finally:
            for module, original in originals:
                module.forward = original
        plain_after = [run() for _ in range(2)]
    baseline = plain_before[0][1]
    assert all(sig == baseline for _, sig in plain_before + plain_after)
    assert all(row['signature'] == baseline for row in instrumented)
    report = {'scope': '4 resident real frames, synchronized instrumentation, not a speedup test; nested keys overlap',
              'torch': torch.__version__, 'frames_seconds': [0, 5, 15, 30],
              'plain_seconds': [r[0] for r in plain_before + plain_after],
              'instrumented': instrumented,
              'plain_mean_seconds': statistics.mean(r[0] for r in plain_before + plain_after),
              'instrumented_mean_seconds': statistics.mean(r['seconds'] for r in instrumented),
              'parts_mean_seconds': {key: statistics.mean(r['parts'][key] for r in instrumented) for key in timings},
              'outputs_exact': True, 'production_unchanged': before == bench.fingerprint()}
    assert report['production_unchanged']
    bench.dump(out, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'instrumented'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
