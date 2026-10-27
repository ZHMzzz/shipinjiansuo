"""Exploratory follow-up, chosen after viewing the prespecified ablation.

Keeps all primary results intact. Run with Demo stopped; restore it in the caller.
"""
import json
import time
import urllib.request

from PIL import Image, ImageOps

from ablate_reranker import (OUT, IDS, QUERIES, SPECIAL, VideoReranker,
                            ResourceSampler, decode, dump, score)


def main():
    try:
        urllib.request.urlopen('http://127.0.0.1:9670/mvp-api/actions/status', timeout=1)
    except OSError:
        pass
    else:
        raise RuntimeError('Stop Demo before this experiment')
    neutral = [('no_helmet_neutral', '一个人没有佩戴安全帽'),
               ('helmet_neutral', '一个人佩戴安全帽')]
    visible = [('bare_head', '一个头部裸露、可以看到头发的人'),
               ('helmet_visible', '一个头上戴着安全帽的人')]
    configs = [dict(id=crop or 'full', frames=16, pixels=1048576,
                    real_time=True, crop=crop) for crop in [None, 'head']]
    dump('followup-design.json', {
        'exploratory': True,
        'reason': 'PPE instruction plus neutral query separated 3/3 native image pairs; video single-factor variants did not improve.',
        'configs': configs, 'neutral_queries': neutral, 'visible_queries': visible,
        'paired_resize': '256x256, preserve aspect ratio, gray letterbox, no new detail',
        'video_instructions': {'neutral_queries': SPECIAL, 'visible_queries': 'original generic instruction'},
    })
    model = VideoReranker()
    report = {'status': 'running', 'exploratory': True, 'model_load_seconds': model.load(), 'runs': []}
    pairs = json.loads((OUT/'paired/manifest.json').read_text())
    for instruction in ['generic', 'ppe']:
        cfg = {} if instruction == 'generic' else {'instruction': SPECIAL}
        rows = []
        with ResourceSampler() as resources:
            for item in pairs['items']:
                im = Image.open(OUT/'paired'/item['file']).convert('RGB')
                im = ImageOps.pad(im, (256, 256), method=Image.Resampling.BICUBIC, color=(127, 127, 127))
                for category, query in QUERIES + neutral:
                    t = time.perf_counter()
                    value, detail = score(model, query, cfg, picture=im)
                    rows.append({**item, 'category': category, 'query': query, 'score': value,
                                 'seconds': time.perf_counter()-t, **detail})
        report['runs'].append({'id': 'paired_256_'+instruction, 'rows': rows, 'resources': resources.result})
        dump('followup-results.json', report)
        print(report['runs'][-1]['id'], round(resources.result['total_seconds'], 2), flush=True)
    for base in configs:
        for instruction, queries in [('ppe', neutral), ('generic', visible)]:
            cfg = {**base}
            if instruction == 'ppe':
                cfg['instruction'] = SPECIAL
            rows = []
            with ResourceSampler() as resources:
                for name in IDS:
                    frames, meta, _, boxes = decode(name, 16, base['crop'])
                    for category, query in queries:
                        t = time.perf_counter()
                        value, detail = score(model, query, cfg, frames, meta)
                        rows.append({'clip': name, 'category': category, 'query': query, 'score': value,
                                     'seconds': time.perf_counter()-t, 'crop_boxes': boxes, **detail})
            report['runs'].append({'id': base['id']+'_'+instruction, 'config': cfg,
                                   'rows': rows, 'resources': resources.result})
            dump('followup-results.json', report)
            print(report['runs'][-1]['id'], round(resources.result['total_seconds'], 2), flush=True)
    report['status'] = 'completed'
    dump('followup-results.json', report)
    model.unload()


if __name__ == '__main__':
    main()
