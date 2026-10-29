"""Full 44-video frame galleries; official SDPA via an isolated model worker."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import bench
from frame_worker import FrameWorker
from owl_sdpa_bridge import SDPAWorker

OUT = bench.OUT/'owl-sdpa-full-20261028'


def main():
    from action_search import media_catalog
    from evaluate import compare, verify_queries
    OUT.mkdir(exist_ok=False)
    before = bench.fingerprint()
    bench.dump(OUT/'production-before.json', before)
    catalog = media_catalog(Path(before['project']))
    inputs = OUT/'inputs'; inputs.mkdir()
    manifest = json.loads((bench.OUT/'queries.json').read_text())
    verify_queries(manifest)
    media = []
    for row in catalog:
        source = Path(row['location'])/row['path']
        if not source.is_absolute(): source = bench.ROOT/source
        name = f'{row["id"]:04d}-'+Path(row['path']).name
        (inputs/name).symlink_to(source)
        media.append({**row, 'path': name, 'location': str(inputs), 'sha256': bench.sha(source)})
    manifest['media'] = media
    manifest['production_project'] = before['project']
    ids = {m['id'] for m in media}
    for query in manifest['queries']:
        query['subset_source_media'] = sorted(set(query['source_media']) & ids)
        query['subset_known_class_media'] = sorted(set(query['known_class_media']) & ids)
    manifest['limitations'].append('All current 44 videos re-extracted at full configured frame rate; query labels remain limited, not exhaustive new annotations.')
    bench.dump(OUT/'queries.json', manifest)
    shutil.copy2(bench.OUT/'query-baseline.npz', OUT/'queries.npz')
    provenance = json.loads((bench.OUT/'query-baseline.json').read_text())
    provenance.update(manifest_sha256=bench.sha(OUT/'queries.json'), vectors_sha256=bench.sha(OUT/'queries.npz'),
                      original_provenance_sha256=bench.sha(bench.OUT/'query-baseline.json'))
    # evaluate expects vectors.with_suffix('.json'); keep manifest distinct.
    (OUT/'queries.json').rename(OUT/'manifest.json')
    provenance['manifest_sha256'] = bench.sha(OUT/'manifest.json')
    bench.dump(OUT/'queries.json', provenance)
    report = {'status': 'running', 'videos': len(media), 'duration_seconds': sum(m['duration'] for m in media),
              'scope': 'All frames at existing 2fps, original prepared videos; fresh complete OWL/face galleries, same frozen query-side vectors; no Qwen re-encoding.', 'trials': []}
    bench.dump(OUT/'result.json', report)
    try:
        for name, cls in [('baseline', FrameWorker), ('sdpa', SDPAWorker)]:
            folder = OUT/name; folder.mkdir(); project = folder/'project'
            worker = cls(bench.ROOT, threading.Event())
            row = {'variant': name, 'status': 'running'}; report['trials'].append(row)
            bench.dump(OUT/'result.json', report)
            with (folder/'extract.log').open('w', buffering=1) as log:
                try:
                    start = time.perf_counter()
                    row['worker'] = worker.extract([str(inputs), '--project-dir', str(project), '--num-workers', '0', '--yes',
                        '--video-feature-id', bench.OWL, '--video-feature-id', bench.FACE], log)
                    row['extract_seconds'] = time.perf_counter()-start
                finally:
                    worker.close()
            start = time.perf_counter()
            with (folder/'index.log').open('w') as log:
                subprocess.run([sys.executable, '-u', '-m', 'wise', 'create-index', '--project-dir', str(project), '--overwrite'],
                    cwd=bench.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            row['index_seconds'] = time.perf_counter()-start
            row['status'] = 'completed'
            bench.dump(OUT/'result.json', report)
            print(name, row['extract_seconds'], row['index_seconds'], flush=True)
        result = compare(OUT/'baseline/project', OUT/'sdpa/project', OUT/'manifest.json', OUT/'queries.npz',
                         OUT/'retrieval.json', people_db=bench.ROOT/'data/people.sqlite')
        for row in result['queries']: row['scope'] = 'full_library_44_reextracted'
        for row in result['summary']: row['scope'] = 'full_library_44_reextracted'
        result['limitations'].append('Comparison is baseline re-indexing versus SDPA re-indexing, not historical mixed-production embeddings.')
        bench.dump(OUT/'retrieval.json', result)
        for row in media:
            assert bench.sha(Path(row['location'])/row['path']) == row['sha256']
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        report['production_unchanged'] = bench.fingerprint() == before
        bench.dump(OUT/'result.json', report)
        assert report['production_unchanged']


if __name__ == '__main__':
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
