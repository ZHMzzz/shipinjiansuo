"""Offline, fixed-input PPE benchmark. Never imports or changes the production app."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / '.deps'))
sys.path.insert(0, str(ROOT / 'scripts'))
(HERE / '.config' / 'Ultralytics').mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(HERE / '.config')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_VERBOSE'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO, settings
from resource_metrics import ResourceSampler

settings.update({'sync': False})
torch.set_num_threads(1)
cv2.setNumThreads(1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def concurrent_experiments():
    found = []
    for p in psutil.process_iter(['pid', 'cmdline', 'memory_info']):
        cmd = p.info['cmdline'] or []
        scripts = [x for x in cmd if 'experiments/' in x and x.endswith('.py')]
        if scripts and p.pid != os.getpid() and not any('ppe-pretrained' in x for x in scripts):
            found.append({'pid': p.pid, 'scripts': scripts, 'rss_bytes': p.info['memory_info'].rss})
    return found


def stats(values):
    return {'count': len(values), 'total_seconds': sum(values), 'median_ms': statistics.median(values) * 1000,
            'p95_ms': float(np.percentile(values, 95)) * 1000, 'mean_ms': statistics.mean(values) * 1000,
            'frames_per_second': len(values) / sum(values)}


def run(model_id, tag, isolate=False):
    output = HERE / f'{model_id}-{tag}.json'
    if output.exists():
        raise SystemExit(f'Refusing to overwrite {output}')
    if not (HERE / 'labels.json').exists():
        raise SystemExit('Freeze labels before predicting')
    frozen = json.loads((HERE / 'labels-freeze.json').read_text())
    assert frozen['dataset_sha256'] == sha(HERE / 'dataset.json')
    assert frozen['labels_sha256'] == sha(HERE / 'labels.json')
    other = concurrent_experiments()
    if other and not isolate:
        raise SystemExit('Concurrent experiment detected: ' + json.dumps(other))
    initial_wait = time.perf_counter()
    while isolate and concurrent_experiments():
        time.sleep(1)
    initial_wait = time.perf_counter() - initial_wait
    other = concurrent_experiments()
    if not torch.backends.mps.is_available():
        raise SystemExit('MPS unavailable; do not silently change device')
    model_info = next(m for m in json.loads((HERE / 'models.json').read_text()) if m['id'] == model_id)
    assert sha(model_info['path']) == model_info['sha256']
    dataset = json.loads((HERE / 'dataset.json').read_text())
    images, frames = [], []
    read_start = time.perf_counter()
    for case in dataset['cases']:
        for f in case['frames']:
            assert sha(f['path']) == f['sha256'], f['path']
            im = cv2.imread(f['path'])
            assert im is not None and im.shape[:2] == (f['height'], f['width'])
            frames.append({**f, 'case_id': case['id']})
            images.append(im)
    image_read_seconds = time.perf_counter() - read_start
    options = dict(device='mps', imgsz=640, rect=False, batch=1, conf=0.25, iou=0.7,
                   half=False, max_det=100, augment=False, agnostic_nms=False, verbose=False,
                   save=False, stream=False)
    data = {'model': model_info, 'tag': tag, 'dataset_sha256': sha(HERE / 'dataset.json'),
            'labels_sha256': sha(HERE / 'labels.json'), 'pid': os.getpid(), 'options': options,
            'frame_count': len(frames), 'png_read_and_hash_seconds': image_read_seconds,
            'environment': {'python': sys.version, 'platform': platform.platform(),
                'packages': {n: importlib.metadata.version(n) for n in ['torch', 'torchvision', 'ultralytics', 'numpy', 'psutil']},
                'torch_threads': torch.get_num_threads(), 'mps_fallback_env': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')},
            'other_experiments_before': other, 'concurrent_process_checks': [], 'rounds': [],
            'isolate_each_prediction': isolate, 'initial_wait_seconds': initial_wait,
            'inter_frame_wait_seconds': 0.0, 'discarded_contended_predictions': []}
    with ResourceSampler() as total_resources:
        with ResourceSampler() as load_resources:
            begin = time.perf_counter()
            model = YOLO(model_info['path'])
            data['checkpoint_load_seconds'] = time.perf_counter() - begin
        data['load_resources'] = load_resources.result
        data['names'] = model.names
        data['model_parameters'] = sum(x.numel() for x in model.model.parameters())
        data['checkpoint_ultralytics_version'] = model.ckpt.get('version')
        data['architecture'] = model.model.yaml.get('yaml_file', model.ckpt.get('train_args', {}).get('model'))
        input_signatures = set()
        def observe_inputs(module, inputs):
            x = inputs[0]
            if isinstance(x, torch.Tensor):
                input_signatures.add((tuple(x.shape), str(x.dtype), str(x.device)))
        hook = model.model.register_forward_pre_hook(observe_inputs)
        no_helmet_ids = [int(k) for k, v in model.names.items() if v.lower().replace('-', '_') == 'no_helmet']
        helmet_ids = [int(k) for k, v in model.names.items() if v.lower().replace('-', '_') in {'helmet', 'hard_hat'}]
        person_ids = [int(k) for k, v in model.names.items() if v.lower() in {'human', 'person', 'worker'}]
        assert len(no_helmet_ids) == 1 and helmet_ids and person_ids, model.names
        data['class_mapping'] = {'no_helmet': no_helmet_ids, 'helmet': helmet_ids, 'person': person_ids}
        with ResourceSampler() as cold_resources:
            torch.mps.synchronize()
            begin = time.perf_counter()
            model.predict(images[0], **options)
            torch.mps.synchronize()
            data['first_predict_seconds'] = time.perf_counter() - begin
        data['cold_resources'] = cold_resources.result
        begin = time.perf_counter()
        for _ in range(5):
            model.predict(images[0], **options)
        torch.mps.synchronize()
        data['additional_warmup_seconds'] = time.perf_counter() - begin
        data['additional_warmup_calls'] = 5
        for round_id in range(3):
            records = []
            with ResourceSampler() as resources:
                begin_round = time.perf_counter()
                for i, (frame, im) in enumerate(zip(frames, images)):
                    while True:
                        wait_start = time.perf_counter()
                        while isolate and concurrent_experiments():
                            time.sleep(1)
                        data['inter_frame_wait_seconds'] += time.perf_counter() - wait_start
                        torch.mps.synchronize()
                        begin = time.perf_counter()
                        result = model.predict(im, **options)[0]
                        torch.mps.synchronize()
                        boxes = result.boxes.data.detach().cpu().numpy().tolist()
                        seconds = time.perf_counter() - begin
                        contended = concurrent_experiments() if isolate else []
                        if not contended:
                            break
                        data['discarded_contended_predictions'].append({'round': round_id + 1, 'frame_id': frame['id'],
                            'seconds': seconds, 'other_experiments': contended})
                    rows = [{'xyxy': b[:4], 'confidence': b[4], 'class_id': int(b[5]), 'class_name': model.names[int(b[5])]} for b in boxes]
                    records.append({'frame_id': frame['id'], 'case_id': frame['case_id'], 'timestamp': frame['timestamp'],
                        'predict_seconds': seconds, 'ultralytics_speed_ms': result.speed, 'detections': rows})
                    resources.sample()
                    if i == len(frames) - 1 or frame['case_id'] != frames[i + 1]['case_id']:
                        data['concurrent_process_checks'].append({'round': round_id + 1, 'case': frame['case_id'],
                            'experiments': concurrent_experiments()})
                        print(json.dumps({'model': model_id, 'round': round_id + 1, 'done_frames': i + 1, 'case': frame['case_id']}), flush=True)
                wall_seconds = time.perf_counter() - begin_round
            data['rounds'].append({'number': round_id + 1, 'wall_seconds': wall_seconds,
                'predict_stats': stats([r['predict_seconds'] for r in records]), 'records': records, 'resources': resources.result})
            save(output, data)
    data['all_phases_resources'] = total_resources.result
    hook.remove()
    data['observed_model_inputs'] = sorted(input_signatures)
    data['other_experiments_after'] = concurrent_experiments()
    data['repeated_outputs_identical_4decimals'] = len({json.dumps([
        [(d['class_id'], round(d['confidence'], 4), [round(x, 3) for x in d['xyxy']]) for d in r['detections']]
        for r in rnd['records']]) for rnd in data['rounds']}) == 1
    data['combined_predict_stats'] = stats([r['predict_seconds'] for rnd in data['rounds'] for r in rnd['records']])
    save(output, data)
    print(json.dumps({'model': model_id, 'result': str(output), 'summary': data['combined_predict_stats'],
        'resources': data['all_phases_resources']['sampled_peak'], 'repeat_identical': data['repeated_outputs_identical_4decimals']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', choices=['yolo11-ppe', 'yolov8-ppe'])
    parser.add_argument('--tag', default='mps-fp32')
    parser.add_argument('--isolate', action='store_true', help='Only accept calls with no other experiment before and after')
    args = parser.parse_args()
    run(args.model_id, args.tag, args.isolate)
