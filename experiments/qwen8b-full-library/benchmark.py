"""Isolated, sequential deployment benchmark; never writes to production data.

Run with the project's existing environment after sourcing scripts/env.sh.
The 2B baseline imports the production extractor without modifying it.
The 8B candidate uses a separately downloaded llama.cpp Q8_0 server.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / 'experiments/qwen8b-int8'
SOURCE = ROOT / 'experiments/action-search'
MANIFEST = json.loads((SOURCE / 'manifest.json').read_text())
FRAMES = HERE / 'frames'
PROMPT_PREFIX = "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\n"
PROMPT_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n'


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def prepare():
    import cv2
    import numpy as np
    FRAMES.mkdir(exist_ok=True)
    clips = []
    for item in MANIFEST['clips']:
        source = SOURCE / item['file']
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == item['sha256'], item['id']
        cap = cv2.VideoCapture(str(source))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        indices = np.linspace(0, count - 1, 16).astype(int)
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode {source} frame {index}')
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        array = np.stack(frames)
        np.save(FRAMES / (item['id'] + '.npy'), array)
        (FRAMES / (item['id'] + '.rgb')).write_bytes(array.tobytes())
        h, w = array.shape[1:3]
        # A lossless transport for the same 16 RGB frames. Synthetic 2fps matches
        # the current production extractor's list-of-frames metadata behavior.
        cmd = [str(ROOT / '.media-runtime/bin/ffmpeg'), '-v', 'error', '-y',
               '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{w}x{h}',
               '-framerate', '2', '-i', 'pipe:0', '-c:v', 'ffv1', '-pix_fmt', 'bgr0',
               str(FRAMES / (item['id'] + '.mkv'))]
        subprocess.run(cmd, input=array.tobytes(), check=True)
        # Verify the lossless transport, including frame count and RGB values.
        cap = cv2.VideoCapture(str(FRAMES / (item['id'] + '.mkv')))
        recovered = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            recovered.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        assert np.array_equal(array, np.stack(recovered)), item['id']
        clips.append({**item, 'source_duration_seconds': count / fps,
                      'source_indices': indices.tolist(), 'rgb_shape': list(array.shape),
                      'rgb_sha256': hashlib.sha256(array.tobytes()).hexdigest()})
    save(HERE / 'dataset.json', {'clips': clips, 'queries': MANIFEST['queries'],
         'notes': MANIFEST['notes'], 'transport': '16 lossless RGB frames at synthetic 2 fps',
         'pixels_budget_2b': 1048576})
    print('Prepared and verified', len(clips), 'clips', flush=True)


def footprint(pid):
    """macOS physical footprint, same accounting for PyTorch and llama.cpp."""
    lib = ctypes.CDLL('/usr/lib/libproc.dylib')
    buf = ctypes.create_string_buffer(256)
    if lib.proc_pid_rusage(pid, 0, ctypes.byref(buf)) != 0:
        return None
    return int.from_bytes(buf.raw[72:80], 'little')


def align_frames():
    """Use the production qwen_vl_utils resize before libmtmd's RGB boundary."""
    import numpy as np
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    dataset = json.loads((HERE / 'dataset.json').read_text())
    baseline = json.loads((HERE / '2b-fp16.json').read_text())
    for item in dataset['clips']:
        array = np.load(FRAMES / (item['id'] + '.npz'))['frames']
        messages = [{'role': 'user', 'content': [{'type': 'video',
            'video': [Image.fromarray(frame) for frame in array], 'total_pixels': 1048576}]}]
        _, videos, _ = process_vision_info(messages, image_patch_size=16,
            return_video_metadata=True, return_video_kwargs=True)
        floats = videos[0][0].numpy().transpose(0, 2, 3, 1)
        rgb = np.rint(floats).clip(0, 255).astype('uint8')
        grid = next(x for x in baseline['clips'] if x['id'] == item['id'])['input']['video_grid_thw'][0]
        assert list(rgb.shape) == [16, grid[1] * 16, grid[2] * 16, 3]
        (FRAMES / (item['id'] + '.aligned.rgb')).write_bytes(rgb.tobytes())
        item['aligned_rgb_shape'] = list(rgb.shape)
        item['aligned_rgb_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
        item['rgb_rounding_max_abs_error'] = float(np.max(np.abs(floats - rgb)))
    dataset['transport'] = 'same sampled frames; production resize; uint8 rounding for libmtmd RGB API'
    save(HERE / 'aligned-inputs.json', {'clips': dataset['clips'], 'transport': dataset['transport']})
    print('Aligned frames to production visual token grids', flush=True)


class Monitor:
    def __init__(self, pid):
        import psutil
        self.psutil = psutil
        self.proc = psutil.Process(pid)
        self.samples = []
        self.stop = threading.Event()
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        processes = [self.proc] + self.proc.children(recursive=True)
        rss = physical = 0
        for p in processes:
            try:
                rss += p.memory_info().rss
                physical += footprint(p.pid) or 0
            except self.psutil.NoSuchProcess:
                pass
        try:
            phase = json.loads((HERE / 'phase.json').read_text())['phase']
        except (FileNotFoundError, json.JSONDecodeError):
            phase = 'starting'
        vm = self.psutil.virtual_memory()
        row = {'phase': phase, 'unix': time.time(), 'seconds': time.perf_counter() - self.started, 'rss_bytes': rss,
               'physical_footprint_bytes': physical, 'system_available_bytes': vm.available,
               'swap_used_bytes': self.psutil.swap_memory().used}
        self.samples.append(row)
        return row

    def loop(self):
        while not self.stop.wait(.1):
            try:
                self.sample()
            except self.psutil.NoSuchProcess:
                return

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        try:
            self.sample()
        except self.psutil.NoSuchProcess:
            pass

    def summary(self):
        return {'sample_interval_seconds': .1, 'scope': 'worker and descendants',
                'peak_rss_bytes': max(x['rss_bytes'] for x in self.samples),
                'peak_physical_footprint_bytes': max(x['physical_footprint_bytes'] for x in self.samples),
                'minimum_system_available_bytes': min(x['system_available_bytes'] for x in self.samples),
                'initial_swap_used_bytes': self.samples[0]['swap_used_bytes'],
                'peak_swap_used_bytes': max(x['swap_used_bytes'] for x in self.samples)}


def metrics(ranking, expected):
    relevant = set(expected)
    if not relevant: return {}
    flags = [x['id'] in relevant for x in ranking]
    first = next((i + 1 for i, flag in enumerate(flags) if flag), None)
    return {'hit_at_1': int(any(flags[:1])), 'hit_at_3': int(any(flags[:3])),
            'hit_at_5': int(any(flags[:5])), 'recall_at_3': sum(flags[:3]) / len(relevant),
            'recall_at_5': sum(flags[:5]) / len(relevant), 'reciprocal_rank': 1 / first if first else 0}


def worker(args):
    import numpy as np
    import psutil
    dataset = json.loads((HERE / 'dataset.json').read_text())
    if args.backend == '8b-q8':
        aligned = json.loads((HERE / 'aligned-inputs.json').read_text())
        by_id = {c['id']: c for c in aligned['clips']}
        dataset['clips'] = [by_id[c['id']] for c in dataset['clips']]
    out = HERE / (args.backend + '.json')
    report = {'backend': args.backend, 'status': 'running', 'clips': [], 'queries': [],
              'query_rounds': args.rounds, 'pid': os.getpid(), 'model_load_seconds': None}
    save(out, report)
    def phase(name):
        save(HERE / 'phase.json', {'phase': name, 'unix': time.time()})
        report.setdefault('phase_boundaries', []).append({'phase': name, 'unix': time.time()})
    phase('loading')
    server = None
    if args.backend == '2b-fp16':
        import torch
        sys.path.insert(0, str(ROOT / 'wise/src'))
        from wise.feature.qwen3vl import Qwen3VLEmbeddingFeatureExtractor as Extractor
        torch.set_num_threads(1)
        torch.mps.set_per_process_memory_fraction(.55)
        metal_peaks = {'allocated_bytes': 0, 'driver_bytes': 0}
        def sample_metal():
            while True:
                metal_peaks['allocated_bytes'] = max(metal_peaks['allocated_bytes'], torch.mps.current_allocated_memory())
                metal_peaks['driver_bytes'] = max(metal_peaks['driver_bytes'], torch.mps.driver_allocated_memory())
                time.sleep(.05)
        threading.Thread(target=sample_metal, daemon=True).start()
        shapes = {}
        t = time.perf_counter()
        extractor = Extractor('hf/Qwen/Qwen3-VL-Embedding/2B', device='mps', warmup=False,
            config=Extractor.Config(device='mps', compile=False, warmup=False,
                                    dtype='float16', max_total_pixels=1048576))
        _ = extractor._model
        torch.mps.synchronize()
        report['model_load_seconds'] = time.perf_counter() - t

        def capture(module, inputs, kwargs):
            shapes['tokens'] = int(kwargs['input_ids'].shape[-1])
            if kwargs.get('video_grid_thw') is not None:
                shapes['video_grid_thw'] = kwargs['video_grid_thw'].cpu().tolist()
        extractor._model.register_forward_pre_hook(capture, with_kwargs=True)

        def video(item):
            frames = np.load(FRAMES / (item['id'] + '.npz'))['frames']
            tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
            t = time.perf_counter()
            v = extractor.extract_video_segment_features(tensor).vectors[0]
            torch.mps.synchronize()
            return v, time.perf_counter() - t, dict(shapes)

        def text(query):
            t = time.perf_counter()
            v = extractor.extract_text_features([query])[0]
            torch.mps.synchronize()
            return v, time.perf_counter() - t

        def memory():
            return {'rss_bytes': psutil.Process().memory_info().rss,
                    'physical_footprint_bytes': footprint(os.getpid()),
                    'mps_allocated_bytes': torch.mps.current_allocated_memory(),
                    'mps_driver_bytes': torch.mps.driver_allocated_memory(),
                    'sampled_metal_peak_bytes': metal_peaks['driver_bytes']}
        report['runtime'] = {'torch': torch.__version__, 'device': 'mps', 'dtype': 'float16'}
    else:
        import select
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        torch.set_num_threads(1)
        binary = HERE / 'aligned-encoder'
        cmd = [str(binary), str(ASSETS / 'models/Qwen3-VL-Embedding-8B.Q8_0.gguf'),
               str(ASSETS / 'models/Qwen3-VL-Embedding-8B.mmproj-Q8_0.gguf'),
               str(ASSETS / 'runtime/llama-b11944')]
        report['server_command'] = cmd
        log = (HERE / '8b-q8-server.log').open('w')
        t = time.perf_counter()
        server = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=log, text=True, bufsize=1)
        report['server_pid'] = server.pid

        def read_response(timeout=180):
            if not select.select([server.stdout], [], [], timeout)[0]:
                raise TimeoutError('aligned encoder timed out')
            line = server.stdout.readline()
            if not line:
                raise RuntimeError('aligned encoder exited; inspect 8b-q8-server.log')
            result = json.loads(line)
            if 'error' in result:
                raise RuntimeError(result['error'])
            return result

        try:
            ready = read_response(240)
            assert ready['ready']
        except BaseException:
            server.terminate()
            server.wait(timeout=15)
            raise
        report['model_load_seconds'] = time.perf_counter() - t
        metal_memory = {'metal_allocated_bytes': ready['metal_allocated_bytes'],
                        'sampled_metal_peak_bytes': ready['metal_peak_bytes']}
        def request(payload):
            t = time.perf_counter()
            server.stdin.write(json.dumps(payload, ensure_ascii=False) + '\n')
            server.stdin.flush()
            data = read_response()
            metal_memory.update(metal_allocated_bytes=data['metal_allocated_bytes'],
                sampled_metal_peak_bytes=max(metal_memory['sampled_metal_peak_bytes'], data['metal_peak_bytes']),
                request_metal_peak_bytes=data['metal_peak_bytes'])
            return np.array(data['embedding'], dtype='float32'), time.perf_counter() - t, data['input']

        def video(item):
            array = np.load(FRAMES / (item['id'] + '.npz'))['frames']
            started = time.perf_counter()
            messages = [{'role': 'user', 'content': [{'type': 'video',
                'video': [Image.fromarray(frame) for frame in array], 'total_pixels': 1048576}]}]
            _, videos, _ = process_vision_info(messages, image_patch_size=16,
                return_video_metadata=True, return_video_kwargs=True)
            floats = videos[0][0].numpy().transpose(0, 2, 3, 1)
            rgb = np.rint(floats).clip(0, 255).astype('uint8')
            (FRAMES / (item['id'] + '.aligned.rgb')).write_bytes(rgb.tobytes())
            preprocess_seconds = time.perf_counter() - started
            h, w = item['aligned_rgb_shape'][1:3]
            result = request({'prefix': PROMPT_PREFIX, 'suffix': PROMPT_SUFFIX,
                'timestamps': [f'<{i + .25:.1f} seconds>' for i in range(8)],
                'rgb_path': str(FRAMES / (item['id'] + '.aligned.rgb')), 'width': w, 'height': h})
            total_seconds = time.perf_counter() - started
            reference = json.loads((ASSETS / 'template-reference.json').read_text())
            expected_text = [x for x in reference['tokens'] if x != 151656]
            info = result[2]
            info['preprocess_seconds'] = preprocess_seconds
            info['encoder_request_seconds'] = result[1]
            if info['text_token_ids'] != expected_text:
                save(HERE / 'alignment-error.json', {'expected': expected_text, 'actual': info})
                raise ValueError('Video template/timestamp/EOS token alignment failed')
            baseline = json.loads((HERE / '2b-fp16.json').read_text())
            clip = next(x for x in baseline['clips'] if x['id'] == item['id'])
            grid = clip['input']['video_grid_thw'][0]
            expected_visual = [grid[1] * grid[2] // 4] * grid[0]
            if info['visual_chunk_tokens'] != expected_visual:
                save(HERE / 'alignment-error.json', {'expected_visual': expected_visual, 'actual': info})
                raise ValueError('Visual token budget alignment failed')
            return result[0], total_seconds, info

        def text(query):
            v, seconds, _ = request({'prompt': PROMPT_PREFIX + query + PROMPT_SUFFIX})
            return v, seconds

        def memory():
            return {'server_rss_bytes': psutil.Process(server.pid).memory_info().rss,
                    'server_physical_footprint_bytes': footprint(server.pid),
                    'client_rss_bytes': psutil.Process().memory_info().rss, **metal_memory}
        report['runtime'] = {'llama_cpp': 'b11944 / 1b4554eff', 'device': 'Metal',
                             'language_quantization': 'Q8_0', 'vision_quantization': 'Q8_0',
                             'transport': 'raw RGB frames, aligned template/timestamps/EOS',
                             'context': 4096, 'batch': 2048, 'threads': 4}
    try:
        report['loaded_memory'] = memory()
        save(out, report)
        print('LOADED', args.backend, report['model_load_seconds'], flush=True)
        phase('video')
        vectors = []
        video_started = time.perf_counter()
        for item in dataset['clips']:
            v, seconds, shape = video(item)
            assert np.isfinite(v).all() and np.linalg.norm(v) > .5
            v = v / np.linalg.norm(v)
            vectors.append(v)
            report['clips'].append({'id': item['id'], 'encode_seconds': seconds, 'input': shape,
                                    'dimension': len(v), 'memory': memory()})
            save(out, report)
            print('VIDEO', item['id'], round(seconds, 3), shape, flush=True)
        report['all_segments_wall_seconds'] = time.perf_counter() - video_started
        report['video_phase_memory'] = memory()
        phase('video_repeat')
        # Repeat after warm-up to separate first-run setup from sustained encoding.
        report['warm_video_repeats'] = []
        for item in dataset['clips'][:3]:
            v, seconds, shape = video(item)
            original = vectors[[x['id'] for x in dataset['clips']].index(item['id'])]
            report['warm_video_repeats'].append({'id': item['id'], 'encode_seconds': seconds,
                'cosine_to_first': float(original @ (v / np.linalg.norm(v)))})
        matrix = np.stack(vectors).astype('float32')
        np.save(HERE / (args.backend + '-vectors.npy'), matrix)
        import faiss
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        report['post_video_memory'] = memory()
        if args.backend == '2b-fp16':
            metal_peaks.update(allocated_bytes=0, driver_bytes=0)
        else:
            metal_memory['sampled_metal_peak_bytes'] = 0
        phase('query')
        qvectors = []
        for round_index in range(args.rounds):
            for qindex, query in enumerate(dataset['queries']):
                v, seconds = text(query['text'])
                assert np.isfinite(v).all() and np.linalg.norm(v) > .5
                v = (v / np.linalg.norm(v)).astype('float32')
                t = time.perf_counter()
                scores, ids = index.search(v[None], len(matrix))
                search_seconds = time.perf_counter() - t
                ranking = [{'id': dataset['clips'][int(i)]['id'], 'score': float(s)}
                           for i, s in zip(ids[0], scores[0])]
                row = {**query, 'round': round_index, 'query_index': qindex,
                       'encode_seconds': seconds, 'search_seconds': search_seconds,
                       'ranking': ranking, 'memory': memory(), 'metrics': metrics(ranking, query['expected'])}
                report['queries'].append(row)
                qvectors.append(v)
                save(out, report)
                print('QUERY', round_index, query['text'], round(seconds, 3), ranking[:1], flush=True)
        np.save(HERE / (args.backend + '-query-vectors.npy'), np.stack(qvectors))
        report['final_memory'] = memory()
        phase('finishing')
        faiss.write_index(index, str(HERE / (args.backend + '-index.faiss')))
        report['status'] = 'completed'
        save(out, report)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def parent(args):
    # One model at a time. Parent samples both worker and its server/decode children.
    env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    cmd = [sys.executable, str(Path(__file__).resolve()), '--worker',
           '--backend', args.backend, '--rounds', str(args.rounds)]
    started = time.perf_counter()
    save(HERE / 'phase.json', {'phase': 'starting', 'unix': time.time()})
    with (HERE / (args.backend + '.log')).open('w') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        with Monitor(proc.pid) as monitor:
            reason = None
            while proc.poll() is None:
                if monitor.samples[-1]['system_available_bytes'] < 2 * 1024**3:
                    reason = 'less_than_2GiB_available'
                if time.perf_counter() - started > 1800:
                    reason = '30_minute_timeout'
                if reason:
                    for child in monitor.proc.children(recursive=True):
                        child.terminate()
                    proc.terminate()
                    break
                time.sleep(.2)
            try:
                code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
    summary = {**monitor.summary(), 'exit_code': code, 'stop_reason': reason,
               'wall_seconds': time.perf_counter() - started, 'samples': monitor.samples}
    save(HERE / (args.backend + '-resources.json'), summary)
    if code:
        path = HERE / (args.backend + '.json')
        report = json.loads(path.read_text()) if path.exists() else {}
        report.update(status='failed', exit_code=code, stop_reason=reason or 'process_error')
        save(path, report)
    print(json.dumps({k: v for k, v in summary.items() if k != 'samples'}, indent=2))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--align-frames', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--backend', choices=['2b-fp16', '8b-q8'], default='2b-fp16')
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if args.prepare:
        raise SystemExit('Use prepare.py, then freeze_labels.py before encoding')
    elif args.align_frames:
        align_frames()
    elif args.worker:
        worker(args)
    else:
        sys.exit(parent(args))
