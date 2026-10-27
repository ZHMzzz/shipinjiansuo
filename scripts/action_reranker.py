"""Local video reranking, adapted from Qwen's official Transformers example.

Reference: Qwen/Qwen3-VL-Reranker-2B at 4bd860ac4f15ad1897a214615cccc700f8f71818
scripts/qwen3_vl_reranker.py (Apache-2.0). Same prompt and yes-minus-no head;
changes: MPS FP16, bounded video frames/pixels, fail on decode errors, monitoring.
No filenames, human labels, or embedding scores enter the neural model.
"""
import gc
import math
import logging
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'Qwen/Qwen3-VL-Reranker-2B'
REVISION = '4bd860ac4f15ad1897a214615cccc700f8f71818'
SETTINGS = {'model': MODEL_ID, 'revision': REVISION, 'device': 'mps',
            'dtype': 'float16', 'frames': 16, 'max_total_pixels': 1048576,
            'instruction': 'Given a search query, retrieve relevant candidates that answer the query.'}


def search_with_reranking(actions, reranker, project, query, limit=20, enabled=False, candidates=10):
    with ResourceSampler() as resources:
        t = time.perf_counter()
        result = actions.search(project, query, candidates if enabled else limit)
        result['retrieval_seconds'] = time.perf_counter()-t
        result['rerank'] = {'requested': enabled, 'applied': False, 'settings': SETTINGS}
        if enabled:
            hits = result['video_results']['unmerged_windows']
            result['baseline_results'] = list(hits)
            try:
                t = time.perf_counter()
                actions.unload()
                switch_seconds = time.perf_counter()-t
                ranked, timing = reranker.rank(query, hits, actions.snapshot[0]['media'])
                result['video_results']['unmerged_windows'] = ranked
                result['rerank'].update(applied=True, candidate_count=len(hits), timing=timing,
                                       embedding_unload_seconds=switch_seconds)
            except Exception as exc:
                logging.exception('Reranker failed; returning explicitly labelled original results')
                reranker.unload()
                result['rerank']['error'] = '重排未完成，当前显示原始排序：'+str(exc)[:300]
    result['resources'] = resources.result
    return result


class ResourceSampler:
    """Sample the whole server process; overlapping RSS/MPS figures must not be added."""
    def snapshot(self):
        import psutil
        import torch
        return {'rss_bytes': psutil.Process().memory_info().rss,
                'mps_allocated_bytes': torch.mps.current_allocated_memory(),
                'mps_driver_bytes': torch.mps.driver_allocated_memory(),
                'system_available_bytes': psutil.virtual_memory().available,
                'system_swap_used_bytes': psutil.swap_memory().used}

    def __enter__(self):
        self.started = time.perf_counter()
        self.before = self.snapshot()
        self.peaks = dict(self.before)
        self.min_available = self.before['system_available_bytes']
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self.monitor, daemon=True)
        self.worker.start()
        return self

    def sample(self):
        current = self.snapshot()
        for key, value in current.items(): self.peaks[key] = max(self.peaks[key], value)
        self.min_available = min(self.min_available, current['system_available_bytes'])

    def monitor(self):
        while not self.stop.wait(.05): self.sample()

    def __exit__(self, *args):
        self.stop.set(); self.worker.join()
        self.sample()
        self.result = {'scope': 'current_process', 'sample_interval_seconds': .05,
                       'total_seconds': time.perf_counter()-self.started,
                       'before': self.before, 'after': self.snapshot(),
                       'sampled_peak': self.peaks, 'min_system_available_bytes': self.min_available}


def read_frames(media, segment):
    import cv2
    from PIL import Image
    path = Path(media['location']) / media['path']
    if not path.is_absolute(): path = ROOT/path
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count < 1:
            raise RuntimeError('重排候选片段无法解码')
        first = min(count-1, max(0, math.ceil(segment['ts']*fps)))
        last = min(count-1, max(first, math.ceil(segment['te']*fps)-1))
        frames = []
        for number in np.linspace(first, last, SETTINGS['frames']).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(number))
            ok, frame = cap.read()
            if not ok: raise RuntimeError(f'重排抽帧失败：{number/fps:.2f}s')
            # Bound host RAM too; preserve aspect ratio before Qwen smart resize.
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((1024, 1024))
            frames.append(image)
        return frames
    finally:
        cap.release()


def reorder_hits(hits, scores):
    if len(hits) != len(scores) or not all(math.isfinite(s) for s in scores):
        raise RuntimeError('重排分数无效，未应用重排')
    ranked = [{**hit, 'original_rank': i+1, 'embedding_score': hit['distance'],
               'rerank_score': float(score)} for i, (hit, score) in enumerate(zip(hits, scores))]
    # Original embedding score remains separate; it is never mixed with rerank scores.
    return sorted(ranked, key=lambda hit: (-hit['rerank_score'], hit['original_rank']))


class VideoReranker:
    def __init__(self):
        self.model = self.processor = self.score_weight = None

    def unload(self):
        if self.model is not None:
            import torch
            self.model = self.processor = self.score_weight = None
            gc.collect(); torch.mps.empty_cache()

    def load(self):
        import psutil
        import torch
        if psutil.virtual_memory().available < 2 * 1024**3:
            raise RuntimeError('可用内存不足 2 GiB，未执行重排')
        if self.model is not None: return 0.0
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        try:
            hf_hub_download(MODEL_ID, 'model.safetensors', revision=REVISION, local_files_only=True)
        except LocalEntryNotFoundError as exc:
            raise RuntimeError('重排模型尚未下载完整，请完成模型下载后重试') from exc
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        started = time.perf_counter()
        torch.set_num_threads(1)
        torch.mps.set_per_process_memory_fraction(.55)
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION,
            local_files_only=True, padding_side='left')
        lm = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, revision=REVISION,
            local_files_only=True, dtype=torch.float16, device_map={'': 'mps'})
        vocab = processor.tokenizer.get_vocab()
        with torch.no_grad():
            weight = (lm.lm_head.weight[vocab['yes']] - lm.lm_head.weight[vocab['no']]).detach().clone()
        self.model = lm.model.eval()
        self.processor, self.score_weight = processor, weight
        del lm
        torch.mps.synchronize()
        return time.perf_counter()-started

    def score(self, query, frames):
        import torch
        from qwen_vl_utils import process_vision_info
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text':
                'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'}]},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': '<Instruct>: '+SETTINGS['instruction']},
                {'type': 'text', 'text': '<Query>:'},
                {'type': 'text', 'text': query},
                {'type': 'text', 'text': '\n<Document>:'},
                {'type': 'video', 'video': frames, 'total_pixels': SETTINGS['max_total_pixels']}]}]
        # Preserve the official prompt/template; reject errors instead of scoring NULL.
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, kwargs = process_vision_info(messages, image_patch_size=16,
            return_video_kwargs=True, return_video_metadata=True)
        if videos is None: raise RuntimeError('重排没有收到视频帧')
        videos, metadata = zip(*videos)
        inputs = self.processor(text=[text], images=images, videos=list(videos),
            video_metadata=list(metadata), padding=True, truncation=False,
            do_resize=False, return_tensors='pt', **kwargs)
        if inputs['input_ids'].shape[-1] > 8192:
            raise RuntimeError('重排输入过长，未截断视频内容')
        inputs = inputs.to('mps')
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state[:, -1]
            # Binary yes/no logits as in official reference, with FP32 projection for stability.
            score = torch.sigmoid(hidden.float() @ self.score_weight.float()).item()
        torch.mps.synchronize()
        if not math.isfinite(score): raise RuntimeError('重排模型返回无效分数')
        return score

    def rank(self, query, hits, catalog):
        import psutil
        started = time.perf_counter()
        load_seconds = self.load()
        by_id = {m['id']: m for m in catalog}
        scores, details = [], []
        for hit in hits:
            if psutil.virtual_memory().available < 2 * 1024**3:
                raise RuntimeError('可用内存不足，重排已中止；未应用部分排序')
            t = time.perf_counter()
            frames = read_frames(by_id[hit['media_id']], hit)
            decoded = time.perf_counter()
            scores.append(self.score(query, frames))
            details.append({'media_id': hit['media_id'], 'ts': hit['ts'], 'te': hit['te'],
                            'decode_seconds': decoded-t, 'inference_seconds': time.perf_counter()-decoded})
        return reorder_hits(hits, scores), {'load_seconds': load_seconds,
            'total_seconds': time.perf_counter()-started, 'candidates': details}
