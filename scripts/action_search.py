"""Versioned Qwen segment index. Only pixels and query text reach the model."""
import json
import math
import sqlite3
import time
import gc
import hashlib
from collections import OrderedDict
from pathlib import Path

import faiss
import numpy as np

SETTINGS = {'schema': 1, 'model': 'hf/Qwen/Qwen3-VL-Embedding/2B',
            'dtype': 'float16', 'max_total_pixels': 1048576,
            'window_seconds': 8.0, 'stride_seconds': 4.0, 'frames': 16}
ROOT = Path(__file__).resolve().parents[1]


def windows(duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('视频时长无效')
    # Stop once a window reaches the end: no redundant tiny tail window.
    start = 0.0
    while True:
        end = min(start + SETTINGS['window_seconds'], duration)
        yield round(start, 6), round(end, 6)
        if end >= duration: break
        start += SETTINGS['stride_seconds']


def media_catalog(project):
    with sqlite3.connect(Path(project)/'metadata/internal.db') as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("""SELECT m.id, m.path, m.duration, s.location
            FROM media m JOIN source_collections s ON m.source_collection_id=s.id
            WHERE m.media_type IN ('VIDEO','AV') ORDER BY m.id""").fetchall()
    return [dict(r) for r in rows]


def plan(catalog):
    return [{'media_id': m['id'], 'ts': start, 'te': end}
            for m in catalog for start, end in windows(float(m['duration']))]


def read_snapshot(project, complete=False):
    folder = Path(project)/'action-index'
    if not folder.exists():
        if complete: raise ValueError('动作索引尚未建立')
        return None
    meta = json.loads((folder/'metadata.json').read_text())
    if meta['settings'] != SETTINGS: raise ValueError('动作索引配置不匹配，需要重新建立')
    vectors = np.load(folder/'vectors.npy', allow_pickle=False)
    index = faiss.read_index(str(folder/'index.faiss'))
    if vectors.ndim != 2 or vectors.shape[0] != len(meta['segments']) or not np.isfinite(vectors).all():
        raise ValueError('动作向量与片段记录不一致')
    if index.ntotal != len(vectors) or index.d != vectors.shape[1]:
        raise ValueError('动作索引数量或维度不一致')
    if meta['segments'] != plan(meta['media']):
        raise ValueError('动作片段时间覆盖不完整')
    current = media_catalog(project)
    known = {m['id']: m for m in current}
    if any(known.get(m['id']) != m for m in meta['media']):
        raise ValueError('动作索引媒体与视频库不一致')
    if complete and current != meta['media']:
        raise ValueError('尚有视频缺少动作索引，未发布')
    return meta, vectors, index


def select_hits(segments, scores, ids, limit=20):
    """Suppress near-identical overlapping windows; retain separate events/video."""
    selected = []
    for score, index in zip(scores, ids):
        if index < 0: continue
        item = {**segments[int(index)], 'distance': float(score)}
        duplicate = False
        for other in selected:
            if item['media_id'] != other['media_id']: continue
            overlap = max(0, min(item['te'], other['te']) - max(item['ts'], other['ts']))
            union = max(item['te'], other['te']) - min(item['ts'], other['ts'])
            if overlap / max(union, 1e-6) >= .5:
                duplicate = True
                break
        if not duplicate: selected.append(item)
        if len(selected) >= limit: break
    return selected


class ActionEngine:
    """Main event-loop gate serializes calls with the other online models."""
    def __init__(self):
        self.extractor = None
        self.cached_project = None
        self.snapshot = None
        self.last_load_seconds = None
        self.query_vectors = OrderedDict()

    def unload(self):
        if self.extractor is not None:
            import torch
            self.extractor = None
            gc.collect()
            torch.mps.empty_cache()

    def model(self):
        import psutil
        import torch
        if psutil.virtual_memory().available < 2 * 1024**3:
            raise RuntimeError('当前可用内存不足 2GiB，请关闭部分应用后重试动作处理')
        if self.extractor is None:
            from wise.feature.qwen3vl import Qwen3VLEmbeddingFeatureExtractor as Extractor
            started = time.perf_counter()
            torch.set_num_threads(1)
            torch.mps.set_per_process_memory_fraction(.55)
            extractor = Extractor(SETTINGS['model'], device='mps', warmup=False,
                config=Extractor.Config(device='mps', compile=False, warmup=False,
                    dtype=SETTINGS['dtype'], max_total_pixels=SETTINGS['max_total_pixels']))
            _ = extractor._model
            self.extractor = extractor
            self.last_load_seconds = time.perf_counter() - started
        return self.extractor

    def encode_segment(self, media, segment):
        import cv2
        import torch
        model = self.model()
        path = Path(media['location'])/media['path']
        if not path.is_absolute(): path = ROOT/path
        capture = cv2.VideoCapture(str(path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or count < 1:
                raise RuntimeError(f"无法解码动作片段：{media['path']}")
            first = min(count-1, max(0, math.ceil(segment['ts']*fps)))
            last = min(count-1, max(first, math.ceil(segment['te']*fps)-1))
            frames = []
            for number in np.linspace(first, last, SETTINGS['frames']).astype(int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(number))
                ok, frame = capture.read()
                if not ok: raise RuntimeError(f"动作抽帧失败：{media['path']} @ {number/fps:.2f}s")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
            vector = model.extract_video_segment_features(tensor).vectors[0]
            torch.mps.synchronize()
            if not np.isfinite(vector).all(): raise RuntimeError('动作模型返回无效向量')
            return vector.astype('float32')
        finally:
            capture.release()

    def prepare(self, project):
        catalog = media_catalog(project)
        previous = read_snapshot(project)
        old_media, records, vectors = [], [], []
        if previous:
            old_media = previous[0]['media']
            records = list(previous[0]['segments'])
            vectors = list(previous[1])
        completed = {m['id'] for m in old_media}
        missing = [m for m in catalog if m['id'] not in completed]
        return catalog, records, vectors, missing

    def finish(self, project, catalog, records, vectors):
        # Keep metadata/vector positions aligned even if database IDs have gaps.
        order = sorted(range(len(records)), key=lambda i: (records[i]['media_id'], records[i]['ts']))
        records = [records[i] for i in order]
        matrix = np.asarray([vectors[i] for i in order], dtype='float32')
        if not records: raise ValueError('没有可建立动作索引的视频')
        index = faiss.IndexFlatIP(matrix.shape[1]); index.add(matrix)
        folder = Path(project)/'action-index'; folder.mkdir(exist_ok=True)
        np.save(folder/'vectors.npy', matrix)
        faiss.write_index(index, str(folder/'index.faiss'))
        (folder/'metadata.json').write_text(json.dumps({'settings': SETTINGS, 'media': catalog,
            'segments': records, 'built_at': time.time()}, ensure_ascii=False, indent=2))
        read_snapshot(project, complete=True)

    def get_snapshot(self, project):
        project = Path(project).resolve()
        if project != self.cached_project:
            snapshot = read_snapshot(project)
            self.snapshot, self.cached_project = snapshot, project
        return self.snapshot

    def query_vector(self, query='', image=None):
        query = query.strip()
        if not query and image is None:
            raise ValueError('请输入描述或提供参考图片')
        digest = None if image is None else hashlib.sha256(
            str((image.mode, image.size)).encode()+image.tobytes()).hexdigest()
        key = query if image is None else (query, digest)
        cached = key in self.query_vectors
        if cached:
            vector = self.query_vectors[key]
            self.query_vectors.move_to_end(key)
        else:
            model = self.model()
            vector = (model.extract_query_features(image, query) if image is not None else
                      model.extract_text_features([query]) if query else None)
            vector = np.asarray(vector, dtype='float32')
            norm = np.linalg.norm(vector, axis=-1, keepdims=True)
            if not np.isfinite(vector).all() or np.any(norm < 1e-8):
                raise RuntimeError('查询向量无效，请重试')
            vector = vector / norm
            self.query_vectors[key] = vector
            if len(self.query_vectors) > 64: self.query_vectors.popitem(last=False)
        return vector, cached

    def search(self, project, query, limit, *, image=None, unique_videos=False):
        snapshot = self.get_snapshot(project)
        if snapshot is None: raise ValueError('Qwen 片段索引正在补建，请查看视频库中的索引进度')
        meta, vectors, index = snapshot
        started = time.perf_counter()
        vector, cached = self.query_vector(query, image)
        encoded = time.perf_counter()
        # Search every segment for video-level ranking: a long source video must
        # not consume the entire shortlist and hide other source videos.
        scores, ids = index.search(vector, index.ntotal if unique_videos else min(index.ntotal, max(limit*5, 100)))
        if unique_videos:
            hits, seen = [], set()
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0: continue
                item = meta['segments'][int(idx)]
                if item['media_id'] in seen: continue
                seen.add(item['media_id']); hits.append({**item, 'distance':float(score)})
                if len(hits) >= limit: break
        else:
            hits = select_hits(meta['segments'], scores[0], ids[0], limit)
        videos = {m['id']: {'filename': m['path'], 'duration': m['duration']} for m in meta['media']}
        for hit in hits:
            mid = hit['media_id']; preview = (hit['ts']+hit['te'])/2
            hit.update(thumbnail_ts=preview, thumbnail=f'/demo/thumbnail?media_id={mid}&timestamp={preview}',
                       link=f'/demo/media/{mid}#t={hit["ts"]},{hit["te"]}')
        return {'score_kind': 'segment' if unique_videos else 'action', 'query': query,
                'query_kind': 'image_text' if image is not None and query else 'image' if image is not None else 'text',
                'settings': SETTINGS,
                'query_embedding_cached': cached,
                'encode_seconds': encoded-started, 'search_seconds': time.perf_counter()-encoded,
                'indexed_videos': len(meta['media']), 'indexed_segments': len(meta['segments']),
                'video_results': {'videos': videos, 'unmerged_windows': hits}}
