"""Query-time frame comparison; reuse Qwen without adding an ingestion index."""
from collections import OrderedDict
from pathlib import Path
import time
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def read_frame(media, timestamp):
    path = Path(media['location'])/media['path']
    if not path.is_absolute(): path=ROOT/path
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count < 1: raise RuntimeError('无法读取候选视频画面')
        number = min(count-1,max(0,round(timestamp*fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES,number)
        ok,frame = capture.read()
        if not ok: raise RuntimeError('候选画面解码失败')
        return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)), number/fps
    finally:
        capture.release()


class ObjectRefiner:
    def __init__(self):
        self.frames = OrderedDict()

    def refine(self, engine, project, result, image, negative='', limit=20):
        started = time.perf_counter()
        snapshot = engine.get_snapshot(project)
        if snapshot is None: raise ValueError('Qwen 片段索引尚未建立')
        catalog = {m['id']:m for m in snapshot[0]['media']}
        query,_ = engine.query_vector(image=image,negative=negative)
        hits = result['video_results']['unmerged_windows'][:20]
        refined, cached = [], 0
        for rank, hit in enumerate(hits,1):
            timestamp = (hit['ts']+hit['te'])/2
            key = (str(Path(project).resolve()),hit['media_id'],timestamp)
            if key in self.frames:
                vector,actual = self.frames[key]
                self.frames.move_to_end(key);cached += 1
            else:
                frame,actual = read_frame(catalog[hit['media_id']],timestamp)
                vector,_ = engine.query_vector(image=frame)
                self.frames[key] = vector,actual
                if len(self.frames)>256: self.frames.popitem(last=False)
            score = float(vector[0]@query[0])
            refined.append({**hit,'original_rank':rank,'embedding_score':hit['distance'],
                'distance':score,'frame_score':score,'thumbnail_ts':actual,
                'thumbnail':f'/demo/thumbnail?media_id={hit["media_id"]}&timestamp={actual}'})
        refined.sort(key=lambda h:h['distance'],reverse=True)
        return {**result, 'video_results':{**result['video_results'],'unmerged_windows':refined[:limit]},
            'object_refinement':{'applied':True,'candidate_count':len(hits),'cached_frames':cached,
                'seconds':time.perf_counter()-started,'frames_per_candidate':1}}
