"""Library query routing: native OWLv2, Qwen text, and identity constraints."""
from bisect import bisect_left
from collections import defaultdict
import sqlite3
import numpy as np
from fastapi import HTTPException
from wise.api.common import MediaQueryTerm
from wise.api.services.embedding import EmbeddingConfig
from wise.api.standalone.search import search_output_to_response
from wise.data_models import MediaType, ModalityType
if __package__:
    from .qwen_project import OWL, FACE
    from .combined_search import person_scene_search
    from .semantic_filter import filter_semantic_hits
else:
    from qwen_project import OWL, FACE
    from combined_search import person_scene_search
    from semantic_filter import filter_semantic_hits


def matching_media(catalog, filename):
    keyword = filename.strip().casefold()
    return {int(v['id']) for v in catalog if keyword in v['name'].casefold()}


def object_constraints(db_path, allowed, face_times=None):
    """Constrain before ranking; a face must fall inside the object's 4s window."""
    if not allowed: return np.array([], dtype='int64')
    times = {mid: sorted(values) for mid, values in (face_times or {}).items()}
    ids = []
    with sqlite3.connect(db_path) as db:
        rows = db.execute('SELECT id,media_id,timestamp FROM vectors WHERE feature_extractor_id=?', (OWL,))
        for vid, mid, ts in rows:
            if mid not in allowed: continue
            if face_times is not None:
                values = times.get(mid, [])
                if ts is None: continue
                i = bisect_left(values, ts)
                if i == len(values) or values[i] >= ts + 4: continue
            ids.append(vid)
    return np.asarray(ids, dtype='int64')


def normalize(data, catalog, mode, limit, threshold):
    by_id = {str(v['id']): v for v in catalog}
    hits = (data.get('video_results') or {}).get('unmerged_windows', [])
    if mode in ('text', 'person_scene'):
        hits = filter_semantic_hits(hits,
            score_key='scene_similarity' if mode == 'person_scene' else 'distance')
    items, seen = [], set()
    for hit in hits:
        mid = str(hit['media_id'])
        if mid not in by_id: continue
        if mode == 'face' and hit['distance'] < threshold: continue
        if mid in seen: continue
        seen.add(mid)
        v = by_id[mid]
        # Qwen results are windows; WISE face/object results are frame hits.
        start = hit['ts'] if mode == 'text' else hit.get('thumbnail_ts', hit['ts'])
        items.append({**v, 'thumbnail':hit['thumbnail'], 'start':start,
            'end':min(hit['te'], v['duration']), 'score':hit['distance'], 'kind':mode,
            'face_similarity':hit.get('face_similarity'),
            'scene_ts':hit.get('scene_ts'), 'scene_te':hit.get('scene_te')})
        if len(items) >= limit: break
    return {'items':items, 'total':len(items), 'mode':mode,
        'limited':len(hits) > limit if mode == 'text' else len(hits) >= limit}


def search_library(*, catalog, query, person, filename, image_bytes, threshold,
                   limit, project, search, embedding, actions, config, people_db):
    if query and image_bytes is not None:
        raise HTTPException(400, '自然语言与参考图片请分开查询')
    allowed = matching_media(catalog, filename)
    vector = None
    if person:
        with sqlite3.connect(people_db) as db:
            row = db.execute('SELECT vector,model FROM people WHERE name=?', (person,)).fetchone()
        if row is None: raise HTTPException(404, '未登记此姓名，请先在人脸维护中登记')
        if row[1] != FACE: raise HTTPException(409, '人脸模型已改变，请重新登记')
        vector = np.frombuffer(row[0], dtype=np.float32).reshape(1,-1).copy()
    selected = [v for v in catalog if int(v['id']) in allowed]
    if not query and not person and image_bytes is None:
        return {'items':selected,'total':len(selected),'mode':'browse','limited':False}
    if not allowed: return {'items':[],'total':0,'mode':'empty','limited':False}
    if person and query:
        result, evidence = person_scene_search(vector, query, threshold, face_model=FACE,
            search=search, actions=actions, project=project.wise_project._project_dir, allowed_media=allowed)
        data = search_output_to_response(config,project,[],MediaType.VIDEO,result,1).model_dump(mode='json',exclude={'query'})
        for hit in (data.get('video_results') or {}).get('unmerged_windows',[]):
            hit.update(evidence.get(str(hit['vector_id']),{}))
        return normalize(data,selected,'person_scene',limit,threshold)
    if query:
        # Select videos before truncation; one extra video detects a real cap.
        data = actions.search(project.wise_project._project_dir,query,limit+1,
            media_ids=allowed,group_by_video=True)
        return normalize(data,selected,'text',limit,threshold)
    face_times = None
    if person:
        count = search.search_indices[ModalityType.VIDEO][FACE].index.ntotal
        faces = search.search_with_feature(vector,modality_type=ModalityType.VIDEO,
            feature_extractor_id=FACE,start=0,end=count,filter_specs={})
        face_times = defaultdict(list)
        for score, meta in zip(faces.distances,faces.metadata):
            if meta.media_id in allowed and score >= threshold and meta.timestamp is not None:
                face_times[meta.media_id].append(meta.timestamp)
        if image_bytes is None:
            data=search_output_to_response(config,project,[],MediaType.VIDEO,faces,1).model_dump(mode='json',exclude={'query'})
            return normalize(data,selected,'face',limit,threshold)
        allowed &= set(face_times)
        if not allowed: return {'items':[],'total':0,'mode':'person_object','limited':False}
    terms = [MediaQueryTerm(term_id='image',is_negative=False,src=image_bytes,qtype='visual')]
    constraint = None
    if filename or person:
        constraint = object_constraints(project.wise_project._project_dir/'metadata/internal.db',allowed,face_times)
    result = search.search(terms,EmbeddingConfig(query_prefix='',text_queries_weight=config.text_queries_weight,
        negative_queries_weight=config.negative_queries_weight),modality_type=ModalityType.VIDEO,
        feature_extractor_id=OWL,start=0,end=limit,filter_specs={},vector_id_constraint=constraint)
    data=search_output_to_response(config,project,terms,MediaType.VIDEO,result,1).model_dump(mode='json',exclude={'query'})
    return normalize(data,selected,'person_object' if person else 'object',limit,threshold)
