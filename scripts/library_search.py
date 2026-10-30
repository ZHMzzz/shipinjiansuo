"""Library query routing: native OWLv2, Qwen text/joint queries, and identity constraints."""
from bisect import bisect_left
from collections import defaultdict
import sqlite3
import numpy as np
from fastapi import HTTPException
from wise.api.common import MediaQueryTerm, TextQueryTerm, VectorQueryTerm
from wise.api.services.embedding import EmbeddingConfig
from wise.api.standalone.search import search_output_to_response
from wise.data_models import MediaType, ModalityType
if __package__:
    from .qwen_project import OWL, FACE
    from .combined_search import person_scene_search
    from .semantic_filter import filter_semantic_hits
    from .native_qwen import QWEN
    from .query_image import decode_query_image
    from .people_store import load_person_vectors
    from .face_search import search_registered_faces
else:
    from qwen_project import OWL, FACE
    from combined_search import person_scene_search
    from semantic_filter import filter_semantic_hits
    from native_qwen import QWEN
    from query_image import decode_query_image
    from people_store import load_person_vectors
    from face_search import search_registered_faces


# Regional candidates are separate from the API's final video limit. Match the
# upstream WISE frontend's 500-hit request budget; repeated regions must not
# consume the video's display slots before normalize() groups them.
OBJECT_CANDIDATE_LIMIT = 500


def native_segment_search(project, search, config, query, limit, allowed=None, *, image=None, actions=None):
    """WISE embedding, top-k and segment response; no relevance post-filter."""
    terms = [TextQueryTerm(term_id='text', is_negative=False, txt=query)]
    constraint = None
    if QWEN not in search.search_indices.get(ModalityType.VIDEO, {}):
        raise HTTPException(409, 'Qwen 片段索引尚未就绪，请等待索引建立完成后重试')
    count = search.search_indices[ModalityType.VIDEO][QWEN].index.ntotal
    if allowed is not None:
        with sqlite3.connect(project.wise_project._project_dir/'metadata/internal.db') as db:
            constraint = np.asarray([vid for vid, mid in db.execute(
                'SELECT id,media_id FROM vectors WHERE feature_extractor_id=?', (QWEN,))
                if mid in allowed], dtype='int64')
        count = len(constraint)
    end = min(limit, config.max_search_results, count)
    if end == 0:
        return {'video_results':None}, False
    if image is not None:
        vector, _ = actions.query_vector(query, image=image)
        terms = [VectorQueryTerm(term_id='image_text', is_negative=False, vector=vector)]
    result = search.search(terms, EmbeddingConfig(query_prefix='',
        text_queries_weight=config.text_queries_weight, negative_queries_weight=config.negative_queries_weight),
        modality_type=ModalityType.VIDEO, feature_extractor_id=QWEN,
        start=0, end=end, filter_specs={}, vector_id_constraint=constraint)
    data = search_output_to_response(config, project, terms, MediaType.VIDEO, result, 1)
    return data.model_dump(mode='json', exclude={'query'}), count > end


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
    hits = (data.get('video_results') or {}).get('merged_windows' if mode in ('text', 'image_text') else 'unmerged_windows', [])
    if mode in ('person_scene', 'person_image_text'):
        hits = filter_semantic_hits(hits, score_key='scene_similarity')
    items, seen = [], set()
    for hit in hits:
        mid = str(hit['media_id'])
        if mid not in by_id: continue
        if mode == 'face' and hit['distance'] < threshold: continue
        if mid in seen: continue
        seen.add(mid)
        v = by_id[mid]
        # WISE merged segments use the highest-scoring hit as representative.
        start = hit.get('thumbnail_ts', hit['ts'])
        items.append({**v, 'thumbnail':hit['thumbnail'], 'start':start,
            'end':min(hit['te'], v['duration']), 'score':hit['distance'], 'kind':mode,
            'face_similarity':hit.get('face_similarity'),
            'scene_ts':hit.get('scene_ts'), 'scene_te':hit.get('scene_te')})
        if len(items) >= limit: break
    return {'items':items, 'total':len(items), 'mode':mode,
        'limited':len(hits) > limit if mode == 'text' else len(hits) >= limit}


def search_library(*, catalog, query, person, filename, image_bytes, threshold,
                   limit, project, search, embedding, actions, config, people_db):
    allowed = matching_media(catalog, filename)
    vector = None
    if person:
        vector = load_person_vectors(people_db, person, FACE)
    selected = [v for v in catalog if int(v['id']) in allowed]
    if not query and not person and image_bytes is None:
        return {'items':selected,'total':len(selected),'mode':'browse','limited':False}
    if not allowed: return {'items':[],'total':0,'mode':'empty','limited':False}
    image = decode_query_image(image_bytes) if query and image_bytes is not None else None
    if person and query:
        image_args = {'query_image':image} if image is not None else {}
        result, evidence = person_scene_search(vector, query, threshold, face_model=FACE,
            search=search, actions=actions, project=project.wise_project._project_dir, allowed_media=allowed, **image_args)
        data = search_output_to_response(config,project,[],MediaType.VIDEO,result,1).model_dump(mode='json',exclude={'query'})
        for hit in (data.get('video_results') or {}).get('unmerged_windows',[]):
            hit.update(evidence.get(str(hit['vector_id']),{}))
        return normalize(data,selected,'person_image_text' if image is not None else 'person_scene',limit,threshold)
    if query:
        image_args = {'image':image, 'actions':actions} if image is not None else {}
        data, limited = native_segment_search(project, search, config, query, limit,
            allowed=allowed if filename else None, **image_args)
        response = normalize(data, selected, 'image_text' if image is not None else 'text', limit, threshold)
        response['limited'] = limited
        return response
    face_times = None
    if person:
        faces = search_registered_faces(vector, search=search, face_model=FACE, threshold=threshold)
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
        feature_extractor_id=OWL,start=0,end=OBJECT_CANDIDATE_LIMIT,filter_specs={},vector_id_constraint=constraint)
    data=search_output_to_response(config,project,terms,MediaType.VIDEO,result,1).model_dump(mode='json',exclude={'query'})
    response = normalize(data,selected,'person_object' if person else 'object',limit,threshold)
    hits = (data.get('video_results') or {}).get('unmerged_windows', [])
    selected_ids = {str(v['id']) for v in selected}
    matched_ids = {str(hit['media_id']) for hit in hits} & selected_ids
    # The candidate pool may be truncated, or the requested video limit may
    # exclude distinct videos. Duplicate regions alone do not imply truncation.
    response['limited'] = len(hits) >= OBJECT_CANDIDATE_LIMIT or len(matched_ids) > limit
    return response
