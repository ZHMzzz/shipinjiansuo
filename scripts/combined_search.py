"""Identity-gated face hits ranked by their containing Qwen video windows."""
from wise.api.services.search.base import SearchOutput
from wise.api.services.search.face_text import fuse_face_text_frame_results
from wise.data_models import ModalityType


def person_scene_search(vector, scene, negative, threshold, *, face_model, search, actions, project):
    count = search.search_indices[ModalityType.VIDEO][face_model].index.ntotal
    faces = search.search_with_feature(vector, modality_type=ModalityType.VIDEO,
        feature_extractor_id=face_model, start=0, end=count, filter_specs={})
    matches = [(vid, score, meta, ext) for vid, score, meta, ext in
               zip(faces.ids, faces.distances, faces.metadata, faces.ext_metadata)
               if score >= threshold and meta.timestamp is not None]
    if not matches: return SearchOutput(), {}
    snapshot = actions.get_snapshot(project)
    if snapshot is None: raise ValueError('Qwen 片段索引尚未建立')
    meta, matrix, _ = snapshot
    query, _ = actions.query_vector(scene, negative=negative)
    scores = matrix @ query[0]
    segments = {}
    for segment, score in zip(meta['segments'], scores):
        segments.setdefault(segment['media_id'], []).append((segment, float(score)))
    items, scene_scores, evidence = [], {}, {}
    for vid, face_score, face, ext in matches:
        covering = [(s, score) for s, score in segments.get(face.media_id, [])
                    if s['ts'] <= face.timestamp < s['te']]
        if not covering: continue
        segment, score = max(covering, key=lambda pair: pair[1])
        key = (face.media_id, face.timestamp)
        items.append((key, face_score, (vid, face, ext)))
        scene_scores[key] = score
        evidence[str(vid)] = {'face_similarity':float(face_score), 'scene_similarity':score,
                             'scene_ts':segment['ts'], 'scene_te':segment['te']}
    fused = fuse_face_text_frame_results(items, sorted(scene_scores, key=scene_scores.get, reverse=True),
        k=60, face_weight=1, text_weight=2, start=0, end=100)
    result = SearchOutput()
    for (vid, face, ext), score in fused:
        result.ids.append(vid); result.distances.append(score)
        result.metadata.append(face); result.ext_metadata.append(ext)
    return result, evidence
