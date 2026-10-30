"""Identity-gated face hits ranked by their containing Qwen video windows."""
from wise.api.services.search.base import SearchOutput
from wise.api.services.search.face_text import fuse_face_text_frame_results
if __package__:
    from .face_search import search_registered_faces
else:
    from face_search import search_registered_faces


def person_scene_search(vector, scene, threshold, *, face_model, search, actions, project, allowed_media=None, query_image=None):
    faces = search_registered_faces(vector, search=search, face_model=face_model, threshold=threshold)
    matches = [(vid, score, meta, ext) for vid, score, meta, ext in
               zip(faces.ids, faces.distances, faces.metadata, faces.ext_metadata)
               if score >= threshold and meta.timestamp is not None
               and (allowed_media is None or meta.media_id in allowed_media)]
    if not matches: return SearchOutput(), {}
    snapshot = actions.get_snapshot(project)
    if snapshot is None: raise ValueError('Qwen 片段索引尚未建立')
    meta, matrix, _ = snapshot
    query, _ = (actions.query_vector(scene, image=query_image) if query_image is not None
                else actions.query_vector(scene))
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
