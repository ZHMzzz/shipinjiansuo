"""Named-person + scene search, aligned to the exact same video frame."""
from wise.api.common import TextQueryTerm
from wise.api.services.embedding import EmbeddingConfig
from wise.api.services.search.base import SearchOutput
from wise.api.services.search.face_text import fuse_face_text_frame_results
from wise.data_models import ModalityType


def person_scene_search(vector, scene, negative, threshold, *, face_model, text_model,
                        search, embedding, project, config):
    # Find all indexed face candidates, then apply an identity threshold BEFORE
    # scene ranking. RRF scores must never be compared with face similarity.
    count = search.search_indices[ModalityType.VIDEO][face_model].index.ntotal
    faces = search.search_with_feature(vector, modality_type=ModalityType.VIDEO,
        feature_extractor_id=face_model, start=0, end=count, filter_specs={})
    matches = [(vid, score, meta, ext) for vid, score, meta, ext in
               zip(faces.ids, faces.distances, faces.metadata, faces.ext_metadata)
               if score >= threshold and meta.timestamp is not None]
    if not matches:
        return SearchOutput(), {}
    pairs = list(dict.fromkeys((meta.media_id, meta.timestamp) for _, _, meta, _ in matches))
    mids, timestamps = zip(*pairs)
    ids = project.wise_project.get_vector_ids(list(mids), ModalityType.VIDEO,
                                             text_model, timestamps=list(timestamps))
    if not ids:
        return SearchOutput(), {}
    terms = []
    if scene: terms.append(TextQueryTerm(term_id='scene', is_negative=False, txt=scene))
    if negative: terms.append(TextQueryTerm(term_id='exclude', is_negative=True, txt=negative))
    features = embedding.embed(text_model, EmbeddingConfig(query_prefix='',
        text_queries_weight=config.text_queries_weight,
        negative_queries_weight=config.negative_queries_weight), terms)
    scenes = search.search_with_feature(features, modality_type=ModalityType.VIDEO,
        feature_extractor_id=text_model, start=0, end=len(ids), filter_specs={},
        vector_id_constraint=ids)
    scene_scores = {(meta.media_id, meta.timestamp): score
                    for score, meta in zip(scenes.distances, scenes.metadata)}
    items = [((meta.media_id, meta.timestamp), score, (vid, meta, ext))
             for vid, score, meta, ext in matches]
    fused = fuse_face_text_frame_results(items, list(scene_scores),
        k=60, face_weight=1, text_weight=2, start=0, end=100)
    result = SearchOutput()
    evidence = {}
    face_scores = {vid: score for vid, score, _, _ in matches}
    for (vid, meta, ext), score in fused:
        result.ids.append(vid); result.distances.append(score)
        result.metadata.append(meta); result.ext_metadata.append(ext)
        evidence[str(vid)] = {'face_similarity': float(face_scores[vid]),
                             'scene_similarity': float(scene_scores[(meta.media_id, meta.timestamp)])}
    return result, evidence
