"""Union enrollment queries by indexed face ID, retaining the strongest score."""
import math

import numpy as np
from wise.api.services.search.base import SearchOutput
from wise.data_models import ModalityType


def search_registered_faces(vectors, *, search, face_model, threshold=None, limit=None):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError('登记的人脸特征无效')
    count = search.search_indices[ModalityType.VIDEO][face_model].index.ntotal
    if not count:
        return SearchOutput()
    best = {}
    # WISE's search_with_feature consumes only row zero, so query photos separately.
    for vector in vectors:
        hits = search.search_with_feature(vector[None, :], modality_type=ModalityType.VIDEO,
            feature_extractor_id=face_model, start=0, end=count, filter_specs={})
        for vid, score, meta, ext in zip(hits.ids, hits.distances, hits.metadata, hits.ext_metadata):
            if vid is None or not math.isfinite(score) or (threshold is not None and score < threshold):
                continue
            if vid not in best or score > best[vid][0]:
                best[vid] = (score, meta, ext)
    ranked = sorted(best.items(), key=lambda entry: entry[1][0], reverse=True)
    if limit is not None:
        ranked = ranked[:limit]
    result = SearchOutput()
    for vid, (score, meta, ext) in ranked:
        result.ids.append(vid)
        result.distances.append(score)
        result.metadata.append(meta)
        result.ext_metadata.append(ext)
    return result
