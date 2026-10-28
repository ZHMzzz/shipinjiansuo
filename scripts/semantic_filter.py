"""Conservative Qwen 2B score filtering, calibrated on the local video library.

Scores are similarities, not probabilities. See experiments/relevance-filter.
Do not apply these constants to OWLv2 scores or fused identity ranking scores.
"""
import math

MIN_BEST_SCORE = 0.52
MIN_RESULT_SCORE = 0.45
MAX_SCORE_GAP = 0.08


def filter_semantic_hits(hits, *, score_key='distance'):
    valid = []
    for hit in hits:
        score = hit.get(score_key)
        if score is not None and math.isfinite(score):
            valid.append(hit)
    best = max((h[score_key] for h in valid), default=-math.inf)
    if best < MIN_BEST_SCORE:
        return []
    cutoff = max(MIN_RESULT_SCORE, best - MAX_SCORE_GAP)
    return [h for h in valid if h[score_key] >= cutoff]
