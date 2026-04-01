from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any, Sequence

from backend.config import get_settings
from backend.services.sparse_encoder import overlap_score, tokenize

logger = logging.getLogger(__name__)

_reranker_load_failed = False


@lru_cache(maxsize=1)
def get_reranker_model():
    global _reranker_load_failed
    if _reranker_load_failed:
        return None

    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(get_settings().reranker_model)
    except Exception as exc:
        logger.warning("Falling back to lexical reranking because CrossEncoder failed to load: %s", exc)
        _reranker_load_failed = True
        return None


def _normalize_score(raw_score: float) -> float:
    return 1.0 / (1.0 + math.exp(-raw_score))


def rerank_candidates(query: str, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []

    model = get_reranker_model()
    reranked: list[dict[str, Any]] = []

    if model is not None:
        pairs = [(query, (candidate.get("payload", {}).get("chunk_text") or "")) for candidate in candidates]
        scores = model.predict(pairs)
        for candidate, raw_score in zip(candidates, scores, strict=False):
            enriched = dict(candidate)
            enriched["rerank_score"] = _normalize_score(float(raw_score))
            reranked.append(enriched)
    else:
        query_tokens = tokenize(query)
        for candidate in candidates:
            chunk_text = candidate.get("payload", {}).get("chunk_text") or ""
            lexical = overlap_score(query_tokens, chunk_text)
            fused = candidate.get("score", 0.0)
            rerank_score = max(0.0, min(1.0, (0.65 * lexical) + (0.35 * max(fused, 0.0))))
            enriched = dict(candidate)
            enriched["rerank_score"] = rerank_score
            reranked.append(enriched)

    return sorted(reranked, key=lambda candidate: candidate.get("rerank_score", 0.0), reverse=True)
