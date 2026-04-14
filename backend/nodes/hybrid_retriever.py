from __future__ import annotations

import logging
from typing import Any, Dict

from backend.config import get_settings
from backend.services.embedding_service import embed_text
from backend.services.qdrant_store import hybrid_search
from backend.services.sparse_encoder import encode_sparse_text
from backend.state import AgentState

logger = logging.getLogger(__name__)


def run_hybrid_retriever(state: AgentState) -> Dict[str, Any]:
    query = (state.get("normalized_query") or state.get("scrubbed_query") or "").strip()
    if not query:
        return {"retrieval_candidates": [], "retrieval_confidence": 0.0}

    try:
        settings = get_settings()
        dense_vector = embed_text(query)
        sparse_vector = encode_sparse_text(query)
        candidates = hybrid_search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=max(settings.dense_retrieval_limit, settings.sparse_retrieval_limit),
            metadata_filters=state.get("metadata_filters"),
        )
        
        # Apply strict relevance filter
        threshold = settings.retrieval_confidence_threshold
        filtered_candidates = [c for c in candidates if c.get("score", 0.0) >= threshold]
        
        top_score = filtered_candidates[0]["score"] if filtered_candidates else 0.0
        return {
            "retrieval_candidates": filtered_candidates,
            "retrieval_confidence": float(max(top_score, 0.0)),
        }
    except Exception as exc:
        logger.exception("Hybrid retrieval failure: %s", exc)
        return {"retrieval_candidates": [], "retrieval_confidence": 0.0}
