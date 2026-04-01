from __future__ import annotations

from typing import Any, Dict

from backend.config import get_settings
from backend.services.reranker_service import rerank_candidates
from backend.state import AgentState


def run_reranker(state: AgentState) -> Dict[str, Any]:
    query = (state.get("normalized_query") or state.get("scrubbed_query") or "").strip()
    candidates = list(state.get("retrieval_candidates") or [])
    if not query or not candidates:
        return {"reranked_candidates": [], "retrieval_confidence": 0.0}

    reranked = rerank_candidates(query, candidates)
    top_candidates = reranked[: get_settings().rerank_top_k]
    top_score = top_candidates[0].get("rerank_score", 0.0) if top_candidates else 0.0
    return {
        "reranked_candidates": top_candidates,
        "retrieval_confidence": float(top_score),
    }
