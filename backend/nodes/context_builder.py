from __future__ import annotations

from typing import Any, Dict

from backend.config import get_settings
from backend.state import AgentState


def run_context_builder(state: AgentState) -> Dict[str, Any]:
    reranked_candidates = list(state.get("reranked_candidates") or [])
    if not reranked_candidates:
        return {
            "selected_context": "",
            "retrieved_context": "",
            "citations": [],
        }

    selected = reranked_candidates[: get_settings().final_context_k]
    context_parts = []
    citations = []

    for candidate in selected:
        payload = candidate.get("payload", {})
        chunk_text = (payload.get("chunk_text") or "").strip()
        citation = {
            "doc_id": payload.get("doc_id", candidate.get("id", "")),
            "source_file": payload.get("source_file", ""),
            "topic": payload.get("topic", ""),
            "sheet": payload.get("sheet", ""),
            "chunk_index": payload.get("chunk_index", -1),
            "score": candidate.get("rerank_score", candidate.get("score", 0.0)),
        }
        if chunk_text:
            context_parts.append(
                f"[Citation: {citation['doc_id']} | Topic: {citation['topic']} | Source: {citation['source_file']}]\n"
                f"{chunk_text}"
            )
            citations.append(citation)

    selected_context = "\n\n".join(context_parts)
    return {
        "selected_context": selected_context,
        "retrieved_context": selected_context,
        "citations": citations,
    }
