"""Shared LangGraph agent state."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Lifecycle fields for a single user turn through the graph."""

    user_query: str
    is_safe: bool
    scrubbed_query: str
    normalized_query: str
    query_intent: str
    metadata_filters: dict[str, str]
    retrieval_candidates: list[dict[str, Any]]
    reranked_candidates: list[dict[str, Any]]
    selected_context: str
    citations: list[dict[str, Any]]
    retrieval_confidence: float
    grounding_passed: bool
    retrieved_context: str
    final_response: str
