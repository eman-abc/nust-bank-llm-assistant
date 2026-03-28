"""Shared LangGraph agent state."""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Lifecycle fields for a single user turn through the graph."""

    user_query: str
    is_safe: bool
    scrubbed_query: str
    retrieved_context: str
    final_response: str
