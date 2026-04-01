from __future__ import annotations

import re
from typing import Any, Dict

from backend.config import get_settings
from backend.state import AgentState


NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
FALLBACK_RESPONSE = (
    "I don't have enough verified information in the bank's records to answer that confidently. "
    "Please contact an official branch or support channel for confirmation."
)


def _numeric_consistency(answer: str, context: str) -> bool:
    answer_numbers = set(NUMBER_PATTERN.findall(answer))
    if not answer_numbers:
        return True
    context_numbers = set(NUMBER_PATTERN.findall(context))
    return answer_numbers.issubset(context_numbers)


def run_grounding_checker(state: AgentState) -> Dict[str, Any]:
    selected_context = (state.get("selected_context") or "").strip()
    final_response = (state.get("final_response") or "").strip()
    citations = list(state.get("citations") or [])
    retrieval_confidence = float(state.get("retrieval_confidence") or 0.0)
    query_intent = state.get("query_intent", "general_faq")

    if not selected_context or not citations:
        return {
            "grounding_passed": False,
            "final_response": FALLBACK_RESPONSE,
        }

    if retrieval_confidence < get_settings().retrieval_confidence_threshold:
        return {
            "grounding_passed": False,
            "final_response": FALLBACK_RESPONSE,
        }

    if query_intent == "rate_lookup" and not _numeric_consistency(final_response, selected_context):
        return {
            "grounding_passed": False,
            "final_response": FALLBACK_RESPONSE,
        }

    return {"grounding_passed": True}
