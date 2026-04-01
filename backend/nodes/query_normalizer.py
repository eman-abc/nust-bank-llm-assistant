from __future__ import annotations

import re
from typing import Any, Dict

from backend.state import AgentState


WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def _infer_intent(query: str) -> tuple[str, dict[str, str]]:
    lowered = query.lower()

    rate_terms = ("rate", "profit", "payout", "tenor", "%", "monthly", "annual", "yield")
    app_terms = ("app", "mpin", "beneficiar", "wallet", "top-up", "raast", "biometric", "forgot password")
    comparison_terms = ("compare", "difference", "better", "versus", "vs")

    if any(term in lowered for term in rate_terms):
        return "rate_lookup", {"sheet": "Rate Sheet"}
    if any(term in lowered for term in app_terms):
        return "mobile_app", {"sheet": "Mobile App"}
    if any(term in lowered for term in comparison_terms):
        return "comparison", {}
    return "general_faq", {}


def run_query_normalizer(state: AgentState) -> Dict[str, Any]:
    query = (state.get("scrubbed_query") or state.get("user_query") or "").strip()
    if not query:
        return {
            "normalized_query": "",
            "query_intent": "unknown",
            "metadata_filters": {},
        }

    normalized_query = _normalize_whitespace(query.replace("“", '"').replace("”", '"').replace("’", "'"))
    query_intent, metadata_filters = _infer_intent(normalized_query)
    return {
        "normalized_query": normalized_query,
        "query_intent": query_intent,
        "metadata_filters": metadata_filters,
    }
