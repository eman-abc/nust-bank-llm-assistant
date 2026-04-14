from __future__ import annotations

import logging

from transformers import pipeline

logger = logging.getLogger(__name__)

_injection_classifier = None


def _get_injection_classifier():
    global _injection_classifier
    if _injection_classifier is not None:
        return _injection_classifier

    try:
        logger.info("Loading DeBERTa guardrail model...")
        _injection_classifier = pipeline(
            "text-classification",
            model="ProtectAI/deberta-v3-base-injection",
        )
    except Exception as exc:
        logger.error("Failed to load guardrail model: %s", exc)
        _injection_classifier = False
    return _injection_classifier


def guardrail_node(state: dict):
    """
    Semantic firewall that checks for prompt injection before the main evaluator.
    """
    # 1. Broad Pattern Recognition
    query = (state.get("user_query") or "").lower()
    
    blocked_keywords = [
        "ignore previous", "system prompt", "dan mode", "step by step", 
        "training data", "chat logs", "customer records", "agent interactions",
        "sample interaction", "reveal your instructions", "interactions", "logs", "example conversation"
    ]
    
    if any(k in query for k in blocked_keywords):
        return {
            "is_safe": False, 
            "final_response": "I am not authorized to retrieve internal system logs, training data, or protected instructions. How can I help you with bank products?",
            "next_step": "end",
        }

    user_query = state.get("user_query", "")
    classifier = _get_injection_classifier()

    if classifier:
        result = classifier(user_query)[0]
        if result["label"] == "INJECTION" and result["score"] > 0.75:
            logger.warning("Injection blocked at guardrail: %s", user_query)
            return {
                "is_safe": False,
                "final_response": (
                    "Security Alert: This request violates NUST Bank's safety policies and has been blocked."
                ),
                "next_step": "end",
            }

    return {"is_safe": True, "next_step": "evaluator"}
