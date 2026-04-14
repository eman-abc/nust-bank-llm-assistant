"""
Intent Classifier Node: Detects greetings to skip retrieval.
"""

from __future__ import annotations
from typing import Dict
from backend.state import AgentState

GREETINGS = {
    "hi", "hello", "hey", "greetings", "good morning", 
    "good afternoon", "good evening", "thanks", "thank you"
}

def run_intent_classifier(state: AgentState) -> Dict[str, str]:
    query = (state.get("normalized_query") or "").lower().strip()
    
    # Simple rule-based check
    if any(greet in query for greet in GREETINGS) and len(query.split()) < 4:
        return {"query_intent": "greeting"}
    
    return {"query_intent": "banking_query"}
