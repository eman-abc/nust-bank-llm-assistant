# File: backend/nodes/guardrail.py
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

# 1. Load the model globally so it only initializes once
try:
    logger.info("Loading DeBERTa Guardrail Model...")
    injection_classifier = pipeline(
        "text-classification", 
        model="ProtectAI/deberta-v3-base-injection"
    )
except Exception as e:
    logger.error(f"Failed to load guardrail model: {e}")
    injection_classifier = None

# 2. Define the LangGraph Node
def guardrail_node(state: dict):
    """
    Semantic Firewall: Checks for prompt injection BEFORE hitting the LLM.
    """
    user_query = state.get("user_query", "")
    
    if injection_classifier:
        # Run the fast classification
        result = injection_classifier(user_query)[0]
        
        # The model returns labels like 'INJECTION' or 'SAFE'
        if result['label'] == 'INJECTION' and result['score'] > 0.75:
            logger.warning(f"🛡️ INJECTION BLOCKED: {user_query}")
            return {
                "is_safe": False,
                "final_response": "⚠️ SECURITY ALERT: This request violates NUST Bank's safety policies and has been blocked.",
                "next_step": "end" # Tell LangGraph to route to END
            }
            
    # If safe, tell the graph to proceed to your normal evaluator
    return {"is_safe": True, "next_step": "evaluator"}