from __future__ import annotations
import logging
from typing import Any, Dict

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from transformers import pipeline

from backend.state import AgentState

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Custom Security Rules & Constants
# =============================================================================

# Catching numeric leaks (5-13 digits) that standard NER might miss
_ID_PATTERN = Pattern(name="numeric_id_pattern", regex=r"\b\d{5,13}\b", score=0.85)
_ID_RECOGNIZER = PatternRecognizer(supported_entity="SECURE_ID", patterns=[_ID_PATTERN])

# Targeted entities for the NLP engine
_TARGET_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "SECURE_ID"]

# Forbidden substrings (Case-insensitive) - Added financial identifiers
_FORBIDDEN_KEYWORDS = [
    "hack", "bypass", "ignore previous", "system prompt", "override", 
    "jailbreak", "reveal instructions", "return my", "echo back",
    "pin", "cvv", "password", "steal"
]

_REFUSAL = "Security Alert: This request violates NUST Bank's safety policies and has been blocked."


# =============================================================================
# 2. Global Engine Initialization (Singleton Pattern)
# =============================================================================

# Global engines to prevent reloading on every request
_analyzer = None
_anonymizer = None
_injection_classifier = None

def _get_presidio_engines():
    """Initializes Presidio with explicit spaCy configuration."""
    global _analyzer, _anonymizer
    if _analyzer and _anonymizer:
        return _analyzer, _anonymizer

    try:
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        analyzer.registry.add_recognizer(_ID_RECOGNIZER)
        
        _analyzer = analyzer
        _anonymizer = AnonymizerEngine()
        return _analyzer, _anonymizer
    except Exception as e:
        logger.error(f"Failed to initialize Presidio engines: {e}")
        raise

def _get_firewall_engine():
    """Initializes the Hugging Face Prompt Injection classifier."""
    global _injection_classifier
    # If it's already loaded, or if it failed previously (False), return it
    if _injection_classifier is not None:
        return _injection_classifier
        
    try:
        logger.info("Loading DeBERTa Guardrail Model (this takes ~15s on first run)...")
        _injection_classifier = pipeline(
            "text-classification", 
            model="protectai/deberta-v3-base-prompt-injection-v2"
        )
    except Exception as e:
        logger.error(f"Failed to load guardrail model: {e}")
        # Set to False so we don't keep trying and crashing the app
        _injection_classifier = False 
        
    return _injection_classifier


# =============================================================================
# 3. The Unified Evaluator Node
# =============================================================================

def run_evaluator(state: AgentState) -> Dict[str, Any]:
    """
    Unified Security Hub: 
    1. Heuristic checks (Fast)
    2. Semantic Prompt Injection checks (Medium)
    3. PII Scrubbing (Deep NLP)
    """
    query = (state.get("user_query") or "").strip()
    
    if not query:
        return {"is_safe": False, "final_response": "Please enter a question."}

    # ---------------------------------------------------------
    # STEP 1: Heuristic Guardrail (Fast Check)
    # ---------------------------------------------------------
    lowered_query = query.lower()
    if any(word in lowered_query for word in _FORBIDDEN_KEYWORDS):
        logger.warning(f"🛡️ KEYWORD BLOCKED: {query}")
        return {
            "is_safe": False, 
            "scrubbed_query": "", 
            "final_response": _REFUSAL
        }

    # ---------------------------------------------------------
    # STEP 2: Semantic Firewall (Prompt Injection Detection)
    # ---------------------------------------------------------
    classifier = _get_firewall_engine()
    if classifier:  # Ensure it loaded successfully
        try:
            result = classifier(query)[0]
            # DeBERTa v2 returns "INJECTION" or "SAFE"
            if result['label'] == 'INJECTION' and result['score'] > 0.6:
                logger.warning(f"🛡️ INJECTION BLOCKED: {query}")
                return {
                    "is_safe": False,
                    "scrubbed_query": "",
                    "final_response": _REFUSAL
                }
        except Exception as e:
            logger.error(f"Semantic Firewall failed to process query: {e}")
            # If the firewall crashes, we fail-open or fail-closed depending on strictness. 
            # We will log the error and let it proceed to PII scrubbing.

    # ---------------------------------------------------------
    # STEP 3: PII Scrubbing (Deep NLP + Regex)
    # ---------------------------------------------------------
    try:
        analyzer, anonymizer = _get_presidio_engines()
        
        results = analyzer.analyze(
            text=query, 
            language="en", 
            entities=_TARGET_ENTITIES
        )
        
        anonymized = anonymizer.anonymize(text=query, analyzer_results=results)
        
        # If it passed keywords and semantic injection, it is safe!
        return {
            "is_safe": True, 
            "scrubbed_query": anonymized.text
        }
        
    except Exception as e:
        logger.error(f"Evaluator Node PII Failure: {e}")
        return {
            "is_safe": False, 
            "final_response": "Internal security error. Please try again."
        }