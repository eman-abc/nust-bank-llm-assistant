from __future__ import annotations
import logging
from typing import Any, Dict, List
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from backend.state import AgentState

logger = logging.getLogger(__name__)

# 1. Custom Security Rules
# Catching numeric leaks (5-13 digits) that standard NER might miss
_ID_PATTERN = Pattern(name="numeric_id_pattern", regex=r"\b\d{5,13}\b", score=0.85)
_ID_RECOGNIZER = PatternRecognizer(supported_entity="SECURE_ID", patterns=[_ID_PATTERN])

# Targeted entities for the NLP engine
_TARGET_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "SECURE_ID"]

# Forbidden substrings (Case-insensitive)
_FORBIDDEN_KEYWORDS = [
    "hack", "bypass", "ignore previous", "system prompt", "override", 
    "jailbreak", "reveal instructions", "return my", "echo back"
]

_REFUSAL = "Security Alert: This request violates NUST Bank's safety policies and has been blocked."

# Global engines to prevent reloading on every request (LRU-style)
_analyzer = None
_anonymizer = None

def _get_engines():
    """Initializes Presidio with explicit spaCy en_core_web_lg configuration."""
    global _analyzer, _anonymizer
    if _analyzer and _anonymizer:
        return _analyzer, _anonymizer

    try:
        # Explicitly configure spaCy Large model for high accuracy
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        
        # Build Analyzer with our custom SECURE_ID recognizer
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        analyzer.registry.add_recognizer(_ID_RECOGNIZER)
        
        _analyzer = analyzer
        _anonymizer = AnonymizerEngine()
        return _analyzer, _anonymizer
    except Exception as e:
        logger.error(f"Failed to initialize Presidio engines: {e}")
        raise

def run_evaluator(state: AgentState) -> Dict[str, Any]:
    """
    Orchestrator node to sanitize input and defend against prompt injections.
    """
    query = (state.get("user_query") or "").strip()
    
    if not query:
        return {"is_safe": False, "final_response": "Please enter a question."}

    # STEP 1: Heuristic Guardrail (Fast check)
    lowered_query = query.lower()
    if any(word in lowered_query for word in _FORBIDDEN_KEYWORDS):
        return {
            "is_safe": False, 
            "scrubbed_query": "", 
            "final_response": _REFUSAL
        }

    # STEP 2: PII Scrubbing (Deep NLP + Regex check)
    try:
        analyzer, anonymizer = _get_engines()
        
        # Detect entities
        results = analyzer.analyze(
            text=query, 
            language="en", 
            entities=_TARGET_ENTITIES
        )
        
        # Mask entities
        anonymized = anonymizer.anonymize(text=query, analyzer_results=results)
        
        return {
            "is_safe": True, 
            "scrubbed_query": anonymized.text
        }
    except Exception as e:
        logger.error(f"Evaluator Node Failure: {e}")
        return {
            "is_safe": False, 
            "final_response": "Internal security error. Please try again."
        }