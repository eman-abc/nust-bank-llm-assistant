from __future__ import annotations

import logging
from typing import Any, Dict

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from backend.config import get_settings
from backend.state import AgentState

logger = logging.getLogger(__name__)

_ID_PATTERN = Pattern(name="numeric_id_pattern", regex=r"\b\d{5,13}\b", score=0.85)
_ID_RECOGNIZER = PatternRecognizer(supported_entity="SECURE_ID", patterns=[_ID_PATTERN])
_TARGET_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "SECURE_ID"]
_FORBIDDEN_KEYWORDS = [
    "hack",
    "bypass",
    "system prompt",
    "override",
    "jailbreak",
    "reveal instructions",
    "return my",
    "echo back",
    "pin",
    "cvv",
    "password",
    "steal",
]
_REFUSAL = "Security Alert: This request violates NUST Bank's safety policies and has been blocked."

_analyzer = None
_anonymizer = None


def _get_presidio_engines():
    global _analyzer, _anonymizer
    if _analyzer and _anonymizer:
        return _analyzer, _anonymizer

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": get_settings().spacy_model}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    analyzer.registry.add_recognizer(_ID_RECOGNIZER)

    _analyzer = analyzer
    _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


def run_privacy_sanitizer(state: AgentState) -> Dict[str, Any]:
    query = (state.get("user_query") or "").strip()
    if not query:
        return {"is_safe": False, "final_response": "Please enter a question."}

    lowered_query = query.lower()
    if any(keyword in lowered_query for keyword in _FORBIDDEN_KEYWORDS):
        logger.warning("Sensitive or disallowed query blocked: %s", query)
        return {
            "is_safe": False,
            "scrubbed_query": "",
            "final_response": _REFUSAL,
        }

    try:
        analyzer, anonymizer = _get_presidio_engines()
        results = analyzer.analyze(text=query, language="en", entities=_TARGET_ENTITIES)
        anonymized = anonymizer.anonymize(text=query, analyzer_results=results)
        return {
            "is_safe": True,
            "scrubbed_query": anonymized.text.strip(),
        }
    except Exception as exc:
        logger.error("Privacy sanitizer failed: %s", exc)
        return {
            "is_safe": False,
            "final_response": "Internal security error. Please try again.",
        }
