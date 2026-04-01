from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

from backend.services.embedding_service import embed_texts


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = Counter(normalize_text(prediction).split())
    ref_tokens = Counter(normalize_text(reference).split())
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = sum((pred_tokens & ref_tokens).values())
    precision = overlap / sum(pred_tokens.values())
    recall = overlap / sum(ref_tokens.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def semantic_similarity(prediction: str, reference: str) -> float:
    vectors = embed_texts([prediction, reference])
    pred_vector, ref_vector = vectors[0], vectors[1]
    denominator = float((pred_vector @ pred_vector) ** 0.5 * (ref_vector @ ref_vector) ** 0.5)
    if denominator == 0:
        return 0.0
    return float((pred_vector @ ref_vector) / denominator)


def numeric_consistency(prediction: str, reference: str) -> float:
    pred_numbers = set(NUMBER_PATTERN.findall(prediction))
    ref_numbers = set(NUMBER_PATTERN.findall(reference))
    if not ref_numbers:
        return 1.0
    return 1.0 if pred_numbers.issubset(ref_numbers) else 0.0


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if not rank else 1.0 / rank


def ndcg(rank: int | None) -> float:
    return 0.0 if not rank else 1.0 / math.log2(rank + 1)


def citation_rank(expected_doc_id: str, citations: Sequence[dict[str, str]]) -> int | None:
    for index, citation in enumerate(citations, start=1):
        if citation.get("doc_id") == expected_doc_id:
            return index
    return None
