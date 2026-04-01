from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from backend.config import get_settings


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[/-][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class SparseEmbedding:
    indices: list[int]
    values: list[float]

    def to_qdrant(self) -> dict[str, list[int] | list[float]]:
        return {"indices": self.indices, "values": self.values}


def _stable_hash(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16) % get_settings().sparse_hash_space


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _weight_token(token: str, frequency: int) -> float:
    weight = 1.0 + math.log1p(frequency)
    if any(char.isdigit() for char in token):
        weight *= 1.25
    if len(token) > 10:
        weight *= 1.05
    return weight


def encode_sparse_text(text: str) -> SparseEmbedding:
    counts = Counter(tokenize(text))
    if not counts:
        return SparseEmbedding(indices=[], values=[])

    hashed: dict[int, float] = {}
    for token, frequency in counts.items():
        token_id = _stable_hash(token)
        hashed[token_id] = hashed.get(token_id, 0.0) + _weight_token(token, frequency)

    ordered = sorted(hashed.items())
    return SparseEmbedding(
        indices=[index for index, _ in ordered],
        values=[value for _, value in ordered],
    )


def encode_sparse_texts(texts: Sequence[str]) -> list[SparseEmbedding]:
    return [encode_sparse_text(text) for text in texts]


def overlap_score(query_tokens: Iterable[str], document_text: str) -> float:
    query_counter = Counter(token.lower() for token in query_tokens if token)
    doc_counter = Counter(tokenize(document_text))
    if not query_counter or not doc_counter:
        return 0.0

    shared = sum(min(query_counter[token], doc_counter[token]) for token in query_counter)
    total = max(sum(query_counter.values()), 1)
    return shared / total
