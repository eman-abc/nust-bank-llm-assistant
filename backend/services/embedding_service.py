from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    model = get_embedding_model()
    vectors = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embed_text(text: str) -> list[float]:
    vector = embed_texts([text])[0]
    return vector.tolist()
