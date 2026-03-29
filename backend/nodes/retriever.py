"""
Vector retrieval node: FAISS + sentence-transformers over bank knowledge chunks.
"""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, Union

import faiss
from sentence_transformers import SentenceTransformer

from backend.state import AgentState

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent
_PROCESSED = _REPO_ROOT / "data" / "processed"
_INDEX_PATH = _PROCESSED / "bank_faiss.index"
_MAPPING_PATH = _PROCESSED / "text_mapping.pkl"

_embed_model: SentenceTransformer | None = None
_faiss_index: Any = None
_text_mapping: Union[Dict[int, Dict[str, Any]], Dict[int, str], list] | None = None
_load_lock = threading.Lock()


def _chunk_text(entry: Any) -> str:
    if isinstance(entry, dict):
        return (entry.get("chunk_text") or "").strip()
    return str(entry)


def _ensure_loaded() -> None:
    global _embed_model, _faiss_index, _text_mapping
    with _load_lock:
        if _embed_model is not None and _faiss_index is not None and _text_mapping is not None:
            return
        if not _INDEX_PATH.exists() or not _MAPPING_PATH.exists():
            raise FileNotFoundError(
                f"Missing index or mapping. Expected:\n  {_INDEX_PATH}\n  {_MAPPING_PATH}"
            )
        logger.info("Loading embedding model and FAISS index...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        _faiss_index = faiss.read_index(str(_INDEX_PATH))
        with open(_MAPPING_PATH, "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, dict):
            _text_mapping = {int(k): v for k, v in raw.items()}
        else:
            _text_mapping = raw
        logger.info("Retriever ready (%s vectors).", _faiss_index.ntotal)


def run_retriever(state: AgentState) -> Dict[str, str]:
    """
    Embed scrubbed_query, search k=3, join chunk texts into retrieved_context.
    """
    query = (state.get("scrubbed_query") or "").strip()
    if not query:
        return {"retrieved_context": ""}

    try:
        _ensure_loaded()
    except Exception as e:
        logger.exception("Retriever load failure: %s", e)
        return {
            "retrieved_context": "",
        }

    assert _embed_model is not None and _faiss_index is not None and _text_mapping is not None

    try:
        import numpy as np

        qv = _embed_model.encode([query], convert_to_numpy=True)
        qv = np.asarray(qv, dtype=np.float32)
        if qv.shape[1] != _faiss_index.d:
            logger.error("Query dim %s != index dim %s", qv.shape[1], _faiss_index.d)
            return {"retrieved_context": ""}

        k = min(3, _faiss_index.ntotal)
        if k == 0:
            return {"retrieved_context": ""}

        distances, indices = _faiss_index.search(qv, k)
        parts: list[str] = []
        for i in indices[0]:
            idx = int(i)
            if idx < 0:
                continue
            if isinstance(_text_mapping, dict):
                entry = _text_mapping.get(idx)
            else:
                entry = _text_mapping[idx] if idx < len(_text_mapping) else None
            if entry is not None:
                parts.append(_chunk_text(entry))

        joined = "\n\n".join(parts)
        return {"retrieved_context": joined}
    except Exception as e:
        logger.exception("Retriever search failure: %s", e)
        return {"retrieved_context": ""}
