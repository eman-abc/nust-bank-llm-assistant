from __future__ import annotations

import re
import uuid
from functools import lru_cache
from typing import Any, Iterable, Sequence

from backend.config import get_settings
from backend.services.sparse_encoder import SparseEmbedding, encode_sparse_texts


def _qdrant_models():
    from qdrant_client.http import models

    return models


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return cleaned.strip("-") or "document"


def build_doc_id(source_file: str, chunk_index: int, source_row_index: int = -1) -> str:
    return f"{_slugify(source_file)}::{source_row_index}::{chunk_index}"


@lru_cache(maxsize=1)
def get_qdrant_client():
    from qdrant_client import QdrantClient

    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(vector_size: int | None = None) -> None:
    settings = get_settings()
    client = get_qdrant_client()
    models = _qdrant_models()

    collections = client.get_collections().collections
    existing_names = {collection.name for collection in collections}
    if settings.qdrant_collection in existing_names:
        return

    try:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                settings.qdrant_dense_vector_name: models.VectorParams(
                    size=vector_size or settings.embedding_vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                settings.qdrant_sparse_vector_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
    except Exception:
        collections = client.get_collections().collections
        existing_names = {collection.name for collection in collections}
        if settings.qdrant_collection not in existing_names:
            raise


def build_payload(
    *,
    chunk_text: str,
    question: str,
    sheet: str,
    topic: str,
    source_row_index: int,
    source_file: str,
    chunk_index: int,
    ingested_at: str,
    answer: str = "",
    source_type: str = "knowledge",
    doc_id: str | None = None,
) -> dict[str, Any]:
    resolved_doc_id = doc_id or build_doc_id(source_file, chunk_index, source_row_index)
    return {
        "doc_id": resolved_doc_id,
        "chunk_text": chunk_text,
        "question": question,
        "answer": answer,
        "sheet": sheet,
        "topic": topic,
        "source_row_index": source_row_index,
        "source_file": source_file,
        "source_type": source_type,
        "chunk_index": chunk_index,
        "ingested_at": ingested_at,
    }


def _to_sparse_vector(sparse_embedding: SparseEmbedding):
    models = _qdrant_models()
    return models.SparseVector(indices=sparse_embedding.indices, values=sparse_embedding.values)


def build_filter(metadata_filters: dict[str, str] | None):
    if not metadata_filters:
        return None

    models = _qdrant_models()
    must_conditions = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in metadata_filters.items()
        if value
    ]
    if not must_conditions:
        return None
    return models.Filter(must=must_conditions)


def upsert_documents(
    *,
    dense_vectors: Sequence[Sequence[float]] | Any,
    sparse_vectors: Sequence[SparseEmbedding],
    payloads: Sequence[dict[str, Any]],
    ids: Iterable[str] | None = None,
    vector_size: int | None = None,
) -> int:
    if len(payloads) == 0:
        return 0
    if len(dense_vectors) != len(payloads):
        raise ValueError("The number of dense vectors must match the number of payloads.")
    if len(sparse_vectors) != len(payloads):
        raise ValueError("The number of sparse vectors must match the number of payloads.")

    ensure_collection(vector_size=vector_size)
    client = get_qdrant_client()
    models = _qdrant_models()
    settings = get_settings()

    point_ids = list(ids) if ids is not None else [payload["doc_id"] for payload in payloads]
    if len(point_ids) != len(payloads):
        raise ValueError("The number of Qdrant point IDs must match the number of payloads.")

    points = [
        models.PointStruct(
            id=point_id or uuid.uuid4().hex,
            vector={
                settings.qdrant_dense_vector_name: list(dense_vector),
                settings.qdrant_sparse_vector_name: _to_sparse_vector(sparse_vector),
            },
            payload=payload,
        )
        for point_id, dense_vector, sparse_vector, payload in zip(
            point_ids, dense_vectors, sparse_vectors, payloads, strict=False
        )
    ]

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
        wait=True,
    )
    return len(points)


def upsert_embeddings(
    *,
    vectors: Sequence[Sequence[float]] | Any,
    payloads: Sequence[dict[str, Any]],
    ids: Iterable[str] | None = None,
    vector_size: int | None = None,
) -> int:
    sparse_vectors = encode_sparse_texts([payload.get("chunk_text", "") for payload in payloads])
    return upsert_documents(
        dense_vectors=vectors,
        sparse_vectors=sparse_vectors,
        payloads=payloads,
        ids=ids,
        vector_size=vector_size,
    )


def hybrid_search(
    *,
    dense_vector: Sequence[float],
    sparse_vector: SparseEmbedding,
    limit: int,
    metadata_filters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    ensure_collection()
    client = get_qdrant_client()
    models = _qdrant_models()
    settings = get_settings()
    query_filter = build_filter(metadata_filters)

    prefetch = [
        models.Prefetch(
            query=list(dense_vector),
            using=settings.qdrant_dense_vector_name,
            limit=settings.dense_retrieval_limit,
            filter=query_filter,
        )
    ]

    if sparse_vector.indices and sparse_vector.values:
        prefetch.append(
            models.Prefetch(
                query=_to_sparse_vector(sparse_vector),
                using=settings.qdrant_sparse_vector_name,
                limit=settings.sparse_retrieval_limit,
                filter=query_filter,
            )
        )

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
        with_vectors=False,
        query_filter=query_filter,
    )

    points = getattr(response, "points", response)
    return [
        {
            "id": str(getattr(point, "id", "")),
            "score": float(getattr(point, "score", 0.0)),
            "payload": dict(getattr(point, "payload", {}) or {}),
        }
        for point in points
    ]


def search(vector: Sequence[float], limit: int = 3) -> list[dict[str, Any]]:
    candidates = hybrid_search(
        dense_vector=vector,
        sparse_vector=SparseEmbedding(indices=[], values=[]),
        limit=limit,
        metadata_filters=None,
    )
    return [candidate["payload"] for candidate in candidates]
