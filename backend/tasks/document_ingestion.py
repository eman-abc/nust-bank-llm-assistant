from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.celery_app import celery_app
from backend.config import get_settings
from backend.services.document_parser import parse_document
from backend.services.embedding_service import embed_texts
from backend.services.qdrant_store import build_payload, upsert_embeddings

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="backend.tasks.document_ingestion.ingest_document_task")
def ingest_document_task(self, file_path: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    source_path = Path(file_path)
    settings = get_settings()

    self.update_state(
        state="STARTED",
        meta={"status": "processing", "source_file": metadata.get("source_file", source_path.name)},
    )

    try:
        text = parse_document(source_path)
        if not text.strip():
            raise ValueError("Uploaded document did not contain any extractable text.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]
        if not chunks:
            raise ValueError("No chunks were generated from the uploaded document.")

        vectors = np.asarray(embed_texts(chunks), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("Embedding service returned invalid vector output.")
        ingested_at = datetime.now(timezone.utc).isoformat()
        source_file = metadata.get("source_file", source_path.name)
        topic = metadata.get("topic", source_file)

        payloads = [
            build_payload(
                chunk_text=chunk,
                question=metadata.get("question", "Uploaded policy"),
                sheet=metadata.get("sheet", "User Upload"),
                topic=topic,
                source_row_index=int(metadata.get("source_row_index", -1)),
                source_file=source_file,
                chunk_index=index,
                ingested_at=ingested_at,
                answer=metadata.get("answer", ""),
                source_type=metadata.get("source_type", "upload"),
            )
            for index, chunk in enumerate(chunks)
        ]

        points_upserted = upsert_embeddings(
            vectors=vectors,
            payloads=payloads,
            vector_size=vectors.shape[1],
        )
        return {
            "status": "completed",
            "points_upserted": points_upserted,
            "source_file": source_file,
            "collection": settings.qdrant_collection,
        }
    except Exception:
        logger.exception("Document ingestion failed for %s", source_path)
        raise
    finally:
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Unable to delete staged upload: %s", source_path)
