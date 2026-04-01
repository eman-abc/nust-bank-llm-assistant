from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from backend.config import PROCESSED_DIR
from backend.services.qdrant_store import build_payload, upsert_embeddings


def _load_text_mapping(mapping_path: Path):
    with mapping_path.open("rb") as handle:
        raw = pickle.load(handle)

    if isinstance(raw, dict):
        return {int(key): value for key, value in raw.items()}
    return {index: value for index, value in enumerate(raw)}


def _reconstruct_vectors(index) -> np.ndarray:
    vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
    for row_index in range(index.ntotal):
        vectors[row_index] = index.reconstruct(row_index)
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy FAISS vectors into Qdrant.")
    parser.add_argument(
        "--index-path",
        default=str(PROCESSED_DIR / "bank_faiss.index"),
        help="Path to the legacy FAISS index.",
    )
    parser.add_argument(
        "--mapping-path",
        default=str(PROCESSED_DIR / "text_mapping.pkl"),
        help="Path to the legacy FAISS text mapping pickle.",
    )
    args = parser.parse_args()

    index_path = Path(args.index_path)
    mapping_path = Path(args.mapping_path)

    index = faiss.read_index(str(index_path))
    text_mapping = _load_text_mapping(mapping_path)
    vectors = _reconstruct_vectors(index)
    ingested_at = datetime.now(timezone.utc).isoformat()

    payloads = []
    for row_index in range(index.ntotal):
        entry = text_mapping.get(row_index, "")
        if isinstance(entry, dict):
            chunk_text = (entry.get("chunk_text") or "").strip()
            payloads.append(
                build_payload(
                    chunk_text=chunk_text,
                    question=entry.get("question", "Migrated knowledge"),
                    sheet=entry.get("sheet", "Migration"),
                    topic=entry.get("topic", "FAISS Migration"),
                    source_row_index=int(entry.get("source_row_index", row_index)),
                    source_file=entry.get("topic", "FAISS Migration"),
                    chunk_index=row_index,
                    ingested_at=ingested_at,
                    answer=entry.get("answer", ""),
                    source_type="migration",
                )
            )
        else:
            payloads.append(
                build_payload(
                    chunk_text=str(entry),
                    question="Migrated knowledge",
                    sheet="Migration",
                    topic="FAISS Migration",
                    source_row_index=row_index,
                    source_file="FAISS Migration",
                    chunk_index=row_index,
                    ingested_at=ingested_at,
                    source_type="migration",
                )
            )

    inserted = upsert_embeddings(vectors=vectors, payloads=payloads, vector_size=index.d)
    print(
        f"Migrated {inserted} vectors from '{index_path.name}' into Qdrant collection "
        f"using mapping '{mapping_path.name}'."
    )


if __name__ == "__main__":
    main()
