from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
UPLOADS_DIR = DATA_DIR / "uploads"
EVAL_DIR = REPO_ROOT / "eval"
EVAL_ARTIFACTS_DIR = EVAL_DIR / "artifacts"
DOCS_DIR = REPO_ROOT / "docs"
ARCHITECTURE_ARTIFACTS_DIR = DOCS_DIR / "architecture"

load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    qdrant_url: str
    qdrant_collection: str
    qdrant_dense_vector_name: str
    qdrant_sparse_vector_name: str
    redis_url: str
    celery_broker_url: str
    celery_result_backend: str
    hf_token: str
    embedding_model: str
    embedding_vector_size: int
    reranker_model: str
    spacy_model: str
    uploads_dir: Path
    eval_artifacts_dir: Path
    architecture_artifacts_dir: Path
    dense_retrieval_limit: int
    sparse_retrieval_limit: int
    rerank_top_k: int
    final_context_k: int
    sparse_hash_space: int
    retrieval_confidence_threshold: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return Settings(
        api_base_url=os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api"),
        qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "bank_knowledge"),
        qdrant_dense_vector_name=os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense"),
        qdrant_sparse_vector_name=os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse"),
        redis_url=redis_url,
        celery_broker_url=os.getenv("CELERY_BROKER_URL", redis_url),
        celery_result_backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1"),
        hf_token=os.getenv("HF_TOKEN", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        embedding_vector_size=int(os.getenv("EMBEDDING_VECTOR_SIZE", "384")),
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        spacy_model=os.getenv("SPACY_MODEL", "en_core_web_sm"),
        uploads_dir=Path(os.getenv("UPLOADS_DIR", str(UPLOADS_DIR))),
        eval_artifacts_dir=Path(os.getenv("EVAL_ARTIFACTS_DIR", str(EVAL_ARTIFACTS_DIR))),
        architecture_artifacts_dir=Path(
            os.getenv("ARCHITECTURE_ARTIFACTS_DIR", str(ARCHITECTURE_ARTIFACTS_DIR))
        ),
        dense_retrieval_limit=int(os.getenv("DENSE_RETRIEVAL_LIMIT", "8")),
        sparse_retrieval_limit=int(os.getenv("SPARSE_RETRIEVAL_LIMIT", "8")),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "6")),
        final_context_k=int(os.getenv("FINAL_CONTEXT_K", "3")),
        sparse_hash_space=int(os.getenv("SPARSE_HASH_SPACE", "2000003")),
        retrieval_confidence_threshold=float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.2")),
    )
