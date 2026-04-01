from __future__ import annotations

from celery import Celery

from backend.config import get_settings


settings = get_settings()

celery_app = Celery(
    "nust_bank_assistant",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["backend.tasks.document_ingestion"],
)
celery_app.conf.update(
    task_track_started=True,
    result_extended=True,
)
