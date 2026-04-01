from backend.tasks.document_ingestion import ingest_document_task
import backend.tasks.document_ingestion as ingestion_module


def test_celery_ingestion_task_upserts_chunks(monkeypatch):
    monkeypatch.setattr(ingestion_module, "parse_document", lambda path: "Chunk one. Chunk two.")
    monkeypatch.setattr(ingestion_module, "embed_texts", lambda chunks: [[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr(
        ingestion_module,
        "upsert_embeddings",
        lambda **kwargs: len(kwargs["payloads"]),
    )

    states = []
    monkeypatch.setattr(ingest_document_task, "update_state", lambda **kwargs: states.append(kwargs))

    result = ingest_document_task.run("policy.txt", {"source_file": "policy.txt"})

    assert states[0]["state"] == "STARTED"
    assert result["status"] == "completed"
    assert result["points_upserted"] == 1
    assert result["source_file"] == "policy.txt"
