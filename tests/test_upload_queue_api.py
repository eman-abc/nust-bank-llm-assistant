from pathlib import Path
import shutil
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.api import app
import backend.api as api_module


def test_upload_endpoint_queues_background_task(monkeypatch):
    uploads_dir = Path.cwd() / "data" / "test_uploads"
    if uploads_dir.exists():
        shutil.rmtree(uploads_dir)

    try:
        monkeypatch.setattr(
            api_module,
            "get_settings",
            lambda: SimpleNamespace(uploads_dir=uploads_dir),
        )
        monkeypatch.setattr(api_module, "queue_document_ingestion", lambda path, metadata: "task-123")

        client = TestClient(app)
        response = client.post(
            "/api/upload",
            files={"file": ("policy.txt", b"sample banking policy", "text/plain")},
        )

        assert response.status_code == 202
        assert response.json() == {
            "task_id": "task-123",
            "status": "queued",
            "filename": "policy.txt",
        }
        staged_files = list(uploads_dir.glob("*_policy.txt"))
        assert len(staged_files) == 1
    finally:
        if uploads_dir.exists():
            shutil.rmtree(uploads_dir)


def test_task_status_endpoint_returns_payload(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "get_task_status_payload",
        lambda task_id: api_module.TaskStatusResponse(
            task_id=task_id,
            state="SUCCESS",
            result={"points_upserted": 3, "source_file": "policy.txt", "collection": "bank_knowledge"},
            error=None,
        ),
    )

    client = TestClient(app)
    response = client.get("/api/tasks/task-123")

    assert response.status_code == 200
    assert response.json()["state"] == "SUCCESS"
    assert response.json()["result"]["points_upserted"] == 3
