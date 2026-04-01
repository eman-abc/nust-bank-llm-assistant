from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.config import get_settings
from backend.orchestrator import bank_bot


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_UPLOAD_SUFFIXES = {".txt", ".csv", ".pdf"}

app = FastAPI(
    title="NUST Bank Agentic API",
    description="REST API gateway for the LangGraph-powered banking assistant.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_query: str


class ChatResponse(BaseModel):
    final_response: str
    is_safe: bool
    scrubbed_query: str
    context_used: bool


class UploadAcceptedResponse(BaseModel):
    task_id: str
    status: str
    filename: str


class TaskStatusResponse(BaseModel):
    task_id: str
    state: str
    result: dict | None = None
    error: str | None = None


def queue_document_ingestion(file_path: Path, metadata: dict[str, str | int]) -> str:
    from backend.tasks.document_ingestion import ingest_document_task

    task = ingest_document_task.delay(str(file_path), metadata)
    return task.id


def get_task_status_payload(task_id: str) -> TaskStatusResponse:
    from celery.result import AsyncResult

    from backend.celery_app import celery_app

    task = AsyncResult(task_id, app=celery_app)
    info = task.info if isinstance(task.info, dict) else None
    error = None

    if task.state == "SUCCESS":
        info = task.result if isinstance(task.result, dict) else info
    elif task.state == "FAILURE":
        error = str(task.result or task.info)

    return TaskStatusResponse(
        task_id=task_id,
        state=task.state,
        result=info,
        error=error,
    )


@app.get("/health")
def health_check():
    return {"status": "operational", "system": "NUST Bank LangGraph Orchestrator"}


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        if not request.user_query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        result = bank_bot.invoke({"user_query": request.user_query})
        return ChatResponse(
            final_response=result.get("final_response", "Error generating response."),
            is_safe=result.get("is_safe", False),
            scrubbed_query=result.get("scrubbed_query", ""),
            context_used=bool(result.get("selected_context") or result.get("retrieved_context")),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("API Error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal Server Error") from exc


@app.post("/api/upload", status_code=202, response_model=UploadAcceptedResponse)
async def upload_document(file: UploadFile = File(...)):
    settings = get_settings()
    staged_path: Path | None = None

    try:
        filename = Path(file.filename or "upload.bin").name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. Allowed types: {sorted(ALLOWED_UPLOAD_SUFFIXES)}",
            )

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty.")

        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        staged_path = settings.uploads_dir / f"{uuid4().hex}_{filename}"
        staged_path.write_bytes(content)

        task_id = queue_document_ingestion(
            staged_path,
            {
                "source_file": filename,
                "topic": filename,
                "question": "Uploaded policy",
                "sheet": "User Upload",
                "source_row_index": -1,
                "source_type": "upload",
            },
        )
        logger.info("Queued upload '%s' as task %s", filename, task_id)
        return UploadAcceptedResponse(task_id=task_id, status="queued", filename=filename)
    except HTTPException:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
        logger.error("Upload queueing error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to queue file: {exc}") from exc


@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse)
def task_status(task_id: str):
    try:
        return get_task_status_payload(task_id)
    except Exception as exc:
        logger.error("Task status lookup failed for %s: %s", task_id, exc)
        raise HTTPException(status_code=500, detail="Failed to fetch task status.") from exc


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        result = bank_bot.invoke({"user_query": request.user_query})
        full_text = result.get("final_response", "")

        words = full_text.split(" ")
        for word in words:
            yield f"{word} "
            await asyncio.sleep(0.04)

    return StreamingResponse(event_generator(), media_type="text/plain")
