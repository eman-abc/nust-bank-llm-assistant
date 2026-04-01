# NUST Bank LLM Assistant

A banking-domain hybrid RAG assistant that combines LangGraph orchestration, FastAPI, Gradio, Qdrant, Redis, Celery, and Hugging Face inference.

The system is designed to answer banking support questions using retrieved knowledge, apply safety and privacy checks before generation, and reject low-confidence or ungrounded answers instead of guessing.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Request Lifecycle](#request-lifecycle)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Running with Docker Compose](#running-with-docker-compose)
- [Running Locally Without Docker](#running-locally-without-docker)
- [API Reference](#api-reference)
- [Document Ingestion](#document-ingestion)
- [Evaluation Workflow](#evaluation-workflow)
- [Architecture Diagram](#architecture-diagram)
- [Legacy FAISS Migration](#legacy-faiss-migration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Overview

This repository contains a multi-stage RAG pipeline for banking-domain question answering.

Instead of sending a user prompt straight to an LLM, the system routes each request through a deterministic LangGraph workflow:

`guardrail -> privacy_sanitizer -> query_normalizer -> hybrid_retriever -> reranker -> context_builder -> synthesizer -> grounding_checker`

That design gives the project a few important properties:

- Prompt-injection and unsafe requests can be blocked early.
- Sensitive user data can be anonymized before downstream processing.
- Retrieval can combine semantic similarity with lexical matching.
- Answers can be filtered through a final grounding check before they are returned.
- Long-running ingestion work is moved off the chat path with Celery.

## Key Features

- Hybrid retrieval with dense and sparse search over Qdrant.
- LangGraph state-machine orchestration for predictable control flow.
- Prompt-injection screening with a DeBERTa-based classifier.
- Privacy sanitization using Presidio and spaCy.
- Query normalization and lightweight intent detection.
- Cross-encoder reranking with lexical fallback when the reranker cannot load.
- Citation-aware context construction from top-ranked chunks.
- Grounding checks for missing evidence, weak retrieval confidence, and numeric inconsistency on rate-style queries.
- FastAPI endpoints for chat, streaming chat, upload queueing, and task-status polling.
- Gradio frontend for chat and knowledge-base uploads.
- Celery worker pipeline for background ingestion of `.txt`, `.csv`, and `.pdf` files.
- Offline evaluation scripts for retrieval and answer-quality reporting.

## Architecture

### Runtime Components

- `frontend`: Gradio UI served on port `7860`
- `backend`: FastAPI service served on port `8000`
- `worker`: Celery worker for document ingestion
- `redis`: broker and result backend for Celery
- `qdrant`: vector database for hybrid retrieval

### LangGraph Workflow

1. `guardrail`
   Blocks prompt-injection attempts before they enter the main pipeline.
2. `privacy_sanitizer`
   Detects and anonymizes sensitive content using Presidio.
3. `query_normalizer`
   Cleans the query, infers intent, and attaches retrieval metadata filters.
4. `hybrid_retriever`
   Generates dense and sparse query representations and retrieves candidates from Qdrant.
5. `reranker`
   Reorders retrieved candidates using a cross-encoder or lexical fallback.
6. `context_builder`
   Builds the final context window and citation list from the top-ranked chunks.
7. `synthesizer`
   Calls the Hugging Face Inference API to produce the final response from grounded context.
8. `grounding_checker`
   Rejects weakly supported answers and replaces them with a safe fallback when needed.

## Technology Stack

- Python 3.11
- FastAPI
- Gradio
- LangGraph
- Qdrant
- Redis
- Celery
- Hugging Face Inference API
- `sentence-transformers`
- `transformers`
- Presidio Analyzer + Presidio Anonymizer
- spaCy
- FAISS for legacy migration tooling

## Project Structure

```text
backend/
  api.py                       FastAPI application and API routes
  celery_app.py                Celery app configuration
  config.py                    Environment-driven settings
  orchestrator.py              LangGraph workflow definition
  state.py                     Shared graph state
  nodes/                       Graph nodes for safety, retrieval, generation, and grounding
  services/                    Retrieval, embedding, parsing, and reranking services
  tasks/                       Background ingestion tasks
  scripts/                     Backend-side migration utilities
frontend/
  gradio_app.py                Gradio chat and upload interface
eval/
  generate_golden_dataset.py   Golden dataset generation
  run_evaluation.py            Offline evaluator
  metrics.py                   Evaluation metrics
scripts/
  generate_architecture_diagram.py
docs/
  architecture/                Mermaid and PNG architecture artifacts
tests/
  Automated tests for API, orchestration, ingestion, retrieval, and evaluation
```

## Request Lifecycle

### Chat Request

1. A user sends a query through Gradio or directly to `POST /api/chat`.
2. FastAPI invokes the compiled LangGraph workflow.
3. The graph runs safety checks, retrieval, reranking, context building, generation, and grounding validation.
4. The final answer is returned only if the request passes the required gates.

### Upload Request

1. A user uploads a `.txt`, `.csv`, or `.pdf` file.
2. FastAPI stages the file under the uploads directory.
3. The backend queues a Celery ingestion task.
4. The worker parses the file, chunks the text, generates embeddings, and upserts chunks into Qdrant.
5. The frontend polls task status until the ingestion completes or fails.

## Prerequisites

- Python 3.11
- Docker and Docker Compose
- Access to a Hugging Face token for response synthesis

## Environment Variables

Create a `.env` file in the repository root.

### Required

```env
HF_TOKEN=your_hugging_face_token
```

### Common Optional Variables

```env
API_BASE_URL=http://127.0.0.1:8000/api
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=bank_knowledge
QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_SPARSE_VECTOR_NAME=sparse
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_VECTOR_SIZE=384
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
SPACY_MODEL=en_core_web_sm
DENSE_RETRIEVAL_LIMIT=8
SPARSE_RETRIEVAL_LIMIT=8
RERANK_TOP_K=6
FINAL_CONTEXT_K=3
RETRIEVAL_CONFIDENCE_THRESHOLD=0.2
```

## Running with Docker Compose

This is the recommended way to run the full stack.

```bash
docker compose up --build
```

### Exposed Services

- Gradio UI: `http://127.0.0.1:7860`
- FastAPI backend: `http://127.0.0.1:8000`
- FastAPI health check: `http://127.0.0.1:8000/health`
- Qdrant: `http://127.0.0.1:6333`
- Redis: `redis://127.0.0.1:6379`

### Notes

- The backend and worker share the same application image.
- Redis is used as both Celery broker and result backend.
- Qdrant stores dense and sparse vectors for hybrid retrieval.
- The backend image installs the configured spaCy English model during build.

## Running Locally Without Docker

1. Create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Start infrastructure dependencies.

- Start Redis
- Start Qdrant

4. Start the backend.

```bash
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

5. Start the Celery worker in a separate terminal.

```bash
celery -A backend.celery_app:celery_app worker --loglevel=info
```

6. Start the Gradio frontend in another terminal.

```bash
python frontend/gradio_app.py
```

## API Reference

### `GET /health`

Returns backend health information.

Example response:

```json
{
  "status": "operational",
  "system": "NUST Bank LangGraph Orchestrator"
}
```

### `POST /api/chat`

Runs the synchronous chat pipeline.

Request:

```json
{
  "user_query": "What are the requirements for a Roshan Digital Account?"
}
```

Response shape:

```json
{
  "final_response": "string",
  "is_safe": true,
  "scrubbed_query": "string",
  "context_used": true
}
```

### `POST /api/chat/stream`

Streams the final response as plain text.

### `POST /api/upload`

Accepts a file upload and queues background ingestion.

Supported upload formats:

- `.txt`
- `.csv`
- `.pdf`

Response shape:

```json
{
  "task_id": "string",
  "status": "queued",
  "filename": "document.pdf"
}
```

### `GET /api/tasks/{task_id}`

Returns Celery task status and final ingestion metadata when available.

## Document Ingestion

The ingestion pipeline is asynchronous and designed to keep uploads off the main request path.

### Flow

1. Parse the uploaded document based on file type.
2. Split content with `RecursiveCharacterTextSplitter`.
3. Generate dense embeddings for each chunk.
4. Generate sparse representations for hybrid retrieval.
5. Upsert the chunk payloads and vectors into Qdrant.

### Parsed File Types

- Text files are read directly.
- CSV files are flattened row-by-row into text.
- PDF files are parsed with `pypdf`.

## Evaluation Workflow

Generate a reviewable golden dataset:

```bash
python -m eval.generate_golden_dataset
```

Run the offline evaluator:

```bash
python -m eval.run_evaluation
```

Artifacts are written under `eval/artifacts/`.

### Reported Metrics

- Recall@K
- MRR
- NDCG
- Citation hit rate
- Exact match
- Token F1
- Semantic similarity
- Numeric consistency
- Groundedness pass rate

## Architecture Diagram

Generate Mermaid and PNG architecture artifacts with:

```bash
python -m scripts.generate_architecture_diagram
```

Artifacts are written to `docs/architecture/`.

## Legacy FAISS Migration

If you have an older FAISS-based index, you can migrate it into Qdrant.

```bash
python -m backend.scripts.migrate_faiss_to_qdrant
```

Optional flags:

```bash
python -m backend.scripts.migrate_faiss_to_qdrant \
  --index-path data/processed/bank_faiss.index \
  --mapping-path data/processed/text_mapping.pkl
```

## Testing

Run the test suite with:

```bash
python -m pytest
```

You can also run targeted tests, for example:

```bash
python -m pytest tests/test_orchestrator.py
python -m pytest tests/test_upload_queue_api.py
```

## Troubleshooting

### Docker build fails with access-denied errors

Make sure inaccessible temp directories are excluded from the Docker build context. This repository already ignores common pytest temp/cache paths through `.dockerignore`.

### Gradio cannot connect to the backend

Check that the backend is running on port `8000` and that `API_BASE_URL` is set correctly for your environment.

### Uploads stay stuck in queued or started state

Verify that:

- Redis is running
- The Celery worker is running
- The worker can reach Qdrant
- The uploaded file type is supported

### The backend returns safe fallback answers too often

Likely causes include:

- Qdrant does not contain the expected knowledge yet
- The retrieval confidence threshold is too strict for the current corpus
- Citations are missing from selected context
- The numeric grounding check rejected the generated answer

### spaCy model errors

Install the configured spaCy model locally:

```bash
python -m spacy download en_core_web_sm
```
