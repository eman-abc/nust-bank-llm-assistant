# NUST Bank LLM Assistant 🏦

A production-grade Agentic RAG system for NUST Bank, capable of handling account queries, loan information, and policy ingestion with high security and accuracy.

## 🚀 Key Features

- **Agentic Orchestration**: Powered by LangGraph for intelligent multi-step reasoning.
- **Hybrid Retrieval**: Combines semantic (dense) and keyword (sparse) search via Qdrant.
- **Security Guardrails**: Hardened against prompt injection, PII requests, and "training data" leakage.
- **Provenance & Citations**: Every answer includes clickable sources with human-readable titles.
- **Custom Model Support**: Optimized for the `eman-abc/gemma-3-4b-bank-merged` model with robust API fallbacks.
- **Multimodal Ingestion**: Background indexing for PDFs and banking documents via Celery.

## 🛠️ Prerequisites

- **Python 3.10+**
- **Docker Desktop** (Running Redis and Qdrant)
- **Hugging Face Token** (with access to Llama/Gemma models)

## 📦 Local Setup

1. **Clone and Install**:
```powershell
git clone https://github.com/eman-abc/nust-bank-llm-assistant.git
cd nust-bank-llm-assistant
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
```

2. **Infrastructure (Docker)**:
```powershell
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
docker run -d -p 6379:6379 redis
```

3. **Environment**:
Create a `.env` file in the root:
```env
HF_TOKEN="your_huggingface_token"
REDIS_URL="redis://127.0.0.1:6379/0"
QDRANT_HOST="127.0.0.1"
```

## 🏃 Running the Assistant

You need to run these **three services** in separate terminals:

### 1. The Backend (FastAPI)
```powershell
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

### 2. The Background Worker (Celery)
```powershell
python -m celery -A backend.celery_app:celery_app worker --loglevel=info --pool=solo
```

### 3. The Interactive UI (Gradio)
```powershell
python frontend/gradio_app.py
```

## 🛡️ Security Note
The system includes a semantic firewall that blocks adversarial prompts and unauthorized requests for internal system records, ensuring customer data privacy and model integrity.
