<<<<<<< HEAD
# NUST Bank LLM Assistant (Local + HF Spaces)

This project provides a **banking Q&A assistant** that uses:

- **FAISS** retrieval over your local knowledge base (`data/processed/bank_faiss.index` + `text_mapping.pkl`)
- **Sentence-Transformers** embeddings (`all-MiniLM-L6-v2`)
- **Qwen2.5-3B-Instruct** (under 6B parameters) via **Hugging Face Inference API** (requires `HF_TOKEN`)
- A **Gradio** chat UI

## Run locally (Windows)

1. Create & activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure retrieval artifacts exist:

- `data/processed/bank_faiss.index`
- `data/processed/text_mapping.pkl`

If they don’t exist, generate them using the notebooks in `notebooks/` (data ingestion + indexing).

4. Provide your Hugging Face token

Option A (recommended): create a local `.env` file in the repo root:

```powershell
HF_TOKEN="YOUR_TOKEN_HERE"
```

Option B: set it as an environment variable in PowerShell:

```powershell
$env:HF_TOKEN="YOUR_TOKEN_HERE"
```

5. Start the Gradio app:

```bash
python frontend/gradio_app.py
```

Open the printed local URL in your browser.

## Deploy on Hugging Face Spaces

1. Create a new Space (Gradio).
2. Push this repository (or copy these files) to the Space.
3. Add a Space secret named `HF_TOKEN`.
4. Set the Space **app file** to `frontend/gradio_app.py` (or move it to root as `app.py` if you prefer).

## Notes

- Optional: override the inference model with `HF_LLM_MODEL` (must stay under **6B parameters** per project policy). Default is `Qwen/Qwen2.5-3B-Instruct`. Example: `google/gemma-2-2b-it`.
- Large artifacts are often gitignored; make sure `data/processed/*` is available in whatever environment you run.

=======
# 🏦 NUST Bank AI Assistant
A **Retrieval-Augmented Generation (RAG)** system designed to provide accurate, context-aware information about NUST Bank’s policies, profit rates, and services.

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/eman-abc/NUST-Bank-Assistant)

---

## Live Demo
You can interact with the live AI assistant here:
**[Click here to open NUST Bank AI](https://huggingface.co/spaces/eman-abc/NUST-Bank-Assistant)**

---

## Overview
Traditional chatbots often struggle with **"hallucinations"** or outdated information. This project implements a RAG architecture to ensure that the AI only answers based on verified bank documentation. By combining a local vector database with a powerful cloud-based LLM, the assistant provides bolded, structured, and factually grounded responses.

## Key Features
* **Contextual Accuracy:** Uses FAISS to retrieve the most relevant bank policy snippets before generating an answer.
* **Hallucination Guardrails:** Strictly instructed to admit when information is missing rather than fabricating answers.
* **High-Performance LLM:** Leverages **Qwen 2.5 7B Instruct** via Hugging Face Inference API for stable and fast reasoning.
* **Professional UI:** Built with **Gradio** for a clean, user-friendly chat interface.

## Tech Stack
* **LLM:** `Qwen/Qwen2.5-7B-Instruct`
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Database:** `FAISS`
* **UI Framework:** `Gradio`
>>>>>>> 58d70ce0e4d5cb3ee1a8a0e05985a1bd39133c2c
