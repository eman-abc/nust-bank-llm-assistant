# NUST Bank LLM Assistant (Local + HF Spaces)

This project provides a **banking Q&A assistant** that uses:

- **FAISS** retrieval over your local knowledge base (`data/processed/bank_faiss.index` + `text_mapping.pkl`)
- **Sentence-Transformers** embeddings (`all-MiniLM-L6-v2`)
- **Qwen2.5-7B-Instruct** via **Hugging Face Inference API** (requires `HF_TOKEN`)
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

- If you want **fully offline** inference on your PC, use the existing `frontend/app.py` Streamlit local model approach instead of the HF Inference API.
- Large artifacts are often gitignored; make sure `data/processed/*` is available in whatever environment you run.

