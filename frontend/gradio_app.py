import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import faiss
import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer




def _repo_root() -> Path:
    # Works whether launched from repo root, frontend/, or elsewhere.
    here = Path(__file__).resolve()
    return here.parents[1]


# Load local secrets file when present (ignored by git via `.gitignore`).
# On Hugging Face Spaces, HF_TOKEN should be provided via Space secrets.
load_dotenv(dotenv_path=_repo_root() / ".env", override=False)


def _artifact_paths() -> tuple[Path, Path]:
    """
    Prefer repo's processed artifacts, but also allow running from a Spaces
    working directory where files may be placed next to the app.
    """
    root = _repo_root()
    preferred_index = root / "data" / "processed" / "bank_faiss.index"
    preferred_mapping = root / "data" / "processed" / "text_mapping.pkl"

    if preferred_index.exists() and preferred_mapping.exists():
        return preferred_index, preferred_mapping

    cwd_index = Path.cwd() / "bank_faiss.index"
    cwd_mapping = Path.cwd() / "text_mapping.pkl"
    return cwd_index, cwd_mapping


def _format_numbers_bold(text: str) -> str:
    # Simple heuristic: bold digit sequences.
    import re

    return re.sub(r"(\d[\d,./-]*)", r"**\1**", text)


@lru_cache(maxsize=1)
def _load_retrieval():
    index_path, mapping_path = _artifact_paths()
    if not index_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(
            "Missing retrieval artifacts. Expected either:\n"
            f"- {index_path}\n- {mapping_path}\n\n"
            "If you're running locally, generate them via the notebooks in `notebooks/` "
            "or copy the files into place."
        )

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(str(index_path))
    with open(mapping_path, "rb") as f:
        text_mapping = pickle.load(f)
    return embed_model, index, text_mapping


def _client() -> InferenceClient:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it as an environment variable (local) or a Space secret (HF Spaces)."
        )
    return InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=token)


def bank_assistant(message: str, history: List[Tuple[str, str]]):
    embed_model, index, text_mapping = _load_retrieval()

    msg_lower = (message or "").strip().lower()
    if any(w in msg_lower for w in ["hi", "hello", "hey", "salam", "assalam"]) or "thank" in msg_lower:
        return "Hello! Welcome to NUST Bank. How can I assist you with our services today?"

    query_vector = embed_model.encode([message])
    distances, indices = index.search(query_vector, k=3)
    context = "\n".join([text_mapping[i] for i in indices[0]])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional NUST Bank AI. Answer using ONLY the provided Context. "
                "If the context does not contain the answer, strictly reply: "
                "'I apologize, but I do not have that specific information in my records.' "
                "Bold all numbers."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"},
    ]

    try:
        response = _client().chat_completion(messages=messages, max_tokens=150, temperature=0.1)
        answer = response.choices[0].message.content.strip()
        return _format_numbers_bold(answer)
    except Exception as e:
        return (
            "System Error: ensure `HF_TOKEN` is set and the Space has access to the model. "
            f"({str(e)})"
        )


demo = gr.ChatInterface(
    fn=bank_assistant,
    title="🏦 NUST Bank AI Assistant",
    description="Ask me about NUST Bank's profit rates, term deposits, and policies!",
)


if __name__ == "__main__":
    # HF Spaces expects 0.0.0.0:7860; locally this also works.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

