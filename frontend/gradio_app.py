import os
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

# --- Paths & env ---
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1]


load_dotenv(dotenv_path=_repo_root() / ".env", override=False)

_retrieval_lock = threading.Lock()
_state: Dict[str, Any] = {
    "embed_model": None,
    "index": None,
    "text_mapping": None,
    "index_path": None,
    "mapping_path": None,
}


def _artifact_paths() -> tuple[Path, Path]:
    root = _repo_root()
    preferred_index = root / "data" / "processed" / "bank_faiss.index"
    preferred_mapping = root / "data" / "processed" / "text_mapping.pkl"
    if preferred_index.exists() and preferred_mapping.exists():
        return preferred_index, preferred_mapping
    cwd_index = Path.cwd() / "bank_faiss.index"
    cwd_mapping = Path.cwd() / "text_mapping.pkl"
    return cwd_index, cwd_mapping


def _normalize_mapping(raw: Any) -> Dict[int, Dict[str, Any]]:
    if isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}
    if isinstance(raw, list):
        out: Dict[int, Dict[str, Any]] = {}
        for i, x in enumerate(raw):
            if isinstance(x, dict):
                out[i] = x
            else:
                out[i] = {
                    "chunk_text": str(x),
                    "question": "",
                    "sheet": "",
                    "topic": "",
                    "source_row_index": i,
                }
        return out
    raise TypeError("text_mapping must be dict or list")


def ensure_retrieval_loaded() -> None:
    if _state["embed_model"] is not None:
        return
    index_path, mapping_path = _artifact_paths()
    if not index_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(
            "Missing retrieval artifacts. Expected either:\n"
            f"- {index_path}\n- {mapping_path}\n\n"
            "Generate them via `notebooks/02_indexing.ipynb` or copy files into place."
        )
    _state["embed_model"] = SentenceTransformer("all-MiniLM-L6-v2")
    _state["index"] = faiss.read_index(str(index_path))
    with open(mapping_path, "rb") as f:
        _state["text_mapping"] = _normalize_mapping(pickle.load(f))
    _state["index_path"] = index_path
    _state["mapping_path"] = mapping_path


def _persist_retrieval() -> None:
    ip, mp = _state["index_path"], _state["mapping_path"]
    if ip is None or mp is None:
        return
    faiss.write_index(_state["index"], str(ip))
    with open(mp, "wb") as f:
        pickle.dump(_state["text_mapping"], f)


def _next_chunk_id(mapping: Dict[int, Any]) -> int:
    if not mapping:
        return 0
    return max(mapping.keys()) + 1


def _read_uploaded_text(file_path: str) -> str:
    p = Path(file_path)
    suffix = p.suffix.lower()
    if suffix == ".txt":
        return p.read_text(encoding="utf-8", errors="replace")
    if suffix == ".csv":
        if pd is None:
            raise RuntimeError("pandas is required to read CSV uploads. pip install pandas")
        df = pd.read_csv(file_path)
        return df.to_csv(index=False)
    raise ValueError(f"Unsupported file type: {suffix}")


def _normalize_upload_path(file_path: Any) -> Optional[str]:
    if file_path is None:
        return None
    if isinstance(file_path, (list, tuple)):
        return str(file_path[0]) if file_path else None
    return str(file_path)


def process_uploaded_file(file_path: Any) -> str:
    """Ingest upload: chunk → embed → extend FAISS + text_mapping → persist."""
    file_path = _normalize_upload_path(file_path)
    if not file_path:
        return "Error: No file uploaded. Choose a .txt or .csv file first."
    try:
        with _retrieval_lock:
            ensure_retrieval_loaded()
            text = _read_uploaded_text(file_path)
            if not text.strip():
                return "Error: File is empty."

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_text(text)
            if not chunks:
                return "Error: No chunks produced from file content."

            embed_model = _state["embed_model"]
            index = _state["index"]
            text_mapping: Dict[int, Dict[str, Any]] = _state["text_mapping"]

            vectors = embed_model.encode(chunks, show_progress_bar=False)
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.shape[1] != index.d:
                return (
                    f"Error: Embedding dimension {arr.shape[1]} does not match index ({index.d})."
                )

            start_id = _next_chunk_id(text_mapping)
            fname = Path(file_path).name
            for i, chunk in enumerate(chunks):
                text_mapping[start_id + i] = {
                    "chunk_text": chunk,
                    "question": "Uploaded policy",
                    "sheet": "User Upload",
                    "topic": fname,
                    "source_row_index": -1,
                }

            index.add(arr)
            _state["index"] = index
            _state["text_mapping"] = text_mapping
            _persist_retrieval()

        return f"Successfully added {len(chunks)} chunks to the database."
    except Exception as e:
        return f"Error: {e}"


def _format_numbers_bold(text: str) -> str:
    import re

    return re.sub(r"(\d[\d,./-]*)", r"**\1**", text)


def _chunk_context_text(entry: Any) -> str:
    if isinstance(entry, dict):
        return (entry.get("chunk_text") or "").strip()
    return str(entry)


def _client() -> InferenceClient:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it as an environment variable (local) or a Space secret (HF Spaces)."
        )
    model_id = os.environ.get("HF_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    return InferenceClient(model=model_id, token=token)


def bank_assistant(message: str, history: Optional[Any] = None) -> str:
    with _retrieval_lock:
        ensure_retrieval_loaded()
        embed_model = _state["embed_model"]
        index = _state["index"]
        text_mapping = _state["text_mapping"]

    msg_lower = (message or "").strip().lower()
    if any(w in msg_lower for w in ["hi", "hello", "hey", "salam", "assalam"]) or "thank" in msg_lower:
        return "Hello! Welcome to NUST Bank. How can I assist you with our services today?"

    query_vector = embed_model.encode([message])
    distances, indices = index.search(query_vector, k=3)
    context = "\n".join(
        [_chunk_context_text(text_mapping[int(i)]) for i in indices[0]]
    )

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
            "System Error: ensure `HF_TOKEN` is set, the model is under 6B (default: Qwen2.5-3B-Instruct), "
            "and your account can access it via Inference API. "
            f"({str(e)})"
        )


def _chat_respond(
    message: str, history: Optional[List[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], str]:
    """Gradio 6 Chatbot uses messages: list of {role, content} dicts."""
    history = list(history or [])
    if not (message or "").strip():
        return history, ""
    reply = bank_assistant(message, history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, ""


def build_demo() -> gr.Blocks:
    try:
        ensure_retrieval_loaded()
        startup_status = "Knowledge base loaded."
    except FileNotFoundError as e:
        startup_status = f"Warning: {e}"

    with gr.Blocks(theme=gr.themes.Soft(), title="NUST Bank Customer Support Hub") as demo:
        gr.Markdown("# NUST Bank Customer Support Hub")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Knowledge Base Manager")
                upload = gr.File(
                    file_types=[".txt", ".csv"],
                    label="Upload New Bank Policy",
                    type="filepath",
                )
                update_btn = gr.Button("Update Knowledge Base")
                status = gr.Textbox(
                    label="System Status",
                    value=startup_status,
                    interactive=False,
                    lines=3,
                )
            with gr.Column():
                gr.Markdown("### Chat")
                chatbot = gr.Chatbot(height=500, label="Assistant")
                user_in = gr.Textbox(
                    label="Your message",
                    placeholder="Ask about policies, rates, or services…",
                    lines=2,
                )
                send_btn = gr.Button("Send")

        update_btn.click(
            fn=process_uploaded_file,
            inputs=[upload],
            outputs=[status],
        )

        send_btn.click(
            fn=_chat_respond,
            inputs=[user_in, chatbot],
            outputs=[chatbot, user_in],
        )
        user_in.submit(
            fn=_chat_respond,
            inputs=[user_in, chatbot],
            outputs=[chatbot, user_in],
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
