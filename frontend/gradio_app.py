import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")
UPLOAD_POLL_INTERVAL_SECONDS = 1.0
UPLOAD_STATUS_TIMEOUT_SECONDS = 120

WELCOME_MSG = """### 🏦 Welcome to NUST Bank Support
I can help you with:
*   **Account Opening** (Roshan Digital, Savings, Current)
*   **Loan Queries** (EV Financing, Home Loans)
*   **Banking Policies** & General Support

---
*How can I assist you today?*"""

CUSTOM_CSS = """
.status-box { border: 1px solid #ddd; background: #f9f9f9; padding: 10px; border-radius: 8px; }
footer {visibility: hidden}
"""

def _format_task_status(task_payload: Dict[str, Any]) -> str:
    state = task_payload.get("state", "UNKNOWN")
    result = task_payload.get("result") or {}
    if state == "SUCCESS":
        return f"✅ Indexed {result.get('points_upserted', 0)} chunks from document."
    if state == "FAILURE":
        return f"❌ Upload Failed: {task_payload.get('error', 'Unknown error')}"
    if state == "STARTED":
        return "⏳ Processing in background..."
    return f"Status: {state}"

def process_uploaded_file(file_path: Any):
    if not file_path:
        yield "Error: No file selected."
        return
    file_path = str(file_path[0] if isinstance(file_path, (list, tuple)) else file_path)
    try:
        with open(file_path, "rb") as h:
            files = {"file": (Path(file_path).name, h)}
            response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=30)
        response.raise_for_status()
        task_id = response.json().get("task_id")
        yield "⬆️ Uploading..."
        deadline = time.time() + UPLOAD_STATUS_TIMEOUT_SECONDS
        while time.time() < deadline:
            status_res = requests.get(f"{API_BASE_URL}/tasks/{task_id}", timeout=15)
            task = status_res.json()
            yield _format_task_status(task)
            if task.get("state") in {"SUCCESS", "FAILURE"}: return
            time.sleep(UPLOAD_POLL_INTERVAL_SECONDS)
    except Exception as exc:
        yield f"⚠️ Server Error: {exc}"

def _chat_respond(message: str, history: List[Dict[str, str]]):
    history = list(history or [])
    if not (message or "").strip(): return history, ""
    
    # User message
    history.append({"role": "user", "content": message})
    
    try:
        response = requests.post(f"{API_BASE_URL}/chat", json={"user_query": message}, timeout=45)
        response.raise_for_status()
        data = response.json()
        reply = data.get("final_response", "Error generating response.")
        
        is_safe = data.get("is_safe", True)
        context_used = data.get("context_used", False)
        debug_footer = f"\n\n*🛡️ Safe: {is_safe} | 📚 Context: {context_used}*"
        
        # Assistant message
        history.append({"role": "assistant", "content": reply + debug_footer})
    except Exception as exc:
        history.append({"role": "assistant", "content": f"🚨 Connection Error: {exc}"})
        
    return history, ""

def build_demo():
    with gr.Blocks(title="NUST Bank | Customer Support Hub") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Management")
                status_box = gr.Textbox(label="System Status", value="🟢 All systems operational", interactive=False)
                upload = gr.File(label="Update PDF", file_types=[".pdf"])
                update_btn = gr.Button("Index Document", variant="secondary")
                gr.Markdown("---")
                gr.Markdown("### NUST Bank Assistant")

            with gr.Column(scale=3):
                gr.Markdown("## 💬 Virtual Assistant")
                # Removed 'type' and other problematic args for Gradio 6.10 compatibility
                chatbot = gr.Chatbot(height=600)
                with gr.Row():
                    user_in = gr.Textbox(placeholder="Ask a question...", scale=9)
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                gr.Examples(
                    examples=[
                        "What are the requirements for a Roshan Digital Account?",
                        "Tell me about the NUST-Green EV Financing scheme.",
                        "How do I open an account online?"
                    ],
                    inputs=user_in,
                    label="Quick Questions"
                )

        update_btn.click(fn=process_uploaded_file, inputs=[upload], outputs=[status_box])
        send_btn.click(fn=_chat_respond, inputs=[user_in, chatbot], outputs=[chatbot, user_in])
        user_in.submit(fn=_chat_respond, inputs=[user_in, chatbot], outputs=[chatbot, user_in])
        
        # Initial Welcome Message using Dictionary format
        demo.load(lambda: [{"role": "assistant", "content": WELCOME_MSG}], None, chatbot)

    return demo

if __name__ == "__main__":
    app = build_demo()
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CUSTOM_CSS
    )
