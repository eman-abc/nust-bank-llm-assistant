import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import gradio as gr
import requests

# Point this to your FastAPI server
API_BASE_URL = "http://127.0.0.1:8000/api"

def process_uploaded_file(file_path: Any) -> str:
    """Sends the uploaded file to the FastAPI backend for ingestion."""
    if not file_path:
        return "Error: No file uploaded."
    
    # Normalize path for Gradio
    if isinstance(file_path, (list, tuple)):
        file_path = str(file_path[0]) if file_path else None
    else:
        file_path = str(file_path)

    try:
        # Hit the API's upload endpoint using multipart/form-data
        with open(file_path, "rb") as f:
            file_name = Path(file_path).name
            files = {"file": (file_name, f)}
            response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=30)
            
        response.raise_for_status() # Check for HTTP errors
        
        data = response.json()
        return data.get("message", "Successfully added to the database.")
    
    except requests.exceptions.ConnectionError:
        return "⚠️ Error: Could not connect to backend. Is FastAPI running on port 8000?"
    except Exception as e:
        return f"⚠️ Server Error: {str(e)}"

def _chat_respond(
    message: str, history: Optional[List[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], str]:
    """Sends the user message to the FastAPI backend and updates the chat history."""
    history = list(history or [])
    if not (message or "").strip():
        return history, ""

    # 1. Append the user's message to the UI history immediately
    history.append({"role": "user", "content": message})

    try:
        # 2. Hit the API's chat endpoint with JSON payload
        response = requests.post(
            f"{API_BASE_URL}/chat", 
            json={"user_query": message},
            timeout=45 # Generous timeout for LLM generation
        )
        response.raise_for_status()
        
        data = response.json()
        reply = data.get("final_response", "Error generating response.")
        
        # Optional: Add debug metrics for your Viva presentation
        is_safe = data.get("is_safe", False)
        context_used = data.get("context_used", False)
        debug_footer = f"\n\n---\n*Debug: [Safe: {is_safe} | Context Retrieved: {context_used}]*"
        
        # 3. Append the API's response to the UI history
        history.append({"role": "assistant", "content": reply + debug_footer})
        
    except requests.exceptions.ConnectionError:
        error_msg = "⚠️ Error: Cannot connect to backend. Is the FastAPI server running on port 8000?"
        history.append({"role": "assistant", "content": error_msg})
    except Exception as e:
        history.append({"role": "assistant", "content": f"⚠️ Server Error: {str(e)}"})

    return history, ""


def chat_with_api_stream(user_message, history):
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": ""}) # Placeholder for bot
    
    # Use 'stream=True' in requests
    with requests.post(f"{API_BASE_URL}/chat/stream", 
                       json={"user_query": user_message}, 
                       stream=True) as r:
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                history[-1]["content"] += chunk
                yield history, "" # This updates the UI word-by-word!

def build_demo() -> gr.Blocks:
    startup_status = "UI Ready. Connect to FastAPI backend for data."

    # FIX 1: Removed 'theme=gr.themes.Soft()' from the Blocks constructor
    with gr.Blocks(title="NUST Bank Customer Support Hub") as demo:
        gr.Markdown("# 🏦 NUST Bank Agentic Command Center")
        
        with gr.Row():
            # Left Column: Knowledge Management
            with gr.Column(scale=1):
                gr.Markdown("### 🗄️ Knowledge Base Manager")
                upload = gr.File(
                    file_types=[".txt", ".csv"],
                    label="Upload New Bank Policy",
                    type="filepath",
                )
                update_btn = gr.Button("Update Knowledge Base", variant="secondary")
                status = gr.Textbox(
                    label="System Status",
                    value=startup_status,
                    interactive=False,
                    lines=3,
                )
                
                gr.Markdown("---")
                gr.Markdown("### ⚙️ Architecture Profile\n- **Frontend:** Gradio Thin Client\n- **Backend:** FastAPI Microservice\n- **Logic:** LangGraph Orchestrator")

            # Right Column: Chat Interface
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Live Agent Chat")
                
                # FIX 2: Removed 'type="messages"' because it is now the Gradio 6 default
                chatbot = gr.Chatbot(height=500, label="Assistant")
                
                with gr.Row():
                    user_in = gr.Textbox(
                        label="Your message",
                        placeholder="e.g., What are the requirements for a Roshan Digital Account?",
                        lines=1,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

        # Wire up the events
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
    # FIX 3: Moved the theme parameter into the launch method
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, theme=gr.themes.Soft())
    # asyncio is a Python standard library module that provides support for writing concurrent code using the async/await syntax.
    # It is used for asynchronous programming, allowing you to handle tasks such as I/O operations (like network requests or file reading)
    # without blocking the execution of your program. With asyncio, you can run multiple tasks seemingly at the same time
    # (i.e., concurrently), which is particularly useful for applications that spend a lot of time waiting for external operations,
    # such as web servers, chatbots, or any service needing efficient handling of many connections or requests simultaneously.