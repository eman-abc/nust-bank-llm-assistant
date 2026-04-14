"""
LLM synthesis node: Robust Hugging Face Inference with Citations.
Handles both Chat and Text-Generation fallbacks.
"""

from __future__ import annotations
import logging
from typing import Dict
from huggingface_hub import InferenceClient
from backend.config import get_settings
from backend.state import AgentState

logger = logging.getLogger(__name__)

# The new merged model is at the top!
_MODELS = [
    "eman-abc/gemma-3-4b-bank-merged",       # 1. Your new FUSED model
    "eman-abc/gemma-3-4b-bank-finetuned",    # 2. Original adapter
    "Qwen/Qwen2.5-7B-Instruct",              # 3. High quality modern chat model
    "mistralai/Mistral-7B-Instruct-v0.2",    # 4. Reliable backup
]
_MAX_TOKENS = 1024

_SYSTEM_PROMPT = (
    "You are a professional NUST Bank customer support agent. "
    "Use ONLY the Context provided to answer the question. If the Context is missing, "
    "politely inform the user you cannot find that info in bank records.\n"
)

def run_synthesizer(state: AgentState) -> Dict[str, str]:
    # Check if a Guardrail already blocked this and set a refusal message
    if state.get("is_safe") is False and state.get("final_response"):
        return {"final_response": state.get("final_response")}

    context = (state.get("selected_context") or state.get("retrieved_context") or "").strip()
    query = (state.get("normalized_query") or state.get("scrubbed_query") or "").strip()
    
    # If it's a greeting, we can answer without context!
    intent = state.get("query_intent", "banking_query")
    if not context and intent != "greeting":
        return {"final_response": "I don't have enough verified information to answer that."}

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    
    flat_prompt = f"<|system|>\n{_SYSTEM_PROMPT}\n<|user|>\nContext:\n{context}\n\nQuestion: {query}\n<|assistant|>\n"

    token = get_settings().hf_token
    if not token:
        return {"final_response": "Server Error: HF_TOKEN missing."}

    for model_id in _MODELS:
        try:
            client = InferenceClient(model=model_id, token=token, timeout=45)
            text = None

            # STRATEGY 1: Try Modern Chat API
            try:
                response = client.chat_completion(messages=messages, max_tokens=_MAX_TOKENS, temperature=0.1)
                text = response.choices[0].message.content
            except Exception as chat_err:
                chat_err_str = str(chat_err).lower()
                if "chat model" in chat_err_str or "not supported" in chat_err_str or "400" in chat_err_str:
                    # STRATEGY 2: Try Legacy Text Generation API
                    text = client.text_generation(flat_prompt, max_new_tokens=_MAX_TOKENS, stop_sequences=["<|"], temperature=0.1)
                else:
                    raise chat_err

            if text:
                final_ans = text.strip()
                # Append Citations (Provenance)
                citations = state.get("citations") or []
                if citations:
                    raw_sources = set(c.get("doc_id", "Unknown Document") for c in citations)
                    friendly_sources = []
                    for s in raw_sources:
                        # Map internal codes or preserve filenames
                        clean_s = s.split("::")[0]
                        name = clean_s.upper()
                        
                        if "NUST_BANK_2026_UPGRADES" in name or "NUST-BANK-2026-UPGRADES" in name:
                            friendly_sources.append("NUST Bank 2026 Strategy Document")
                        elif name == "RDA":
                            friendly_sources.append("NUST Bank Roshan Digital Account Policy")
                        elif name == "WF":
                            friendly_sources.append("NUST Bank Website FAQs")
                        elif len(clean_s) <= 4:
                            friendly_sources.append(f"Bank Policy Manual: {name}")
                        else:
                            # If it's a long string, it's likely a filename, keep it pretty
                            friendly_sources.append(clean_s.replace(".pdf", "").replace("-", " ").replace("_", " ").title())
                    
                    sources = sorted(list(set(friendly_sources)))
                    source_text = "\n\nSources:\n" + "\n".join(f"📍 {s}" for s in sources)
                    final_ans += source_text
                
                return {"final_response": final_ans}

        except Exception as e:
            print(f"[INFO] Model {model_id} status: {e}")
            continue

    return {"final_response": "The bank servers are busy. Please try again soon."}