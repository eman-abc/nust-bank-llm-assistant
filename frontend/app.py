import streamlit as st
import faiss
import pickle
import torch
import warnings
from transformers import pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# Hide warnings for a clean terminal
warnings.filterwarnings("ignore")

# --- 1. Page Configuration ---
st.set_page_config(page_title="NUST Bank Edge AI", page_icon="🏦", layout="centered")
st.title("🏦 NUST Bank AI Assistant")
st.caption("100% Offline Local Edge AI - Powered by Qwen 1.5B & FAISS")

# --- 2. Load the Local AI (Cached so it only runs once!) ---
@st.cache_resource
def load_local_ai():
    # A. Load Retrieval System
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("data/processed/bank_faiss.index")
    with open("data/processed/text_mapping.pkl", "rb") as f:
        text_mapping = pickle.load(f)

    # B. Load the Local Model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        model_kwargs={
            "quantization_config": quantization_config,
            "device_map": "auto"
        }
    )
    return embed_model, index, text_mapping, generator

with st.spinner("Initializing Local AI Engine... (Takes a minute)"):
    embed_model, index, text_mapping, generator = load_local_ai()

# --- 3. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your secure, offline NUST Bank Assistant. Ask me about our profit rates!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("E.g., What are the profit rates for a 1-year Term Deposit?"):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching records and typing..."):
            query_vector = embed_model.encode([prompt])
            distances, indices = index.search(query_vector, k=3)
            context = "\n".join([text_mapping[i] for i in indices[0]])
            
            # Format prompt for Qwen
            system_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
            
            res = generator(
                system_prompt, 
                max_new_tokens=100, 
                temperature=0.1, 
                do_sample=True,
                return_full_text=False,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            
            raw_answer = res[0]["generated_text"].strip()
            clean_answer = raw_answer.split("Question:")[0].split("\n\n")[0].strip()
            
            st.markdown(clean_answer)
            st.session_state.messages.append({"role": "assistant", "content": clean_answer})