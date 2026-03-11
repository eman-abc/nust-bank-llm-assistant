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
