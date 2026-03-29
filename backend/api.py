from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from fastapi.responses import StreamingResponse
import json
import asyncio

# Import your compiled LangGraph
from backend.orchestrator import bank_bot


# Setup basic logging for server monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NUST Bank Agentic API",
    description="REST API gateway for the LangGraph-powered banking assistant.",
    version="1.0.0"
)

# Allow CORS so Gradio (or a future Flutter app) can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define strict input/output schemas using Pydantic
class ChatRequest(BaseModel):
    user_query: str

class ChatResponse(BaseModel):
    final_response: str
    is_safe: bool
    scrubbed_query: str
    context_used: bool

@app.get("/health")
def health_check():
    """System monitor endpoint."""
    return {"status": "operational", "system": "NUST Bank LangGraph Orchestrator"}

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Main routing endpoint for customer queries."""
    try:
        if not request.user_query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        # 1. Initialize the LangGraph State
        initial_state = {"user_query": request.user_query}
        
        # 2. Invoke the Orchestrator
        result = bank_bot.invoke(initial_state)

        # 3. Package the output
        return ChatResponse(
            final_response=result.get("final_response", "Error generating response."),
            is_safe=result.get("is_safe", False),
            scrubbed_query=result.get("scrubbed_query", ""),
            context_used=bool(result.get("retrieved_context"))
        )
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# --- KNOWLEDGE BASE INGESTION LOGIC ---

# Load the embedding model globally so it doesn't reload on every upload
try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    embed_model = None

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Handles dynamic document ingestion: Chunking, Embedding, and FAISS indexing."""
    try:
        # 1. Read the uploaded file into memory
        content = await file.read()
        text = content.decode("utf-8", errors="replace")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="File is empty.")

        # 2. Setup Paths
        root_dir = Path(__file__).resolve().parents[1]
        index_path = root_dir / "data" / "processed" / "bank_faiss.index"
        mapping_path = root_dir / "data" / "processed" / "text_mapping.pkl"

        if not index_path.exists() or not mapping_path.exists():
            raise HTTPException(status_code=500, detail="FAISS index or mapping file not found on server.")

        # 3. Chunk the text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text chunks from file.")

        # 4. Generate Embeddings
        vectors = embed_model.encode(chunks, show_progress_bar=False)
        arr = np.asarray(vectors, dtype=np.float32)

        # 5. Load existing database, update, and save
        index = faiss.read_index(str(index_path))
        with open(mapping_path, "rb") as f:
            text_mapping = pickle.load(f)

        # Normalize keys to integers
        text_mapping = {int(k): v for k, v in text_mapping.items()}
        start_id = max(text_mapping.keys()) + 1 if text_mapping else 0

        # Append new chunks to mapping
        for i, chunk in enumerate(chunks):
            text_mapping[start_id + i] = {
                "chunk_text": chunk,
                "question": "Uploaded policy",
                "sheet": "User Upload",
                "topic": file.filename,
                "source_row_index": -1,
            }

        # Add to FAISS and persist to disk
        index.add(arr)
        faiss.write_index(index, str(index_path))
        with open(mapping_path, "wb") as f:
            pickle.dump(text_mapping, f)

        logger.info(f"Successfully ingested {file.filename} ({len(chunks)} chunks).")
        return {"message": f"✅ Successfully ingested '{file.filename}'. Added {len(chunks)} chunks to vector memory."}

    except Exception as e:
        logger.error(f"Upload processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE (Server-Sent Events) endpoint.
    This allows the UI to 'type out' the response character by character.
    """
    async def event_generator():
        # 1. Run the LangGraph logic
        initial_state = {"user_query": request.user_query}
        result = bank_bot.invoke(initial_state)
        full_text = result.get("final_response", "")

        # 2. Break the text into 'chunks' to simulate streaming
        # What is astream? If bank_bot supports an .astream() method, it provides async streaming of the model's output
        # instead of waiting for the full response (like .invoke). If we hook into bank_bot.astream now, we can yield each
        # chunk generated as soon as it's available from the backend LLM/graph, providing real token-wise or chunk-wise streaming.
        # Example usage:
        # async for chunk in bank_bot.astream(initial_state):
        #     text = chunk.get("final_response", "")
        #     yield text
        #     await asyncio.sleep(0.01)  # optional delay for smoothness
        # This gives true streaming to the UI as the model generates the response, not a simulated one.
        words = full_text.split(" ")
        for word in words:
            yield f"{word} "
            await asyncio.sleep(0.04) # Smooth 'typing' delay

    return StreamingResponse(event_generator(), media_type="text/plain")