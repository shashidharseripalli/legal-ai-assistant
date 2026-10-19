"""
main.py
-------
FastAPI application entry point.

Startup sequence (lifespan):
  1. Build RAG chain (connects to Groq API + loads ChromaDB)
  2. App is ready to serve requests

Run:
  uvicorn main:app --reload --port 8000

Docs:
  http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from rag.rag_pipeline import get_chain, get_advice_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting AI Legal Assistant...")
    print("Loading ChromaDB vectorstore...")
    print("Connecting to Groq API (Llama-3.1-8b-instant)...")
    try:
        get_chain()
        get_advice_chain()
        print("Legal AI Assistant is ready!\n")
    except Exception as exc:
        print(f"Startup warning: {exc}")
        print("App started, but chat endpoints require a valid GROQ_API_KEY.\n")
    yield
    print("Shutting down...")


app = FastAPI(
    title="AI Legal Assistant",
    description=(
        "RAG-powered assistant for Indian law.\n\n"
        "Sources: legal documents added to the local RAG corpus\n"
        "LLM: Groq - Llama-3.1-8b-instant (free API)\n"
        "Embeddings: Ollama - nomic-embed-text (local)"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1", tags=["Legal Chat"])
app.include_router(documents_router, prefix="/api/v1", tags=["Documents"])


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "llm": "llama-3.1-8b-instant (groq)",
        "embeddings": "nomic-embed-text (ollama)",
        "vectorstore": "chromadb",
    }
