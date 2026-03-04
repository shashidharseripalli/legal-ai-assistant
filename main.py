"""
main.py
-------
FastAPI application entry point.

Startup sequence (lifespan):
  1. load ChromaDB vectorstore (singleton)
  2. build RAG chain (singleton)
  3. App is ready to serve requests

Run:
  uvicorn main:app --reload --port 8000

Docs available at:
  http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from rag.rag_pipeline import get_chain


# ── Lifespan: warm up once at startup ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⚖️  Starting AI Legal Assistant...")
    print("   Loading ChromaDB vectorstore...")
    print("   Building RAG chain (Mistral via Ollama)...")
    get_chain()   # initializes both singletons
    print("✅  Legal AI Assistant is ready!\n")
    yield
    print("⏹️  Shutting down...")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Legal Assistant",
    description=(
        "RAG-powered assistant for Indian law.\n\n"
        "Sources: Constitution of India, Indian Penal Code (IPC)\n"
        "Model: Ollama Mistral (local)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow frontend (React/Vue) on localhost to call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(chat_router, prefix="/api/v1", tags=["Legal Chat"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "model": "mistral", "vectorstore": "chromadb"}
