"""
app/core/config.py
------------------
Central settings loaded from .env using Pydantic BaseSettings.
Import `settings` anywhere — never hard-code values.

LLM  → Groq API (cloud, free tier, no RAM needed)
Embed → Ollama nomic-embed-text (local, tiny, ~200MB RAM)
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Groq (LLM) ────────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"

    # ── Ollama (Embeddings only — kept local) ─────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "vectorstore"
    chroma_collection_name: str = "legal_docs"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retriever_k: int = 5
    retriever_fetch_k: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
