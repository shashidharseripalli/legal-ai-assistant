"""
app/core/config.py
------------------
Central settings loaded from .env using Pydantic BaseSettings.
Import `settings` anywhere and avoid hard-coded values.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Groq (LLM)
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"

    # Ollama (embeddings)
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # ChromaDB
    chroma_persist_dir: str = "vectorstore"
    chroma_collection_name: str = "legal_docs"
    chroma_upload_collection_name: str = "legal_uploads"

    # Retrieval
    retriever_k: int = 5
    retriever_fetch_k: int = 20
    upload_retriever_k: int = 8
    upload_retriever_fetch_k: int = 30

    # Uploads
    upload_dir: str = "data/uploads"
    max_upload_mb: int = 15

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
