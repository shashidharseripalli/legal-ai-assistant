"""
app/core/config.py
------------------
Central settings loaded from .env using Pydantic BaseSettings.
Import `settings` anywhere in the project — never hard-code values.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "mistral"
    ollama_embed_model: str = "nomic-embed-text"

    # ChromaDB
    chroma_persist_dir: str = "vectorstore"
    chroma_collection_name: str = "legal_docs"

    # Retrieval
    retriever_k: int = 5
    retriever_fetch_k: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
