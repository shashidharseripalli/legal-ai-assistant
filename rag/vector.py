"""
rag/vector.py
-------------
ChromaDB vectorstore helpers for:
- Base legal corpus
- User-uploaded documents
"""

import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from app.core.config import settings

_embeddings = None
_base_vectorstore = None
_upload_vectorstore = None
_base_retriever = None


def _get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )
    return _embeddings


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_vectorstore() -> Chroma:
    """Base legal corpus vectorstore."""
    global _base_vectorstore
    if _base_vectorstore is None:
        _base_vectorstore = Chroma(
            client=_get_client(),
            collection_name=settings.chroma_collection_name,
            embedding_function=_get_embeddings(),
        )
    return _base_vectorstore


def get_upload_vectorstore() -> Chroma:
    """User-uploaded documents vectorstore."""
    global _upload_vectorstore
    if _upload_vectorstore is None:
        _upload_vectorstore = Chroma(
            client=_get_client(),
            collection_name=settings.chroma_upload_collection_name,
            embedding_function=_get_embeddings(),
        )
    return _upload_vectorstore


def get_retriever(search_filter: dict | None = None):
    """Retriever for base legal corpus."""
    global _base_retriever
    if search_filter is not None:
        return get_vectorstore().as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.retriever_k,
                "fetch_k": settings.retriever_fetch_k,
                "filter": search_filter,
            },
        )

    if _base_retriever is None:
        _base_retriever = get_vectorstore().as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.retriever_k,
                "fetch_k": settings.retriever_fetch_k,
            },
        )
    return _base_retriever


def get_upload_retriever(search_filter: dict | None = None):
    """Retriever for uploaded documents collection."""
    search_kwargs = {
        "k": settings.upload_retriever_k,
        "fetch_k": settings.upload_retriever_fetch_k,
    }
    if search_filter:
        search_kwargs["filter"] = search_filter

    return get_upload_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )
