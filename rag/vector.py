"""
rag/vector.py
-------------
ChromaDB vectorstore singleton + MMR retriever.

Embeddings: Ollama nomic-embed-text (local, ~200MB RAM — fine)
Storage:    ChromaDB persistent on disk (survives restarts)
Retriever:  MMR (Maximum Marginal Relevance) — avoids returning
            5 nearly-identical chunks; picks diverse relevant results

Run scripts/ingest_documents.py ONCE before starting the server.
"""

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from app.core.config import settings

# ── Singletons ────────────────────────────────────────────────────────────────
_vectorstore = None
_retriever   = None


def get_vectorstore() -> Chroma:
    """
    Build (or return cached) ChromaDB vectorstore.
    Uses Ollama nomic-embed-text for embeddings — unchanged from original.
    """
    global _vectorstore
    if _vectorstore is None:
        embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,   # nomic-embed-text
        )

        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

        _vectorstore = Chroma(
            client=client,
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
        )

    return _vectorstore


def get_retriever():
    """
    MMR retriever: fetches fetch_k candidates, returns k diverse results.

    WHY MMR over simple similarity search:
      Legal docs repeat the same preamble/definitions across many pages.
      MMR picks the most RELEVANT but also DIVERSE chunks — better context.
    """
    global _retriever
    if _retriever is None:
        vs = get_vectorstore()
        _retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":       settings.retriever_k,        # chunks returned
                "fetch_k": settings.retriever_fetch_k,  # candidates to consider
            },
        )
    return _retriever
