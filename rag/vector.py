"""
rag/vector.py
-------------
Singleton ChromaDB client + MMR retriever.

Why singleton?
  - ChromaDB opens an SQLite file lock on Windows; re-opening it per request
    causes errors and adds 300–800 ms overhead.
  - The vectorstore is initialized ONCE at FastAPI startup via lifespan.
"""

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import settings

# ── module-level singletons ──────────────────────────────────────────────────
_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    """
    Return (and lazily create) the ChromaDB vectorstore singleton.
    Safe to call multiple times — only opens the DB once.
    """
    global _vectorstore
    if _vectorstore is None:
        embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,   # nomic-embed-text, NOT mistral
        )
        _vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=embeddings,
            collection_name=settings.chroma_collection_name,
        )
    return _vectorstore


def get_retriever() -> VectorStoreRetriever:
    """
    MMR retriever — fetches 20 candidates then re-ranks for diversity,
    returning the top-k most relevant AND non-redundant chunks.

    lambda_mult:
        1.0 = pure relevance (may return near-duplicates)
        0.0 = pure diversity
        0.6 = good balance for legal documents
    """
    return get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retriever_k,
            "fetch_k": settings.retriever_fetch_k,
            "lambda_mult": 0.6,
        },
    )
