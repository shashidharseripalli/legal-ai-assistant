"""ChromaDB vectorstore helpers with automatic local embedding selection."""
import chromadb
import json
import logging
import urllib.error
import urllib.request

from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from app.core.config import settings

logger = logging.getLogger(__name__)
_embeddings = None
_base_vectorstore = None
_upload_vectorstore = None
_base_retriever = None


def _ollama_model_available() -> bool:
    """Check whether the configured model is already installed in the local Ollama."""
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            models = json.loads(response.read().decode("utf-8")).get("models", [])
        configured = settings.ollama_embed_model
        return any(
            model.get("name") == configured
            or model.get("model") == configured
            or model.get("name", "").split(":")[0] == configured.split(":")[0]
            for model in models
        )
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _create_embeddings():
    """Use Ollama if its model is installed; otherwise use a free local HF model."""
    if _ollama_model_available():
        logger.info("Using installed Ollama embedding model '%s'.", settings.ollama_embed_model)
        return OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )

    logger.warning(
        "Ollama/model '%s' is unavailable. Falling back to '%s' on CPU. "
        "The first use downloads model weights and needs internet access.",
        settings.ollama_embed_model,
        settings.hf_embed_model,
    )
    return HuggingFaceEmbeddings(
        model_name=settings.hf_embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = _create_embeddings()
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
