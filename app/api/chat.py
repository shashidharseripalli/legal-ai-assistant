"""
app/api/chat.py
---------------
FastAPI router — three endpoints:

  POST /api/v1/chat          → Full answer + source citations (JSON)
  POST /api/v1/chat/stream   → Token-by-token streaming (plain text)
  POST /api/v1/advice        → Structured legal advice for a described problem

Key fix from Ollama version:
  chain.invoke/stream now receives {"question": ...} dict — not a raw string.
  ChatGroq (unlike Ollama LLM) is a ChatModel and requires dict input.
"""

import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.rag_pipeline import get_chain, get_advice_chain

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str


class AdviceRequest(BaseModel):
    problem: str


class SourceDocument(BaseModel):
    source_file: str
    page: int | str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


class AdviceResponse(BaseModel):
    advice: str
    sources: list[SourceDocument]


# ── POST /chat ────────────────────────────────────────────────────────────────

def _retrieve_documents(retriever, query: str):
    """Return list of docs from retriever (compatible with LangChain versions)."""
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    if hasattr(retriever, "get_documents"):
        return retriever.get_documents(query)
    if hasattr(retriever, "invoke"):
        try:
            return retriever.invoke(query)
        except TypeError:
            return retriever.invoke({"query": query})

    raise RuntimeError("Retriever object has no supported retrieval method")


@router.post("/chat", response_model=ChatResponse, summary="Ask a legal question")
async def chat(request: ChatRequest):
    """
    Ask a question grounded in the indexed legal documents.
    Returns a cited answer grounded in the source documents.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chain, retriever = get_chain()

    try:
        source_docs = await asyncio.to_thread(_retrieve_documents, retriever, request.question)

        # Pass as dict — ChatGroq requires this (unlike plain Ollama LLM)
        answer = await asyncio.to_thread(
            chain.invoke, {"question": request.question}
        )

        sources = []
        for doc in source_docs:
            source_file = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "unknown")
            if str(source_file).lower().endswith(".pdf") and isinstance(page, int):
                page += 1
            sources.append(SourceDocument(source_file=source_file, page=page))

        return ChatResponse(answer=answer, sources=sources)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── POST /chat/stream ─────────────────────────────────────────────────────────

@router.post("/chat/stream", summary="Stream answer token by token")
async def chat_stream(request: ChatRequest):
    """
    Same as /chat but streams tokens as they arrive from Groq.
    Groq is fast — expect ~2-5s to first token vs 15-40s with local Ollama.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chain, _ = get_chain()

    async def token_generator():
        stream_iter = chain.stream({"question": request.question})

        if hasattr(stream_iter, "__aiter__"):
            async for token in stream_iter:
                yield token
        else:
            for token in stream_iter:
                yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


# ── POST /advice ──────────────────────────────────────────────────────────────

@router.post("/advice", response_model=AdviceResponse, summary="Get legal advice for a problem")
async def advice(request: AdviceRequest):
    """
    Describe your legal problem in plain language.
    Returns structured advice: case type, relevant laws, steps to take, warnings.
    """
    if not request.problem.strip():
        raise HTTPException(status_code=400, detail="Problem description cannot be empty.")

    chain, retriever = get_advice_chain()

    try:
        source_docs = await asyncio.to_thread(_retrieve_documents, retriever, request.problem)

        answer = await asyncio.to_thread(
            chain.invoke, {"problem": request.problem}
        )

        sources = []
        for doc in source_docs:
            source_file = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "unknown")
            if str(source_file).lower().endswith(".pdf") and isinstance(page, int):
                page += 1
            sources.append(SourceDocument(source_file=source_file, page=page))

        return AdviceResponse(advice=answer, sources=sources)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
