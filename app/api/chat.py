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

@router.post("/chat", response_model=ChatResponse, summary="Ask a legal question")
async def chat(request: ChatRequest):
    """
    Ask anything about the Constitution of India or IPC.
    Returns a cited answer grounded in the source documents.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chain, retriever = get_chain()

    try:
        source_docs = await asyncio.to_thread(retriever.invoke, request.question)

        # Pass as dict — ChatGroq requires this (unlike plain Ollama LLM)
        answer = await asyncio.to_thread(
            chain.invoke, {"question": request.question}
        )

        sources = [
            SourceDocument(
                source_file=doc.metadata.get("source_file", "unknown"),
                page=doc.metadata.get("page", "unknown"),
            )
            for doc in source_docs
        ]

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
        # Pass as dict — this was the original crash on line 92
        for token in chain.stream({"question": request.question}):
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
        source_docs = await asyncio.to_thread(retriever.invoke, request.problem)

        answer = await asyncio.to_thread(
            chain.invoke, {"problem": request.problem}
        )

        sources = [
            SourceDocument(
                source_file=doc.metadata.get("source_file", "unknown"),
                page=doc.metadata.get("page", "unknown"),
            )
            for doc in source_docs
        ]

        return AdviceResponse(advice=answer, sources=sources)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
