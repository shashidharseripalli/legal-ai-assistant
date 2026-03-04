"""
app/api/chat.py
---------------
FastAPI router with two endpoints:

  POST /api/v1/chat
    - Returns full answer + source citations as JSON
    - Uses asyncio.to_thread() so the blocking Ollama call
      doesn't freeze the FastAPI event loop

  POST /api/v1/chat/stream
    - Streams tokens one-by-one as plain text (Server-Sent Events style)
    - Much better UX when Mistral takes 15–40s locally
"""

import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.rag_pipeline import get_chain

router = APIRouter()


# ── Request / Response schemas ───────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


class SourceDocument(BaseModel):
    source_file: str
    page: int | str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


# ── POST /chat ────────────────────────────────────────────────────────────────
@router.post("/chat", response_model=ChatResponse, summary="Ask a legal question")
async def chat(request: ChatRequest):
    """
    Accepts a natural-language question about Indian law.
    Returns an answer grounded in the Constitution / IPC,
    with citations to the source documents.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chain, retriever = get_chain()

    try:
        # 1. Retrieve relevant chunks (fast — vector search)
        source_docs = await asyncio.to_thread(retriever.invoke, request.question)

        # 2. Run full RAG chain (slow — LLM inference, 15–40s locally)
        answer = await asyncio.to_thread(chain.invoke, request.question)

        # 3. Build source list from metadata
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
    Same as /chat but streams tokens progressively.
    Use this in your frontend for real-time output.

    Response: text/plain stream (connect with fetch + ReadableStream or EventSource)
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chain, _ = get_chain()

    async def token_generator():
        # chain.stream() yields tokens as they are produced by the LLM
        for token in chain.stream(request.question):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")
