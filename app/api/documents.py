"""
app/api/documents.py
--------------------
Upload user documents and analyze legal impact against a problem description.
"""

import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from app.core.config import settings
from rag.rag_pipeline import run_impact_analysis
from rag.vector import get_retriever, get_upload_retriever, get_upload_vectorstore

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


class UploadItem(BaseModel):
    document_id: str
    file_name: str
    chunks_indexed: int


class UploadResponse(BaseModel):
    session_id: str
    uploaded: list[UploadItem]


class SourceDocument(BaseModel):
    source_file: str
    page: int | str


class ImpactRequest(BaseModel):
    problem: str
    session_id: str
    scope: str = "hybrid"  # legal_corpus_only | uploaded_docs_only | hybrid
    document_ids: list[str] | None = None


class ImpactResponse(BaseModel):
    analysis: str
    sources: list[SourceDocument]


def _retrieve_documents(retriever, query: str):
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


def _save_uploaded_file(session_id: str, upload_file: UploadFile, content: bytes) -> Path:
    session_dir = Path(settings.upload_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / upload_file.filename
    file_path.write_bytes(content)
    return file_path


def _load_to_documents(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"page": 1})]

    return []


def _to_sources(docs: list) -> list[SourceDocument]:
    seen = set()
    out = []
    for doc in docs:
        source_file = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "unknown")
        key = (source_file, page)
        if key in seen:
            continue
        seen.add(key)
        out.append(SourceDocument(source_file=source_file, page=page))
    return out


@router.post("/documents/upload", response_model=UploadResponse, summary="Upload legal documents")
async def upload_documents(
    session_id: str = Form(...),
    files: list[UploadFile] = File(...),
):
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )

    vectorstore = get_upload_vectorstore()
    uploaded_items: list[UploadItem] = []

    for upload_file in files:
        if not upload_file.filename:
            raise HTTPException(status_code=400, detail="File name is missing")

        suffix = Path(upload_file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {upload_file.filename}. Allowed: pdf, txt",
            )

        content = await upload_file.read()
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {upload_file.filename}. Max {settings.max_upload_mb} MB",
            )

        file_path = _save_uploaded_file(session_id, upload_file, content)
        docs = await asyncio.to_thread(_load_to_documents, file_path)
        if not docs:
            raise HTTPException(status_code=400, detail=f"Could not parse file: {upload_file.filename}")

        upload_id = str(uuid4())

        for doc in docs:
            doc.metadata["source_file"] = upload_file.filename
            doc.metadata["source_type"] = "uploaded"
            doc.metadata["session_id"] = session_id
            doc.metadata["upload_id"] = upload_id

        chunks = splitter.split_documents(docs)
        ids = [str(uuid4()) for _ in chunks]
        await asyncio.to_thread(vectorstore.add_documents, chunks, ids)

        uploaded_items.append(
            UploadItem(
                document_id=upload_id,
                file_name=upload_file.filename,
                chunks_indexed=len(chunks),
            )
        )

    return UploadResponse(session_id=session_id, uploaded=uploaded_items)


@router.post("/documents/impact", response_model=ImpactResponse, summary="Analyze document impact")
async def analyze_impact(request: ImpactRequest):
    if not request.problem.strip():
        raise HTTPException(status_code=400, detail="Problem description cannot be empty")

    valid_scopes = {"legal_corpus_only", "uploaded_docs_only", "hybrid"}
    if request.scope not in valid_scopes:
        raise HTTPException(status_code=400, detail=f"scope must be one of: {', '.join(sorted(valid_scopes))}")

    docs = []

    if request.scope in {"legal_corpus_only", "hybrid"}:
        base_retriever = get_retriever()
        base_docs = await asyncio.to_thread(_retrieve_documents, base_retriever, request.problem)
        docs.extend(base_docs)

    if request.scope in {"uploaded_docs_only", "hybrid"}:
        upload_retriever = get_upload_retriever(search_filter={"session_id": request.session_id})
        upload_docs = await asyncio.to_thread(_retrieve_documents, upload_retriever, request.problem)

        if request.document_ids:
            allowed = set(request.document_ids)
            upload_docs = [d for d in upload_docs if d.metadata.get("upload_id") in allowed]

        docs.extend(upload_docs)

    if not docs:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Upload documents first or change scope.",
        )

    analysis = await asyncio.to_thread(run_impact_analysis, request.problem, docs)
    sources = _to_sources(docs)

    return ImpactResponse(analysis=analysis, sources=sources)
