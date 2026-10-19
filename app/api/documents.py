from pathlib import Path
from uuid import uuid4
import asyncio

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
    scope: str = "hybrid"
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


def _safe_filename(filename: str) -> str:
    # Strip any client-supplied path; only use a single filename under the session dir.
    return Path(filename.replace("\\", "/")).name


def _save_uploaded_file(session_id: str, upload_file: UploadFile, content: bytes) -> Path:
    session_dir = Path(settings.upload_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / _safe_filename(upload_file.filename or "upload")
    file_path.write_bytes(content)
    return file_path


def _load_to_documents(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"page": 1})] if text.strip() else []
    return []


def _to_sources(docs: list) -> list[SourceDocument]:
    seen = set()
    out = []
    for doc in docs:
        source_file = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "unknown")
        if str(source_file).lower().endswith(".pdf") and isinstance(page, int):
            page += 1
        key = (source_file, page)
        if key not in seen:
            seen.add(key)
            out.append(SourceDocument(source_file=source_file, page=page))
    return out


@router.post("/documents/upload", response_model=UploadResponse, summary="Upload legal documents")
async def upload_documents(session_id: str = Form(...), files: list[UploadFile] = File(...)):
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    # Session identifiers are user-controlled input; restrict them to safe path components.
    if len(session_id) > 128 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in session_id):
        raise HTTPException(status_code=400, detail="session_id may contain only letters, numbers, '_' and '-'")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ".", " " ]
    )
    vectorstore = get_upload_vectorstore()
    uploaded_items: list[UploadItem] = []

    for upload_file in files:
        original_name = upload_file.filename
        if not original_name:
            raise HTTPException(status_code=400, detail="File name is missing")
        safe_name = _safe_filename(original_name)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {safe_name}. Allowed: pdf, txt")

        content = await upload_file.read()
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if not content:
            raise HTTPException(status_code=400, detail=f"File is empty: {safe_name}")
        if len(content) > max_bytes:
            raise HTTPException(status_code=400, detail=f"File too large: {safe_name}. Max {settings.max_upload_mb} MB")

        file_path = _save_uploaded_file(session_id, upload_file, content)
        try:
            docs = await asyncio.to_thread(_load_to_documents, file_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse file: {safe_name}") from exc
        if not docs or not any(doc.page_content.strip() for doc in docs):
            raise HTTPException(status_code=400, detail=f"No readable text found in file: {safe_name}")

        upload_id = str(uuid4())
        for doc in docs:
            doc.metadata["source_file"] = safe_name
            doc.metadata["source_type"] = "uploaded"
            doc.metadata["session_id"] = session_id
            doc.metadata["upload_id"] = upload_id

        chunks = splitter.split_documents(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail=f"No indexable text found in file: {safe_name}")
        ids = [str(uuid4()) for _ in chunks]
        await asyncio.to_thread(vectorstore.add_documents, chunks, ids)
        uploaded_items.append(UploadItem(document_id=upload_id, file_name=safe_name, chunks_indexed=len(chunks)))

    return UploadResponse(session_id=session_id, uploaded=uploaded_items)


@router.post("/documents/impact", response_model=ImpactResponse, summary="Analyze document impact")
async def analyze_impact(request: ImpactRequest):
    if not request.problem.strip():
        raise HTTPException(status_code=400, detail="Problem description cannot be empty")
    if not request.session_id.strip() or len(request.session_id) > 128 or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in request.session_id
    ):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    valid_scopes = {"legal_corpus_only", "uploaded_docs_only", "hybrid"}
    if request.scope not in valid_scopes:
        raise HTTPException(status_code=400, detail=f"scope must be one of: {', '.join(sorted(valid_scopes))}")

    docs = []
    if request.scope in {"legal_corpus_only", "hybrid"}:
        base_docs = await asyncio.to_thread(_retrieve_documents, get_retriever(), request.problem)
        docs.extend(base_docs)

    if request.scope in {"uploaded_docs_only", "hybrid"}:
        # Apply both session and document ID filters in vector search, not after top-k retrieval.
        search_filter: dict = {"session_id": request.session_id}
        if request.document_ids:
            if len(request.document_ids) == 1:
                search_filter["upload_id"] = request.document_ids[0]
            else:
                search_filter["upload_id"] = {"$in": request.document_ids}
        upload_retriever = get_upload_retriever(search_filter=search_filter)
        upload_docs = await asyncio.to_thread(_retrieve_documents, upload_retriever, request.problem)
        docs.extend(upload_docs)

    if not docs:
        raise HTTPException(status_code=404, detail="No relevant documents found. Upload documents first or change scope.")

    analysis = await asyncio.to_thread(run_impact_analysis, request.problem, docs)
    return ImpactResponse(analysis=analysis, sources=_to_sources(docs))
