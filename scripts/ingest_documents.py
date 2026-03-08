"""
scripts/ingest_documents.py
----------------------------
Run this ONCE to:
  1. Load PDFs from data/raw/
  2. Split into chunks (legal-aware settings)
  3. Embed with Ollama nomic-embed-text
  4. Store in ChromaDB

Run:
  python scripts/ingest_documents.py

Re-run only if you add new documents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from app.core.config import settings

DATA_DIR = Path("data/raw")


def ingest():
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No PDFs found in {DATA_DIR}/")
        print("   Add your PDFs:")
        print("   - data/raw/constitution_of_india.pdf")
        print("   - data/raw/ipc.pdf")
        return

    print(f"📄 Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")

    # ── Load all PDFs ─────────────────────────────────────────────────────────
    all_docs = []
    for pdf_path in pdf_files:
        print(f"   Loading: {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        # Tag each chunk with the source filename for citations
        for doc in docs:
            doc.metadata["source_file"] = pdf_path.name

        all_docs.extend(docs)
        print(f"   → {len(docs)} pages loaded")

    print(f"\n📚 Total pages loaded: {len(all_docs)}")

    # ── Chunk text ────────────────────────────────────────────────────────────
    # Legal text needs larger chunks — sections/articles span multiple paragraphs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # larger than default for legal context
        chunk_overlap=150,     # overlap prevents losing cross-paragraph context
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(all_docs)
    print(f"✂️  Split into {len(chunks)} chunks")

    # ── Embed and store ───────────────────────────────────────────────────────
    print(f"\n🔢 Embedding with Ollama ({settings.ollama_embed_model})...")
    print("   This may take several minutes for large PDFs...")

    embeddings = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
    )

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=settings.chroma_collection_name,
    )

    print(f"\n✅ Done! {len(chunks)} chunks stored in ChromaDB at '{settings.chroma_persist_dir}/'")
    print("   You can now start the server: uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    ingest()
