"""
scripts/ingest_documents.py
----------------------------
One-time script to:
  1. Load all PDFs from data/raw/
  2. Split them into chunks (legal-aware separators)
  3. Embed with nomic-embed-text via Ollama
  4. Persist to ChromaDB at vectorstore/

Run ONCE before starting the API:
  python scripts/ingest_documents.py

Re-run any time you add new PDFs to data/raw/.
"""

import sys
from pathlib import Path

# Allow importing from project root
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from app.core.config import settings

DATA_DIR = Path("data/raw")


# ── Step 1: Load PDFs ─────────────────────────────────────────────────────────
def load_documents() -> list:
    """
    Load all PDFs from DATA_DIR.
    Adds source_file and doc_type metadata to every page.
    """
    all_docs = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No PDFs found in {DATA_DIR}. Add your PDFs and re-run.")
        sys.exit(1)

    for pdf_path in pdf_files:
        print(f"📄 Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()

        for page in pages:
            # Enrich metadata — every chunk will inherit this
            page.metadata["source_file"] = pdf_path.name
            page.metadata["doc_type"]    = "legal_document"

        all_docs.extend(pages)
        print(f"   → {len(pages)} pages loaded")

    print(f"\n✅ Total pages loaded: {len(all_docs)}\n")
    return all_docs


# ── Step 2: Chunk ─────────────────────────────────────────────────────────────
def split_documents(docs: list) -> list:
    """
    Split pages into chunks using legal-aware separators.

    Separator priority:
      1. \\n\\nArticle / \\n\\nSection  — respect legal structure
      2. \\n\\n                         — paragraph boundary
      3. \\n                            — line boundary
      4. .                             — sentence boundary
      5. ' '                           — word boundary (last resort)

    chunk_overlap=150 ensures that a sentence starting in one chunk
    and completing in the next is still readable in context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\nArticle ",   # Constitution: "Article 21 — Right to life"
            "\n\nSection ",   # IPC: "Section 302 — Punishment for murder"
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        add_start_index=True,  # stores char offset — enables position citations
    )

    chunks = splitter.split_documents(docs)
    print(f"✅ Total chunks created: {len(chunks)}\n")
    return chunks


# ── Step 3 & 4: Embed + Store ─────────────────────────────────────────────────
def build_vectorstore(chunks: list) -> None:
    """
    Embed chunks with nomic-embed-text and persist to ChromaDB.

    IMPORTANT: uses OllamaEmbeddings (embedding model),
               NOT OllamaLLM / ChatOllama (generative model).
    """
    print(f"🔢 Embedding with model: {settings.ollama_embed_model}")
    print(f"   (Make sure you ran: ollama pull {settings.ollama_embed_model})\n")

    embeddings = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
    )

    print("💾 Storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )

    count = vectorstore._collection.count()
    print(f"\n✅ Vectorstore ready!")
    print(f"   Location  : {settings.chroma_persist_dir}/")
    print(f"   Collection: {settings.chroma_collection_name}")
    print(f"   Documents : {count} chunks stored\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  AI Legal Assistant — Document Ingestion")
    print("=" * 55 + "\n")

    docs   = load_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)

    print("🚀 Done! You can now start the API:")
    print("   uvicorn main:app --reload --port 8000\n")
