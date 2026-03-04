"""
rag/rag_pipeline.py
-------------------
Builds the LangChain RAG chain:
  question → retriever → format_docs → prompt → Ollama Mistral → str

Key decisions:
  - temperature=0.1  →  low creativity, high factual precision (law needs this)
  - Prompt mandates citations and graceful "I don't know" responses
  - format_docs injects [source | page] tags so the LLM can cite them
  - Chain and retriever are singletons (built once, reused across requests)
"""

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.core.config import settings
from rag.vector import get_retriever

# ── Legal-specific prompt ────────────────────────────────────────────────────
LEGAL_PROMPT_TEMPLATE = """You are an expert legal assistant specializing in Indian law.
Your answers are grounded ONLY in the provided context from:
  - Constitution of India
  - Indian Penal Code (IPC)

STRICT RULES:
1. Always cite the exact Article or Section number you are referencing.
   Example: "As per Article 21 of the Constitution..." or "Under Section 302 IPC..."
2. If the context does not contain enough information to answer the question,
   respond with: "The provided legal documents do not cover this specific question.
   Please consult a qualified lawyer."
3. Do NOT add any information that is not present in the context below.
4. Use formal, precise legal language.
5. Structure your answer clearly: state the law, cite the source, explain it.

CONTEXT (excerpts from legal documents):
{context}

QUESTION:
{question}

ANSWER (with Article/Section citations):"""

LEGAL_PROMPT = PromptTemplate.from_template(LEGAL_PROMPT_TEMPLATE)


# ── Helper: format retrieved docs with source labels ─────────────────────────
def format_docs(docs: list) -> str:
    """
    Prepend [source_file | Page X] to each chunk so the LLM
    has the citation information inline in the context.
    """
    parts = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(f"[{source} | Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Build LCEL chain ─────────────────────────────────────────────────────────
def build_rag_chain():
    retriever = get_retriever()

    llm = Ollama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_llm_model,   # mistral
        temperature=0.1,                    # low temp = factual, not creative
    )

    chain = (
        {
            "context":  retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | LEGAL_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ── Singletons ───────────────────────────────────────────────────────────────
_chain     = None
_retriever = None


def get_chain():
    """
    Return (and lazily build) the RAG chain + retriever singletons.
    Called once at startup via FastAPI lifespan.
    """
    global _chain, _retriever
    if _chain is None:
        _chain, _retriever = build_rag_chain()
    return _chain, _retriever
