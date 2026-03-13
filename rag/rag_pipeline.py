"""
rag/rag_pipeline.py
-------------------
Builds LangChain RAG chains using Groq:
  1) get_chain()        -> Legal Q&A
  2) get_advice_chain() -> Legal advice
Also exposes run_impact_analysis(problem, docs) for uploaded-document impact review.
"""

try:
    from langchain_groq import ChatGroq
except ImportError as exc:
    raise ImportError(
        "langchain_groq is not installed. Install with: pip install langchain-groq"
    ) from exc

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.core.config import settings
from rag.vector import get_retriever


# Shared LLM instance
def _build_llm() -> ChatGroq:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Create .env and set GROQ_API_KEY=<your_key>."
        )

    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
        max_tokens=1024,
    )


def format_docs(docs: list) -> str:
    """Attach lightweight citations for each chunk."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{source} | Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _pick_text_field(field: str):
    def _extract(payload):
        if isinstance(payload, dict):
            return payload.get(field, "")
        return payload

    return RunnableLambda(_extract)


LEGAL_QA_PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert legal assistant specializing in Indian law.
Your answers are grounded ONLY in the context provided below from:
  - Constitution of India
  - Indian Penal Code (IPC)

STRICT RULES:
1. Always cite the exact Article or Section number you reference.
2. If context is insufficient, say:
   "The provided legal documents do not cover this question. Please consult a qualified lawyer."
3. Do NOT add information not present in context.
4. Use formal, precise legal language.
5. Structure: state the law -> cite source -> explain simply.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)


LEGAL_ADVICE_PROMPT = ChatPromptTemplate.from_template(
    """
You are a senior Indian legal advisor helping someone understand their legal situation.
Use ONLY the context provided from the Constitution of India and IPC.

Respond in this exact structure:
CASE TYPE:
[Criminal / Civil / Constitutional / Mixed]

RELEVANT LAWS:
[List specific Articles or IPC Sections with one-line explanation]

YOUR LEGAL SITUATION:
[Explain what the law says about this situation in plain language]

STEPS TO TAKE RIGHT NOW:
[Numbered list]

IMPORTANT WARNINGS:
[Deadlines, risks, things not to do]

TYPE OF LAWYER TO CONSULT:
[Specific type]

CONTEXT:
{context}

PROBLEM DESCRIBED:
{problem}

LEGAL ADVICE:
"""
)


IMPACT_PROMPT = ChatPromptTemplate.from_template(
    """
You are an Indian legal analysis assistant.
Use ONLY the provided context.

Task:
1. Explain what the uploaded document excerpts state.
2. Map those points to the user's problem description.
3. Explain likely legal effect and practical implications.
4. Cite relevant Articles/Sections only if present in context.
5. If information is insufficient, explicitly say so.

Output format:
DOCUMENT SUMMARY:
[Plain-English summary]

KEY LEGAL POINTS:
[Bulleted points]

IMPACT ON THE PROBLEM:
[How these documents affect the described issue]

SUGGESTED NEXT STEPS:
[Numbered practical steps]

LIMITATIONS:
[What is unknown / needs lawyer review]

CONTEXT:
{context}

PROBLEM:
{problem}

ANALYSIS:
"""
)


def build_rag_chain():
    retriever = get_retriever()
    llm = _build_llm()
    question_input = _pick_text_field("question")

    chain = (
        {
            "context": question_input | retriever | RunnableLambda(format_docs),
            "question": question_input,
        }
        | LEGAL_QA_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def build_advice_chain():
    retriever = get_retriever()
    llm = _build_llm()
    problem_input = _pick_text_field("problem")

    chain = (
        {
            "context": problem_input | retriever | RunnableLambda(format_docs),
            "problem": problem_input,
        }
        | LEGAL_ADVICE_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def run_impact_analysis(problem: str, docs: list) -> str:
    """Generate a structured impact report for uploaded docs + problem."""
    llm = _build_llm()
    chain = IMPACT_PROMPT | llm | StrOutputParser()
    context = format_docs(docs) if docs else "No document context provided."
    return chain.invoke({"problem": problem, "context": context})


_qa_chain = None
_advice_chain = None
_retriever = None


def get_chain():
    global _qa_chain, _retriever
    if _qa_chain is None:
        _qa_chain, _retriever = build_rag_chain()
    return _qa_chain, _retriever


def get_advice_chain():
    global _advice_chain, _retriever
    if _advice_chain is None:
        _advice_chain, _retriever = build_advice_chain()
    return _advice_chain, _retriever
