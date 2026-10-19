from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import rag.rag_pipeline as pipeline


def _sample_docs():
    return [
        Document(
            page_content="Article 21 protects life and personal liberty.",
            metadata={"source_file": "constitution_of_india.pdf", "page": 0},
        )
    ]


def test_qa_chain_passes_query_text_to_retriever(monkeypatch):
    seen = {}

    def retrieve(query):
        seen["query"] = query
        return _sample_docs()

    def fake_llm(prompt):
        seen["prompt"] = prompt.to_string()
        return "grounded answer"

    monkeypatch.setattr(pipeline, "get_retriever", lambda: RunnableLambda(retrieve))
    monkeypatch.setattr(pipeline, "_build_llm", lambda: RunnableLambda(fake_llm))

    chain, _ = pipeline.build_rag_chain()
    answer = chain.invoke({"question": "What does Article 21 protect?"})

    assert answer == "grounded answer"
    assert seen["query"] == "What does Article 21 protect?"
    assert "constitution_of_india.pdf | Page 1" in seen["prompt"]
    assert "Article 21 protects life and personal liberty." in seen["prompt"]


def test_advice_chain_passes_problem_text_to_retriever(monkeypatch):
    seen = {}

    def retrieve(query):
        seen["query"] = query
        return _sample_docs()

    monkeypatch.setattr(pipeline, "get_retriever", lambda: RunnableLambda(retrieve))
    monkeypatch.setattr(
        pipeline,
        "_build_llm",
        lambda: RunnableLambda(lambda prompt: "structured legal information"),
    )

    chain, _ = pipeline.build_advice_chain()
    answer = chain.invoke({"problem": "My right to personal liberty was restricted."})

    assert answer == "structured legal information"
    assert seen["query"] == "My right to personal liberty was restricted."


def test_impact_analysis_uses_provided_documents(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "_build_llm",
        lambda: RunnableLambda(lambda prompt: prompt.to_string()),
    )

    result = pipeline.run_impact_analysis(
        "Explain Article 21 in this context.",
        _sample_docs(),
    )

    assert "Explain Article 21 in this context." in result
    assert "constitution_of_india.pdf | Page 1" in result
    assert "Article 21 protects life and personal liberty." in result
