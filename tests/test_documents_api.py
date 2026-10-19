from app.api.chat import _retrieve_documents as chat_retrieve
from app.api.documents import _safe_filename, _to_sources


class DummyRetrieverGetRelevant:
    def __init__(self):
        self.called = None

    def get_relevant_documents(self, query):
        self.called = "get_relevant_documents"
        return [query]


class DummyRetrieverInvoke:
    def __init__(self):
        self.called = None

    def invoke(self, query):
        self.called = "invoke"
        return [query]


class DummyDoc:
    def __init__(self, source_file, page):
        self.metadata = {"source_file": source_file, "page": page}


def test_retrieve_documents_get_relevant_documents():
    retriever = DummyRetrieverGetRelevant()
    docs = chat_retrieve(retriever, "hello")
    assert docs == ["hello"]
    assert retriever.called == "get_relevant_documents"


def test_retrieve_documents_invoke():
    retriever = DummyRetrieverInvoke()
    docs = chat_retrieve(retriever, "hello")
    assert docs == ["hello"]
    assert retriever.called == "invoke"


def test_to_sources_deduplicates_source_page_pairs():
    docs = [
        DummyDoc("a.pdf", 0),
        DummyDoc("a.pdf", 0),
        DummyDoc("a.pdf", 1),
    ]
    sources = _to_sources(docs)

    assert len(sources) == 2
    assert sources[0].source_file == "a.pdf"
    assert sources[0].page == 1
    assert sources[1].page == 2



def test_safe_filename_strips_client_supplied_paths():
    assert _safe_filename("../../private.txt") == "private.txt"
    assert _safe_filename(r"C:\\temp\\private.txt") == "private.txt"
