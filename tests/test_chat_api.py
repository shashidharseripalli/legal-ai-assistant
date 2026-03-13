from app.api.chat import _retrieve_documents


class DummyRetriever:
    def __init__(self):
        self.called = None

    def get_relevant_documents(self, query):
        self.called = "get_relevant_documents"
        return [query]


def test_retrieve_documents_get_relevant_documents():
    retriever = DummyRetriever()
    docs = _retrieve_documents(retriever, "hello")
    assert docs == ["hello"]
    assert retriever.called == "get_relevant_documents"
