from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "vectorstore"

def get_vectorstore():
    embeddings = OllamaEmbeddings(model="mistral")

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return db
