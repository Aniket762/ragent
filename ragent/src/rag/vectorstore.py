from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config.settings import settings
from src.rag.embeddings import get_embedding_model

def get_vectorstore()->Chroma:
    return Chroma(
        collection_name= settings.chroma_collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=settings.chroma_presist_dir
    )

def add_documents(docs:list[Document])->int:
    """
    Upsert docs into vector store. Return number of chunks added
    """
    store = get_vectorstore()
    store.add_documents(docs)
    return len(docs)

def get_base_retriever() -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(
        search_kwargs = {"k": settings.retriver_top_k}
    )
