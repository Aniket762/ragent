'''
ChromaDB client with Singleton pattern

without singleton:
1. pattern 20 concurrent requests create 20 chroma objects.
2. each opens it's own file handle to chromaDB SQLite database
3. embedding model loaded into memory once per object - 20x ram usage
4. sqlite has single-writer semantics, concurrent writes serialize and can raise "db locked" error
'''
import threading

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config.settings import settings
from src.rag.embeddings import get_embedding_model

_chroma_lock: threading.Lock = threading.Lock()
_chroma_instance: Chroma | None = None

def get_vectorstore()->Chroma:
    global _chroma_instance

    if _chroma_instance is None:
        with _chroma_lock:
            # recheck inside the lock, another thread might have created while we were waiting for the lock
            if _chroma_instance is None:
                _chroma_instance = Chroma(
                    collection_name= settings.chroma_collection_name,
                    embedding_function=get_embedding_model(),
                    persist_directory=settings.chroma_presist_dir
                )

    return _chroma_instance

def reset_vectorstore()-> None:
    # clear singleton so next call creates a fresh client
    # DO NOT: call in prod, causes next request to re-initialize the connection and cause a brief window of inconsistency
    global _chroma_instance
    with _chroma_lock:
        _chroma_lock=None

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
