from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

from src.config.settings import settings
from src.rag.vectorstore import get_base_retriever

class BM25Retriever(BaseRetriever):
    '''
    keyword retriever over an in-memory document corpus
    '''
    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Document]
    k:int = settings.retriver_top_k

    def _get_relevant_documents(self, query:str, *, run_manager:CallbackManagerForRetrieverRun)-> list[Document]:
        corpus = [doc.page_content.lower().split() for doc in self.documents]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i],reverse=True)[:self.k] # top k 
        return [self.documents[i] for i in top_indices]
    
def build_retriever(corpus: list[Document] | None=None)->BaseRetriever:
    '''
    return a hybrid retriever when use_bm25 is true + corpus provided
    fallsback to pure semantic search

    hybrid wt: 60% semantic + 40% bm 25
    '''
    semantic = get_base_retriever()

    if not settings.use_bm25 or not corpus:
        return semantic
        
    bm25 = BM25Retriever(documents=corpus, k=settings.retriver_top_k)
    return EnsembleRetriever(
        retrievers= [semantic,bm25],
        weights=[0.6,0.4]
    )