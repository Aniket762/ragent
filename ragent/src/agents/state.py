from typing_extensions import NotRequired,TypedDict
from langchain_core.documents import Document

class AgentState(TypedDict):
    ''' Shared state flowing through every node in the master and child graphs '''
    #input
    query:str
    santized_query:str

    #routing
    intent:str 

    #rag
    retrived_docs: list[Document] #top-k chunks from vector store
    generated_response:str

    #quality
    confidence_score: float
    guardrail_flags: list[str] # names of triggered guardrail

    #output
    final_response:str 

    # err propagation - set by any node that fails
    error:str | None

    # multi turn session
    session_id: NotRequired[str]
    frustration_score: NotRequired[float]
    frustration_signal: NotRequired[list[str]] # name of signals in current turn, reset to [] start of each new turn
    escalation_requested: NotRequired[bool] # agent flow stops if true
