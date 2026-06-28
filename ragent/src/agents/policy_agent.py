from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.agents.state import AgentState
from src.config.settings import settings
from src.rag.retriver import build_retriever

_SYSTEM_PROMPT="tbd"

@lru_cache(maxsize=1)
def _get_llm()->ChatAnthropic:
    return ChatAnthropic(
        model=settings.llm_model,
        max_tokens= settings.llm_max_tokens,
        temperature= settings.llm_temperature,
        anthropic_api_key= settings.anthropic_api_key
    )

def _format_context(docs:list[Document])-> str:
    '''
    render retrieved chunks into a numbered context block for the prompt
    '''
    parts:list[str] = []
    for i, doc in enumerate(docs,1):
        policy = doc.metadata.get("policy_name","Unkown policy")
        section = doc.metadata.get("section","")
        header = f"[{i}] {policy}" + (f"- {section}" if section else "")
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n--\n\n".join(parts)

def transform_query_node(state: AgentState)-> dict:
    '''
    normalize query before retrieval
    later will extend with HyDE query exapansion 
    '''
    return {"sanitized_query": state.get("santized_query") or state["query"]}

def retrieve_node(state:AgentState) -> dict:
    '''
    fetch top k policy chunks from vector store
    '''
    query = state.get("santized_query") or state["query"]
    try:
        retriever = build_retriever()
        docs = retriever.invoke(query)
        return {"retrieved_docs":docs}
    except Exception as ex:
        return {"retrieved_docs":[], "error":f"retrieval failed: {ex}"}

def generate_node(state:AgentState)->dict:
    '''
    generate grounded answer with llm using the retirved context
    '''
    if state.get("error"):
        return {"generated_response":"Error occurred during document retrieval."}
    
    docs = state.get("retrived_docs",[])
    query = state.get("santized_query") or state["query"]

    if not docs:
        return{
            "generated_response":(
                "I couldn't find relevant policy documents to answer your question"
            )
        }
    
    context = _format_context(docs)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Policy document excerpts:\n\n{context}\n\nQuestion:{query}")
    ]

    try:
        response = _get_llm().invoke(messages)
        return {"generated_response": str(response.content)}
    except Exception as ex:
        return{
            "generated_response": "Failed to generate a response",
            "error":f"LLM call failed:{ex}"
        }
    
# graph build
def _build_policy_agent()-> StateGraph:
    graph:StateGraph = StateGraph(AgentState)

    graph.add_node("transform_query", transform_query_node)
    graph.add_node("retrieve",retrieve_node)
    graph.add_node("generate",generate_node)

    graph.set_entry_point("transform_query")
    graph.add_edge("transform_query","retrieve")
    graph.add_edge("retrieve","generate")
    graph.add_edge("generate", END)

    return graph.compile()

policy_agent = _build_policy_agent()