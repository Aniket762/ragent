import uuid as _uuid
from functools import lru_cache
from typing import Literal

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agents.policy_agent import policy_agent
from src.config.settings import settings
from src.agents.state import AgentState
from src.guardrails import detect_frustration_signals, check_output,check_input, update_frustration_score, should_escalate

class RouterDecision(BaseModel):
    intent:Literal["policy","out_of_scope"] # either policy agent or null for now
    reasoning:str

@lru_cache(maxsize=1)
def _get_router_llm():
    # for routing swap a cheaper model to save latency + cost
    return ChatAnthropic(
        model=settings.llm_model,
        max_tokens=256,
        temperature=0.0,
        anthropic_api_key= settings.anthropic_api_key
    ).with_structured_output(RouterDecision) 

_ROUTER_SYSTEM_PROMPT=""

def router_node(state:AgentState)-> dict:
    '''
    classify sanitized query + set intent in state
    '''
    query = state.get("santized_query") or state["query"]
    try:
        decision: RouterDecision = _get_router_llm().invoke([
            SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Query:{query}")
        ])
        return {"intent":decision.intent}
    except Exception as ex:
        # keep pipeline alive as it is test master agent
        return{
            "intent":"policy",
            "error": f"router failed (default to policy):{ex}"
        }

def out_of_scope_handler_node(state:AgentState)->dict:
    message ="tbd"
    return {
        "generated_response":message,
        "final_response":message,
        "confidence_score":1.0
    }

def input_guard_node(state:AgentState)->dict:
    return check_input(state)

def output_guard_node(state:AgentState)->dict:
    return check_output(state)

def frustration_check_node(state: AgentState) -> dict:
    query = state.get("santized_query") or state["query"]
    current_score = state.get("frustration_score",0.0)
    detected = detect_frustration_signals(query)
    new_score = update_frustration_score(current_score,detected)
    return{
        "frustration_score": new_score,
        "frustration_signals": detected
    }

def human_escalation_node(state:AgentState) -> dict:
    # for now a emapathic message with a support ticket reference -> integrate with snow/jira + notification to support channel + transcript
    ticket_id= f"HR-{_uuid.uuid4().hex[:8].upper()}"
    message = f"Don't worry we have routed to a human refer {ticket_id}"
    return{
        "final_response":message,
        "generated_response":message,
        "escalation_requested": True,
        "guardrail_flags" : list(state.get("guardrail_flags",[])) + ["escalated_to_human"]
    }

# edge functions: recieve full current state and returns a string key that maps to a node name
def _route_after_input_guard(state:AgentState) -> str:
    if state.get("error"):
        return "end"

    return "frustration_check"

def _route_after_frustration_check(state:AgentState) -> str:
    if state.get("escalation_requested"):
        return "human_escalation"
    if should_escalate(state.get("frustration_score",0.0)):
        return "human_escalation"
    return "router"

def _route_after_router(state:AgentState)-> str:
    return state.get("intent","out_of_scope")

# graph assembly
def _build_master_agent():
    # TODO: Memory-Saver file-based or redis based checkpointer, if server restarts checkpoints are lost
    graph:StateGraph = StateGraph(AgentState)

    #nodes
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("frustration_check", frustration_check_node)
    graph.add_node("router",router_node)
    graph.add_node("policy_agent",policy_agent)
    graph.add_node("out_of_scope_handler",out_of_scope_handler_node)
    graph.add_node("output_guard",output_guard_node)
    graph.add_node("human_escalation", human_escalation_node)

    # entry point
    graph.set_entry_point("input_guard")

    # edges
    graph.add_conditional_edges(
        "input_guard", # starting node
        _route_after_input_guard, # routing fn
        {
            "end":END, 
            "frustration_check":"frustration_check"
        } # decision map
    )

    graph.add_conditional_edges(
        "frustration_check",
        _route_after_frustration_check,
        {
            "human_escalation":"human_escalation", 
            "router":"router"
        }
    )

    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "policy":"policy_agent",
            "out_of_scope": "out_of_scope_handler"
        }
    )

    graph.add_edge("policy_agent","output_guard")
    graph.add_edge("output_guard", END)
    graph.add_edge("out_of_scope_handler", END)
    graph.add_edge("human_escalation",END)

    return graph.compile(checkpointer=MemorySaver())


master_agent = _build_master_agent()

def _build_turn_input(query:str, session_id: str) -> dict:
    # frustration score, escalation requested presisted by checkpointer
    return{
        "query": query, 
        "sanitized_query": "", #set by input_guard_node 
        "intent":"", # set by router_node 
        "retrieved_docs":[],# set by retrieve_node 
        "generated_response":"", # set by generate_node
        "confidence_score":0.0, # set by judge | None
        "guardrail_flags":[], # accumulated fresh each turn
        "final_response":"", # set byoutput_guard
        "error": None, # cleared at turn start
        "session_id": session_id,
        "frustration_signals":[] # signals only for this turn
    }

async def ask_with_session(query: str,session_id: str|None=None) -> tuple[str,str]:
    '''
    multi turn query with session level frustration tracking
    
    pass the returned session_id back on subsequent calls to continue the same conversation thread. Memory Saver checkpointer persists 
    frustration_score and escalation_requested accross turns
    '''
    if session_id is None:
        session_id = str(_uuid.uuid4())

    turn_input = _build_turn_input(query=query,session_id=session_id)
    config = {"configurable":{"thread_id": session_id}}

    result: AgentState = await master_agent.ainvoke(turn_input, config=config)
    response = result.get("final_response") or result.get("generated_response", "No response generated")
    return response, session_id

def ask(query:str)-> str:
    # single turn stateless query
    session_id = str(_uuid.uuid4())
    turn_input = _build_turn_input(query, session_id)
    config = {"configurable":{"thread_id": session_id}}

    result:AgentState = master_agent.invoke(turn_input,config=config)
    return result.get("final_response") or result.get("generated_response","No response generated")