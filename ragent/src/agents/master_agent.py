import uuid as _uuid
from functools import lru_cache
from typing import Literal

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings
from src.agents.state import AgentState
from src.guardrails.input_guard import check_input
from src.guardrails.output_guard import check_output

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
    message =""
    return {
        "generated_response":message,
        "final_response":message,
        "confidence_score":1.0
    }

def input_guard_node(state:AgentState)->dict:
    return check_output(state)

