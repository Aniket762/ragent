'''
is_policy_question() can be invoked anywhere to determine without running full agent graph

Usecase: 
- rejects off topic queries at api boundary before they every reach langgraph saving routing + retrieval cost
- batch pre-filtering
'''

from functools import lru_cache
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage,SystemMessage
from pydantic import BaseModel

from src.config.settings import settings

class _ScopeDecision(BaseModel):
    '''
    internal output model for scope classifier
    '''
    in_scope:bool
    reasoning: str

@lru_cache(maxsize=1)
def _get_scope_llm():
    '''
    cache llm client for scope detection
    '''
    return ChatAnthropic(
        model = settings.llm_model, # baadh mei haiku/gemini 2.5 pro se replace krna hai
        max_tokens=128,
        temperature=0.0,
        anthropic_api_key = settings.anthropic_api_key
    ).with_structured_output(_ScopeDecision)

_SCOPE_SYSTEM_PROMPT="tdb"

def is_policy_question(query:str) -> bool:
    decision: _ScopeDecision = _get_scope_llm().invoke([
        SystemMessage(content=_SCOPE_SYSTEM_PROMPT),
        HumanMessage(content=f"Query:{query}")
    ])
    return decision.in_scope

def classify(query:str) -> Literal["policy","out_of_scope"]:
    return "policy" if is_policy_question(query) else "out_of_scope"