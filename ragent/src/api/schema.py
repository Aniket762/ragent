'''
fastApi - req and res models
'''
from pydantic import BaseModel, Field

class SourceDoc(BaseModel):
    '''
    a single retrieved policy chunk, returned when include_source=true
    '''
    policy_name:str
    section:str
    snipped:str

class AskRequest(BaseModel):
    '''
    req body for POST /ask and /ask/stream
    '''
    query:str = Field(..., min_length=1,max_length=2000,description="The policy question")
    include_sources:bool = Field(
        False,
        description="Include retrieved document snippets in the response"
    )
    session_id:str | None = Field(
        None,
        description=(
            """
            session id for multi-turn conversation, omit on first turn
            pass session id from previous response to continue the thread
            """
        ),
    )

class AskResponse(BaseModel):
    '''
    response body for POST /ask (sync)
    for /ask/stream each sse carries StreamChunk payload
    '''
    response:str = Field(..., description="The final, guardrail checked")
    guardrail_flags:list[str] = Field(
        default_factory=list,
        description="names of any guardrails that were triggered"
    )
    sources:list[SourceDoc]|None = Field(
        None,
        description="retrieved policy chunks"
    )
    session_id: str = Field(
        ...,
        description="session id for this conversation. pass back on subsequent turns"
    )

class StreamChunk(BaseModel):
    '''
    payload for SSE during POST /ask/stream

    stream sends one event per token chunk from llm, then a flag done:true
    when full response is completed with guard flags, sources, session id

    client:
    1. accumulate chunk text until done==true
    2. read flags from final event
    3. save session id from final event for next turn
    '''
    chunk:str = Field("",description="Tokens from llm - empty on final event")
    done:bool = Field(False, description="True on last event in stream")
    guardrail_flags:list[str] = Field(default_factory=list)
    sources:list[SourceDoc] | None = None
    session_id:str|None = Field(
        None,
        description="only present on final event."
    )