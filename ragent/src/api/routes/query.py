import logging
import asyncio
import uuid as _uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from langchain_core.documents import Document 
from fastapi.responses import StreamingResponse

from src.api.dependencies import verify_api_key, get_llm_semaphore, get_request_id
from src.agents.state import AgentState
from src.api.schema import SourceDoc,AskRequest, AskResponse, StreamChunk
from src.agents import _build_turn_input, AgentState, master_agent
from src.config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ask", tags=["Query"])

def _docs_to_sources(docs:list[Document]) -> list[SourceDoc]:
    # convert retrieved langchain docs to api friendly source doc objects
    return[
        SourceDoc(
            policy_name= doc.metadata.get("policy_name", "Unkown"),
            section= doc.metadata.get("section",""),
            snippet = doc.page_content[:300].strip() # first 300 chars as snippet enough context for ui            
        ) for doc in docs
    ]


@router.post(
    "",
    summary="Ask a policy question (synchronous)",
    description="run the full policy pipeline and return the complete answer, blocks until LLM finishes generating",
    dependencies=[Depends(verify_api_key)]
)
async def ask(request: AskRequest, http_request:Request, request_id:str = Depends(get_request_id), semaphore:asyncio.Semaphore=Depends(get_llm_semaphore)) -> AskResponse:
    # once we have the full response from LLM only then we return the response
    # pass session_id from prev. response for multi-turn conversation + frustration score
    session_id = request.session_id or str(_uuid.uuid4())
    turn_input = _build_turn_input(query=request.query, session_id=session_id)
    config = {"configurable":{"thread_id": session_id}} #config tells langgraph which thread to load or save

    logger.info("ask request received",extra={"request_id":request_id,"session_id":session_id})

    try:
        async with semaphore:
            result:AgentState = await asyncio.wait_for(
                master_agent.ainvoke(turn_input,config=config),
                timeout=settings.agent_timeout_seconds
            )

    except asyncio.TimeoutError:
        # pipeline exceeded agent_timeout seconds
        logger.error("agent pipeline timeout")
        raise HTTPException(
            status_code= status.HTTP_504_GATEWAY_TIMEOUT,
            detail="request timed out"
        )

    sources = None
    if request.include_sources and result.get("retrived_docs"):
        sources = _docs_to_sources(result["retrived_docs"])

    logger.info("ask request completed!",extra={
        "request_id":request_id,
        "session_id":session_id,
        "confidence_score": result.get("confidence_score")
    })

    return AskResponse(
        response= result.get("final_response") or result.get("generated_response",""),
        guardrail_flags= result.get("guardrail_flags",[]),
        sources= sources,
        session_id=session_id
    )

@router.post(
    "/stream",
    summary="ask a policy question streaming SSE",
    description="stream llm answer token by token via SSE, final token has done=true",
    dependencies=[Depends(verify_api_key)],
    response_class=StreamingResponse
)
async def ask_stream(request: AskRequest,  http_request:Request, request_id:str = Depends(get_request_id), semaphore:asyncio.Semaphore=Depends(get_llm_semaphore)) -> StreamingResponse:
    session_id = request.session_id or str(_uuid.uuid4())
    turn_input = _build_turn_input(request.query, session_id=session_id)
    config = {"configurable":{"thread_id":session_id}}

    async def event_generator():
        '''
        langgraph's astream_events emits events as the graph executes.

        final state is only available after the graph completes so we collect it from the last on_chain_end event at the graph level
        '''
        async with semaphore:

            final_state:AgentState | None = None

            async for event in master_agent.astream_events(turn_input, config=config, version="v2"):
                event_type: str = event["event"]
                metadata: dict = event.get("metadata",{})

                # on_chat_model_stream fires once per token during llm generation
                # filter by langgraph_node to avoid streaming router tokens

                if(event_type == "on_chat_model_stream" and metadata.get("langgraph_node")=="generate"):
                    chunk_content = event["data"]["chunk"].content
                    if chunk_content:
                        payload = StreamChunk(chunk=chunk_content).model_dump_json()
                        yield f"data: {payload}\n\n"

                elif event_type=="on_chain_end" and event.get("name") == "LangGraph":
                    final_state = event["data"].get("output",{})

            # final done event
            sources = None
            flags:list[str] = []

            if final_state:
                flags = final_state.get("guardrail_flags",[])
                if request.include_sources and final_state.get("retrived_docs"):
                    sources = _docs_to_sources(final_state["retrived_docs"])

            done_payload = StreamChunk(
                chunk="",
                done=True,
                guardrail_flags=flags,
                sources=sources,
                session_id=session_id
            ).model_dump_json()
            yield f"data:{done_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        # headers prevent buffering, which would delay token delivery
        headers={
            "Cache-Control":"no-cache",
            "X-Accel-Buffering":"no", # disable nginx proxy buffering
            "Connection": "keep-alive"
        }
    )
        