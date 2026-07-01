'''
/health: is process running 
/status: is process ready to serve

in k8s, if 
- /health is down pod restarts
- /status is down traffic re-routed to different pods
'''

import logging
from fastapi import APIRouter
from src.api.schema import HealthResponse, StatusResponse
from src.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

@router.get(
            "/health",
            response_model=HealthResponse,
            summary="Liveness probe",
            description="Returns 200 if the server process is running"
            )
async def health() -> HealthResponse:
    return HealthResponse(status="ok")

@router.get(
        "/status",
        response_model=StatusResponse,
        summary="Readiness probe",
        description="checks vector store connectivity and returns system config"
        )
async def status()->StatusResponse:
    '''
    try to connect to chromaDB for doc count
    
    if chromaDB is unreachable, doc_count is None and status is 'degraded'
    service can answer using LLM alone but rag quality will be zero
    '''
    document_count:int|None = None
    overall_status = "ok"

    try:
        from src.rag.vectorstore import get_vectorstore
        store = get_vectorstore()
        document_count = store._collection.count()
    except Exception:
        logger.warning("could not reach vector_store for status", exc_info=True)
        overall_status="degraded"
    
    return StatusResponse(
        status=overall_status,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        embedding_provider=settings.embedding_provider,
        vector_store_path=settings.chroma_presist_dir,
        document_count=document_count,
        langsmith_enabled=settings.langsmith_enabled
    )