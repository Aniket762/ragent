'''
Design Choice: Celery + Redis over Background Tasks
1. crash recovery - re-delivery of msg
2. status polling - bgt followed fire and forget
3. retry on failure - autoretry + backoff for celery + redis
4. progress reporting - prev. only via logs, now update_state + %
5. horizontal scale - tied to one process, now more workers can be added
6. server isolation - seperate worker process, prev. shared event loop
'''

import logging
from pathlib import Path

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import verify_api_key
from src.api.schema import IngestRequest,IngestResponse, IngestStatusResponse
from src.worker.tasks import ingest_documents

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="ingest policy docs into vector store",
    description="enqueue background ingestion job, return immediately with a task_id",
    dependencies=[Depends(verify_api_key)]
)
async def ingest(request: IngestRequest)->IngestResponse:
    if not Path(request.directory).exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"directory not found: {request.directory}"
        )

    result:AsyncResult = ingest_documents.delay(request.directory,request.clear)

    logger.info("ingestion task queued: task_id=%s dir=%s", result.id, request.clear)

    return IngestResponse(
        status="queued",
        task_id=result.id,
        message=f"ingestion job queued for '{request.directory}' " 
    )

@router.get(
    "/status/{task_id}",
    response_model=IngestStatusResponse,
    summary="check ingestion job status",
    description="poll status of background ingestion job",
    dependencies=[Depends(verify_api_key)]
)
async def ingest_status(task_id: str) -> IngestStatusResponse:
    result= AsyncResult(task_id)
    state = result.state

    if state == "SUCCESS":
        info = result.result or {}
        return IngestStatusResponse(
            task_id=task_id,
            state="SUCCESS",
            pct=100,
            step="done",
            chunks_written=info.get("chunks_written")
        )

    if state == "FAILURE":
        return IngestStatusResponse(
            task_id=task_id,
            state="FAILURE",
            error= str(result.result)
        )

    if state == "PROGRESS":
        meta = result.info or {}
        return IngestStatusResponse(
            task_id=task_id,
            state="PROGRESS",
            pct=meta.get("pct"),
            step=meta.get("step"),
            chunks_written=meta.get("chunks_written")
        )

    # pending, started, retry nothing to report
    return IngestStatusResponse(task_id=task_id,state=state)