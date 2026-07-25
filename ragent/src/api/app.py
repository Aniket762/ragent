'''
factory pattern

centralize all app config
1. lifespan: startup/shutdown with langsmith setup
2. middleware: cors, request logging
3. router registration: all route modules mounted under /api/v1
4. execpetion handlers
'''

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import health,ingest, query
from src.config.settings import settings
from src.observability.tracing import setup_langsmith

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app:FastAPI):
    enabled = setup_langsmith()
    if enabled:
        logger.info("langsmith tracing enabled for project: %s", settings.langsmith_project)
    else:
        logger.info("langsmith tracing disabled")

    logger.info(
        "ragent started - LLM: %s | Embeddings: %s/%s",
        settings.llm_model,
        settings.embedding_provider,
        settings.embedding_model
    )

    yield # app runs

    logger.info("ragent shutting down .....")

def create_app()-> FastAPI:
    app = FastAPI(
        title="Ragent",
        description=" rag based qa agent with guardrails and frustration escalation",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET","POST"],
        allow_headers=["*"]
    )

    # req timing mw, logs each req method, path, status code and duration
    # helps identify slow queries w/o APM tooling
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter()-start)*1000
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms
        )

        return response

    # exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception for %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail":"An internal error occurred"}
        )

    #route registration
    prefix = "/api/v1"
    app.include_router(query.router,prefix=prefix)
    app.include_router(ingest.router,prefix=prefix)
    app.include_router(health.router, prefix=prefix)

    return app