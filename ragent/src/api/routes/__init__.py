from src.api.routes.query import router as query_router
from src.api.routes.ingest import router as ingest_router
from src.api.routes.health import router as health_router

__all__ = ["query_router","ingest_router","health_router"]