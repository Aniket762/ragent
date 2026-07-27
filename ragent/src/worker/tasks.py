import logging 
from pathlib import Path

from celery import Task
from celery.exceptions import MaxRetriesExceededError

from src.config.settings import settings
from src.rag.document_processor import load_directory
from src.rag.vectorstore import add_documents, get_vectorstore
from src.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

@celery_app.task(
    name="tasks.ingest_documents",
    bind=True,
    autoretry_for = (IOError,OSError,ConnectionError),
    max_retries=3,

    #exponential backoff
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,

    acks_late= True
)

def ingest_documents(self:Task, directory:str, clear: bool=False) -> dict:
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Ingestion directory not found:{directory}")

    logger.info("[task:%s] starting ingestion: %s (clear=%s)", self.request.id, directory,clear)

    # clear vector store
    self.update_state(
        state="PROGRESS",
        meta={"step":"clearing","pct":0, "chunks_written":0}
    )

    if clear:
        try:
            store = get_vectorstore()
            store._client.delete_collection(settings.chroma_collection_name)
            logger.info("[task: %s] vector store clearerd.", self.request.id)
        except Exception as exec:
            logger.warning("[task:%s] clear failed (non-fatal): %s", self.request.id, exec)

    # load and chunk docs
    self.update_state(
        state="PROGRESS",
        meta={"step":"loading","pct":20,"chunks_written":0}
    )

    try:
        chunks = load_directory(dir_path)
    except Exception as exec:
        logger.exception("[task:%s] load_directory failed", self.request.id)
        raise exec

    if not chunks:
        logger.exception("[task:%s] no documents found in %s", self.request.id, directory)

    logger.info("[task:%s] loaded %d chunks.", self.request.id,  len(chunks))

    # embed and store
    self.update_state(
        state="PROGRESS",
        meta={"step":"embedding","pct":50,"chunks_written":0}
    )

    try:
        n = add_documents(chunks)
    except Exception as exec:
        logger.exception("[task:%s] add_documents failed", self.request.id)
        raise exec

    logger.info("[task:%s] ingestion complete: %d chunks written.", self.request.id, n)

    self.update_state(
        state="PROGRESS",
        meta={"step":"done","pct":100,"chunks_written":n}
    )

    return {"directory":directory, "chunks_written":n, "cleared":clear}