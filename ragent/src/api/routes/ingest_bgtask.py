'''
ARCHIVED: older version-using background task

currently, fire and forget model

/ingest returns "started" the actual loading, chunking, embedding, storing runs async
int background. prevents large document sets from timeouts

curently implemented celery + redis for task queue management
'''
# import asyncio
# import logging 
# from pathlib import Path 

# from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException,status

# from src.api.dependencies import verify_api_key
# from src.api.schema import IngestRequest,IngestResponse
# from src.config.settings import settings
# from src.rag.document_processor import load_directory
# from src.rag.vectorstore import add_documents, get_vectorstore

# logger = logging.getLogger(__name__)

# router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# async def _run_ingestion(directory:str, clear:bool)->None:
#     dir_path=Path(directory)

#     if not dir_path.exists():
#         logger.error("Ingestion failed directory not found: %s",directory)
#         return
    
#     logger.info("Starting ingestion from: %s (clear=%s)",directory,clear)

#     try:
#         if clear:
#             # clear db pre ingestion
#             def _clear():
#                 store = get_vectorstore()
#                 store._client.delete_collection(settings.chroma_collection_name)
#                 logger.info("vector store cleared")
#             await asyncio.to_thread(_clear)
        
#         # load + chunk docs (cpu +i/o bound op)
#         def _load():
#             chunks= load_directory(dir_path)
#             logger.info("loaded %d chunks from %s", len(chunks), dir)
#             return chunks
        
#         chunks = await asyncio.to_thread(_load)

#         if not chunks:
#             logger.warning("no doc found in %s - nothing ingested",dir)
#             return
        
#         #embedd + store
#         def _store():
#             n = add_documents(chunks)
#             logger.info("ingestion completed: %d chunks written to vecctor store",n)
        
#         await asyncio.to_thread(_store)
   
#     except Exception:
#         logger.exception("ingestion job failed for directory: %s",dir)

# @router.post(
#     "",
#     response_model=IngestResponse,
#     status_code=status.HTTP_202_ACCEPTED, # accepted but not complete
#     summary="ingest policy docs into vector store",
#     description="trigger background ingestion",
#     dependencies=[Depends(verify_api_key)]
# )

# async def ingest(req:IngestRequest, background_tasks: BackgroundTasks)->IngestResponse:
#     if not Path(req.directory).exists():
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="directory not found {req.directory}"
#         )
    
#     background_tasks.add_task(_run_ingestion,req.directory,req.clear)

#     return IngestResponse(
#         status="started",
#         message="ingestion started"
#     )