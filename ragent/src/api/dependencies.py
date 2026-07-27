'''
dependency injected into route handlers using Depends() system.

run before each request to reject req early
like: auth, rate limiting, db sessions

Semaphore used to limit coroutines inside a block. 

Why Semaphores per-process instead of global?
each uvicorn worker is a seperate process with it's own event loop.
Semaphore lives in one event loop and cannot be shared accross processes.

Total concurrency = max_concurrent_llm_calls* num_workers.
Set  max_concurrent_llm_calls = (llm_rpm_limit/num_workers)

For cross process semaphonre use Redis + distributed lock library

- FastAPI runs in a single asyncio event loop per worker
'''
import asyncio
from fastapi import Header, HTTPException, status, Depends, Request
from src.config.settings import settings


_llm_semaphore: asyncio.Semaphore | None = None

def get_llm_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
    return _llm_semaphore

def get_request_id(request:Request) -> str:
    return getattr(request.state,"request_id","unkown")

async def verify_api_key(auth:str|None = Header(None, description="Bearer <api-key>"))->None:
    '''
    dummy auth for now will extend later
    @router.post("/ask", dependencies=[Depends(verify_api_key)])
    '''
    api_secret = getattr(settings, "api_secret_key","")
    if not api_secret:
        return
    
    if auth is None or not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="malformed auth header"
        )
    
    token = auth.removeprefix("Bearer ").strip()
    if token!=api_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="you do not have access to this"
        )



