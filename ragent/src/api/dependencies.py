'''
dependency injected into route handlers using Depends() system.

run before each request to reject req early
like: auth, rate limiting, db sessions
'''
from fastapi import Header, HTTPException, status
from src.config.settings import settings

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



