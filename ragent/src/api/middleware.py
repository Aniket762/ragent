'''
http mw stack

1. RequestIDMw: assign a unique uuid to every request. Easy debugging
2. RateLimitMw
3. setup_structured_logging + jsonLogFormatter - cloudwatch, elk, datadog
'''

import json
import logging
import time
import uuid
from collections import defaultdict, deque

from fastapi import status
from fastapi.responses import JSONResponse

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

class RequestIDMiddleware(BaseHTTPMiddleware):
    # id echoed in X-Request-ID response header for client
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint)-> Response:
        request_id = request.headers.get("X-Request_ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    # sliding window rate limiter
    # TODO: replace in-memory deque with Redis INCR, else req hitting worker1,worker2 gets 2x limit
    def __init__(self, app:ASGIApp, limit:int, window_seconds:int=60):
        super().__init__(app)
        self.limit = limit
        self.window = window_seconds
        self._timestamp: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint)->Response:
        # skip rate limit on health endpoint
        if request.url.path in ("/health","/api/v1/health"):
            return await call_next(request)

        client_ip = (
            request.headers.get("X-Forwarded-For","").split(",")[0].strip() 
            or (request.client.host if request.client else "unkown")
        )

        now = time.monotonic()
        window_start = now- self.window
        ts_dequeue = self._timestamp[client_ip]

        while ts_dequeue and ts_dequeue[0]<window_start:
            ts_dequeue.popleft()

        if len(ts_dequeue)>=self.limit:
            retry_after = int(self.window-(now-ts_dequeue[0]))+1
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail":"rate limit exceeded",
                    "request_id": getattr(request.state,"request_id",None)
                },
                headers = {"Retry-After": str(retry_after)}
            )

        ts_dequeue.append(now)
        return await call_next(request)

class JSONLogFormatter(logging.Formatter):
    def format(self,record:logging.LogRecord)-> str:
        payload: dict = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage()
        }

        for key in ("request_id","duration_ms","session_id","query_hash","task_id"):
            if hasattr(record,key):
                payload[key]=getattr(record,key)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload,default=str)

def setup_structured_logging(level:int= logging.INFO) ->None:
    formatter = JSONLogFormatter()
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(formatter)
    else:
        # containers collect stdout and forward to log aggregators
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)

