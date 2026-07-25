'''
starts fastapi server with uvicorn.

run with python main.py
or
to control multiple workers
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
'''

import logging
import uvicorn

from src.api.app import create_app

#TODO: replace basic config with Json formatter for ingestion by splunk, datadog
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True, #TODO: before promoting code make it false, reload adds overhead and multiple workers issues
        log_level="info"
    )