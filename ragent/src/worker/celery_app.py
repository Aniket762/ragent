'''
celery for task queue for ingesting documents script

when /ingest is hit: (producer)
- celery_app.send_task("ingest_documents")
- writes tasks msg to redis list (the queue here)
- returns {task_id, status:"queued"} (202)

celery worker process: (consumer)
- polls redis list for new msg
- picks up "ingest_document" task
- runs load_directory + embed + store in chromaDB
- writes result back to redis with celery jobId

redis is used as broker(lpush/brpop) + db for status store for the celery tasks

key consideration for the config:
1. task_acks_late = True : msg stays in Redis until the task finishes, so worker crash causes redelivery not silent data loss
2. worker_prefetch_multiplier=1: each worker holds exactly one task at a time (critical for long CPU-bound embedding task)
3. JSON serialization: never pickle (pickle can execute arbitary code on deserialization)
'''

from celery import Celery

from src.config.settings import settings

celery_app = Celery("ragent")

celery_app.conf.update(
    broker_url=settings.redis_url,
    result_backend=settings.redis_url,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=86_400, #24 hours
    task_acks_late= True, #safe-retry issue: duplicate ingestion possible
    worker_prefetch_multiplier=1,
    task_default_queue="ingestion",
)

celery_app.autodiscover_tasks(["src.worker.tasks"])