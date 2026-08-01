"""Celery application instance.

This is the ONE Celery app used by both the worker process (started via
`celery -A app.workers.celery_app worker`) and the FastAPI app (which
enqueues tasks via `.delay()`/`.apply_async()`). Both processes import
this same module so task names/routing stay consistent.
"""

"""Celery application instance."""

from celery import Celery

from app import models  # noqa: F401 — registers ALL models before any task runs
from app.config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "knowledge_base",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.document_processing"],
)


celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Acknowledge tasks only AFTER they complete (not immediately on
    # receipt) — if a worker crashes mid-task, the task is redelivered to
    # another worker instead of being silently lost.
    task_acks_late=True,
    # Prevents one worker from grabbing many tasks upfront and starving
    # other workers — important once we run multiple worker replicas.
    worker_prefetch_multiplier=1,
)