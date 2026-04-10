import os
import sys
from celery import Celery, Task
from celery.schedules import crontab
from celery.exceptions import SoftTimeLimitExceeded
import logging
import json

# ML Guard core path injection
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from app.core.config import settings

class ReliableTask(Task):
    """
    Custom task base class that pushes failed, out-of-retry tasks 
    into a Dead-Letter Queue (mlguard.dlq).
    """
    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        except SoftTimeLimitExceeded as e:
            logging.error(f"Task {self.name} exceeded soft time limit: {e}")
            raise

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        super().on_failure(exc, task_id, args, kwargs, einfo)
        trace = einfo.traceback if einfo else str(exc)
        try:
            import redis
            r = redis.Redis.from_url(settings.REDIS_URL)
            dlq_entry = {
                "task_id": task_id,
                "task_name": self.name,
                "args": args,
                "kwargs": kwargs,
                "error": str(exc),
                "traceback": trace
            }
            # We push directly to the redis queue 'mlguard.dlq'
            r.lpush("mlguard.dlq", json.dumps(dlq_entry))
        except Exception as filter_exc:
            pass

celery_app = Celery("ml_guard", broker=settings.REDIS_URL)
celery_app.Task = ReliableTask
celery_app.conf.result_backend = settings.REDIS_URL
celery_app.conf.task_soft_time_limit = 120
celery_app.conf.task_time_limit = 180

# Automatically discover tasks in all task modules
celery_app.autodiscover_tasks([
    "app.workers",
    "app.domain.services",
    "app.services.forecasting",
    "app.tasks",  # covers ingest, red_team, reports, observability
])

# â”€â”€â”€ Celery Beat Schedule â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
celery_app.conf.beat_schedule = {
    # Observability: Drift analysis every hour for all active models
    "drift-scan-hourly": {
        "task": "app.tasks.observability.run_hourly_drift_scan",
        "schedule": crontab(minute=0),  # :00 every hour
    },
    # Observability: Performance snapshot every 6 hours
    "performance-snapshot-6h": {
        "task": "app.tasks.observability.run_performance_snapshot",
        "schedule": crontab(minute=0, hour="*/6"),  # Every 6 hours
    },
}

celery_app.conf.timezone = "UTC"


@celery_app.task(name="test_task", bind=True, max_retries=3, default_retry_delay=10)
def test_task():
    return "Celery is working!"
