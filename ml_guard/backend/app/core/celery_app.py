import os
import sys
from celery import Celery
from celery.schedules import crontab

# ML Guard core path injection
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from app.core.config import settings

celery_app = Celery("ml_guard", broker=settings.REDIS_URL)
celery_app.conf.result_backend = settings.REDIS_URL

# Automatically discover tasks in all task modules
celery_app.autodiscover_tasks([
    "app.workers",
    "app.domain.services",
    "app.services.forecasting",
    "app.tasks",  # covers ingest, red_team, reports, observability
])

# ─── Celery Beat Schedule ─────────────────────────────────────────────────────
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


@celery_app.task(name="test_task")
def test_task():
    return "Celery is working!"
