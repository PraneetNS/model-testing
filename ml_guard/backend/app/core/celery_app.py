import os
import sys
from celery import Celery

# ML Guard core path injection
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from app.core.config import settings

celery_app = Celery("ml_guard", broker=settings.CELERY_BROKER_URL)
celery_app.conf.result_backend = settings.CELERY_RESULT_BACKEND

# Automatically discover tasks in the app
celery_app.autodiscover_tasks(['app.workers', 'app.domain.services'])

@celery_app.task(name="test_task")
def test_task():
    return "Celery is working!"
