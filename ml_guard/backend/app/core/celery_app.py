from celery import Celery
from app.core.config import settings

celery_app = Celery("ml_guard", broker=settings.CELERY_BROKER_URL)
celery_app.conf.result_backend = settings.CELERY_RESULT_BACKEND

# Automatically discover tasks in the app
celery_app.autodiscover_tasks(['app.domain.services'])

@celery_app.task(name="test_task")
def test_task():
    return "Celery is working!"
