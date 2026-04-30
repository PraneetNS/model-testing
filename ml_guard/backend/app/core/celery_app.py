import os
import sys
from celery import Celery, Task
from celery.schedules import crontab
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import task_prerun
import logging
import json
import hmac
import hashlib
import base64
from cryptography.fernet import Fernet
import structlog

# ML Guard core path injection
_pkg_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
for p in [_pkg_parent, _repo_root]:
    if p not in sys.path:
        sys.path.append(p)

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
celery_app.conf.task_soft_time_limit = 600  # 10 minutes
celery_app.conf.task_time_limit = 900       # 15 minutes
celery_app.conf.result_expires = 3600       # 1 hour expiration for task results

# TASK SIGNING & ISOLATION
_hmac_key = hmac.new(
    settings.SECRET_KEY.encode(),
    b"celery-signing",
    digestmod=hashlib.sha256
).hexdigest()

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    security_key=_hmac_key,
    task_always_eager=(settings.MLGUARD_ENV.lower() == "test"),
    task_routes={
        '*drift*': {'queue': 'drift'},
        '*audit*': {'queue': 'audit'},
        '*comprehensive_scan*': {'queue': 'audit'},
        '*red_team*': {'queue': 'red_team'},
    }
)

# Automatically discover tasks in all task modules
celery_app.autodiscover_tasks([
    "app.workers",
    "app.domain.services",
    "app.services.forecasting",
    "app.tasks",  # covers ingest, red_team, reports, observability
    "app.tasks.observability",
    "app.tasks.retraining",
    "app.tasks.scoring",
    "app.tasks.inventory",
    "app.tasks.sandbox",
    "app.tasks.billing",
    "app.tasks.reports",
    "app.tasks.red_team",
    "app.tasks.ingest",
    "app.tasks.notifications",
    "app.tasks.explainability",
])

# Ensure all task modules are imported so decorators run
try:
    import app.tasks.observability
    import app.tasks.retraining
    import app.tasks.scoring
    import app.tasks.inventory
    import app.tasks.sandbox
    import app.tasks.billing
    import app.tasks.reports
    import app.tasks.red_team
    import app.tasks.ingest
    import app.tasks.notifications
    import app.tasks.explainability
    import app.workers.tasks
except Exception as e:
    structlog.get_logger().error("task_import_failed", error=str(e))

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
    # Automated Retraining: Evaluate triggers every hour
    "evaluate-retraining-triggers": {
        "task": "app.tasks.retraining.evaluate_all_retraining_policies",
        "schedule": crontab(minute=0),  # Every hour
    },
    # Governance: Refresh all model scores every hour
    "refresh-scores-hourly": {
        "task": "app.tasks.scoring.refresh_all_scores",
        "schedule": crontab(minute=30),  # :30 every hour
    },
    # Inventory: Check for overdue validations daily
    "inventory-due-check": {
        "task": "app.tasks.inventory.check_validation_due_dates",
        "schedule": crontab(minute=0, hour=1),  # 01:00 AM UTC
    },
    # Sandbox: Cleanup expired sandboxes daily
    "sandbox-cleanup-daily": {
        "task": "app.tasks.sandbox.cleanup_expired",
        "schedule": crontab(minute=0, hour=2),  # 02:00 AM UTC
    },
    # Billing: Sync usage to Stripe daily
    "billing-stripe-sync": {
        "task": "app.tasks.billing.report_monthly_usage",
        "schedule": crontab(minute=0, hour=3),  # 03:00 AM UTC
    },
}

celery_app.conf.timezone = "UTC"

# â”€â”€â”€ ALLOWLIST & PRERUN VERIFICATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ALLOWED_TASKS = {
    "run_comprehensive_scan",
    "run_explainability_task",
    "run_governance_audit_task",
    "generate_aibom_task",
    "cleanup_expired_sandboxes",
    "run_red_team_task",
    "app.tasks.reports.generate_governance_report",
    "app.tasks.red_team.execute_red_team_campaign",
    "app.tasks.observability.run_hourly_drift_scan",
    "app.tasks.observability.run_performance_snapshot",
    "app.tasks.notifications.dispatch_alert",
    "app.tasks.ingest.ingest_batch_task",
    "app.tasks.explainability.run_explainability_task",
    "tasks.data_connector_fetch",
    "app.services.report_card.generate_governance_report",
    "app.services.red_team.execute_red_team_campaign",
    "app.services.forecasting.recompute_all_forecasts",
    "app.domain.services.governance_engine.run_async_training",
    "app.domain.services.governance_engine.run_async_evaluation",
    "app.domain.services.governance_engine.run_scheduled_monitoring",
    "app.domain.services.llm_evaluator.tasks.run_llm_evaluation_task",
    "app.tasks.retraining.evaluate_all_retraining_policies",
    "app.tasks.billing.record_usage_task",
    "app.tasks.scoring.refresh_all_scores",
    "app.tasks.inventory.check_validation_due_dates",
    "app.tasks.sandbox.cleanup_expired",
    "app.tasks.billing.report_monthly_usage",
    "test_task",
}

@task_prerun.connect
def verify_task_allowlist(task_id, task, *args, **kwargs):
    if task.name not in ALLOWED_TASKS:
        structlog.get_logger().error("SECURITY_ALERT", detail=f"Task {task.name} is not in ALLOWED_TASKS")
        raise Exception(f"Task {task.name} is not on the allowlist.")


@celery_app.task(name="test_task", bind=True, max_retries=3, default_retry_delay=10)
def test_task(self):
    return "Celery is working!"

# â”€â”€â”€ ENCRYPTION HELPERS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _get_fernet() -> Fernet:
    # 32-byte url-safe base64 key built off SECRET_KEY
    key = base64.urlsafe_b64encode(hashlib.sha256(settings.SECRET_KEY.encode()).digest())
    return Fernet(key)

def encrypt_task_payload(data: dict, sensitive_keys: list) -> dict:
    f = _get_fernet()
    out = dict(data)
    for k in sensitive_keys:
        if k in out and out[k] is not None:
            val = out[k]
            if isinstance(val, (dict, list)):
                val = json.dumps(val)
            out[k] = f.encrypt(str(val).encode()).decode('utf-8')
    return out

def decrypt_task_payload(data: dict, sensitive_keys: list) -> dict:
    f = _get_fernet()
    out = dict(data)
    for k in sensitive_keys:
        if k in out and out[k] is not None:
            decrypted = f.decrypt(str(out[k]).encode('utf-8')).decode('utf-8')
            try:
                out[k] = json.loads(decrypted)
            except:
                out[k] = decrypted
    return out
