from typing import Optional, Dict, Any
import structlog
from app.tasks.billing import record_usage_task

logger = structlog.get_logger()

def record_usage(
    org_id: str,
    key_id: Optional[str],
    event_type: str,
    quantity: int = 1,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Triggers an asynchronous task to record a billable event.
    Non-blocking and safe to call from any endpoint.
    """
    try:
        # Convert UUIDs to strings for Celery serialization if necessary
        # but SQLAlchemy usually gives us strings or UUID objects that Celery can handle
        # if configured properly. We'll stringify just in case.
        record_usage_task.delay(
            str(org_id), 
            str(key_id) if key_id else None, 
            event_type, 
            quantity, 
            metadata or {}
        )
    except Exception as e:
        # We never want billing recording to crash the main request
        logger.error("METERING_ERROR", error=str(e), event_type=event_type)
