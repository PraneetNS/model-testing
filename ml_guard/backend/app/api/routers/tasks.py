from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
import json
import redis
from app.core.celery_app import celery_app
from app.core.config import settings

router = APIRouter(prefix="/api/tasks", tags=["Tasks"])

@router.get("/dlq", response_model=List[Dict[str, Any]])
def get_dead_letter_queue(limit: int = 50):
    """
    List the last `limit` dead-lettered tasks with their error summaries.
    Reads from the Redis list `mlguard.dlq`.
    """
    try:
        r = redis.Redis.from_url(settings.REDIS_URL)
        # Fetch items from the list
        messages = r.lrange("mlguard.dlq", 0, limit - 1)
        results = []
        for m in messages:
            try:
                results.append(json.loads(m))
            except json.JSONDecodeError:
                pass
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to access DLQ: {e}")

@router.get("/{task_id}/status")
def get_task_status(task_id: str):
    """
    Returns detailed status for a given task ID.
    Includes state, result, error, and timestamps.
    """
    try:
        res = celery_app.AsyncResult(task_id)
        
        # Determine the error serialization if any
        error = None
        if res.state == "FAILURE":
            error = str(res.result)
            if hasattr(res, 'traceback'):
                error += f"\n{res.traceback}"

        result = res.result if res.state == "SUCCESS" else None

        return {
            "task_id": task_id,
            "state": res.state,
            "result": result,
            "error": error,
            "queued_at": None,          # Celery doesn't natively expose queued_at cleanly by default
            "started_at": getattr(res, 'date_done', None), # Approximate if success/failure
            "completed_at": res.date_done if res.state in ["SUCCESS", "FAILURE"] else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch task status: {e}")
