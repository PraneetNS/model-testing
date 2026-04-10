import pytest
import json
from unittest.mock import patch, MagicMock
from app.core.celery_app import ReliableTask
from celery.exceptions import SoftTimeLimitExceeded

def test_task_soft_time_limit_re_raises():
    """
    Test that the ReliableTask catches SoftTimeLimitExceeded,
    logs it, and re-raises it.
    """
    class DummyTask(ReliableTask):
        name = "test_time_limit_task"
        def run(self):
            pass

    task = DummyTask()

    with patch("app.core.celery_app.logging.error") as mock_log:
        with patch("celery.Task.__call__", side_effect=SoftTimeLimitExceeded("Task took too long")):
            with pytest.raises(SoftTimeLimitExceeded):
                task()
        mock_log.assert_called_once()
        assert "exceeded soft time limit" in mock_log.call_args[0][0]


def test_celery_dlq_routing_on_failure():
    """
    Test that the on_failure handler pushes the failed task 
    details into the mlguard.dlq Redis list with traceback.
    """
    class DummyTask(ReliableTask):
        name = "test_dlq_task"
        def run(self):
            pass

    task = DummyTask()
    exc = Exception("Some unrecoverable error")
    mock_einfo = MagicMock()
    mock_einfo.traceback = "Traceback details here"

    with patch("redis.Redis.from_url") as mock_redis_func:
        mock_redis_client = MagicMock()
        mock_redis_func.return_value = mock_redis_client

        task.on_failure(exc, "task-123", ("arg1",), {"kw": 1}, mock_einfo)

        # Assert lpush was called on mlguard.dlq
        mock_redis_client.lpush.assert_called_once()
        call_args = mock_redis_client.lpush.call_args[0]
        
        queue_name = call_args[0]
        payload = json.loads(call_args[1])

        assert queue_name == "mlguard.dlq"
        assert payload["task_id"] == "task-123"
        assert payload["task_name"] == "test_dlq_task"
        assert payload["error"] == "Some unrecoverable error"
        assert payload["traceback"] == "Traceback details here"
        assert payload["args"] == ["arg1"]
        assert payload["kwargs"] == {"kw": 1}
