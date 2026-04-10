import pytest
import uuid
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import get_db

# Create unique tokens
token1 = str(uuid.uuid4())
token2 = str(uuid.uuid4())

# Setup mock jobs returning completed status
job1 = MagicMock()
job1.id = uuid.uuid4()
job1.status = "COMPLETED"
job1.model_id = uuid.uuid4()

job2 = MagicMock()
job2.id = uuid.uuid4()
job2.status = "COMPLETED"
job2.model_id = uuid.uuid4()

scan1 = MagicMock()
scan1.governance_score = 95.0
scan1.gate_status = "CERTIFIED"
scan1.checks_run = ["drift", "fairness"]

scan2 = MagicMock()
scan2.governance_score = 45.0
scan2.gate_status = "FAILED"
scan2.checks_run = ["drift"]

async def override_get_db():
    mock_db = AsyncMock()
    
    async def mock_execute(query):
        query_str = str(query)
        mock_result = MagicMock()
        
        try:
            compiled = str(query.compile(compile_kwargs={"literal_binds": True}))
        except Exception:
            compiled = ""
            
        if "jobs" in query_str and "submission_token" in query_str:
            if token1 in compiled:
                mock_result.scalars.return_value.first.return_value = job1
            elif token2 in compiled:
                mock_result.scalars.return_value.first.return_value = job2
            else:
                mock_result.scalars.return_value.first.return_value = None
                
        elif "scan_records" in query_str and "job_id" in query_str:
            if str(job1.id) in compiled:
                mock_result.scalars.return_value.first.return_value = scan1
            elif str(job2.id) in compiled:
                mock_result.scalars.return_value.first.return_value = scan2
            else:
                mock_result.scalars.return_value.first.return_value = None
                
        return mock_result

    mock_db.execute = mock_execute
    yield mock_db

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

def test_concurrent_jobs_deterministic_polling():
    """Submit 2 jobs and ensure their unique submission_tokens route to their own exact results."""
    
    # Poll job 1
    response1 = client.get(f"/api/v1/gate/result/{token1}")
    assert response1.status_code == 200, response1.text
    res1 = response1.json()
    assert res1["status"] == "COMPLETED"
    assert res1["model_id"] == str(job1.model_id)
    assert res1["score"] == 95.0
    assert res1["verdict"] == "CERTIFIED"
    assert res1["breach_count"] == 2
    
    # Poll job 2
    response2 = client.get(f"/api/v1/gate/result/{token2}")
    assert response2.status_code == 200, response2.text
    res2 = response2.json()
    assert res2["status"] == "COMPLETED"
    assert res2["model_id"] == str(job2.model_id)
    assert res2["score"] == 45.0
    assert res2["verdict"] == "FAILED"  
    assert res2["breach_count"] == 1
    
    # Ensure no collision
    assert res1["model_id"] != res2["model_id"]
