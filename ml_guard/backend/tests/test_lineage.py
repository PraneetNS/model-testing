import pytest
import uuid
import json
from unittest.mock import AsyncMock, MagicMock
from app.db.models import Model, ScanRecord
from ml_guard.core.lineage import build_lineage_graph
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import get_db

# Mock IDs
M1 = str(uuid.uuid4())
M2 = str(uuid.uuid4())
M3 = str(uuid.uuid4())

# Set up mock DB
async def override_get_db():
    mock_db = AsyncMock()
    # State
    models = {
        M1: Model(id=M1, name="root", version=1, parent_model_id=None),
        M2: Model(id=M2, name="child1", version=2, parent_model_id=M1),
        M3: Model(id=M3, name="child2", version=3, parent_model_id=M2),
    }

    async def mock_execute(query):
        query_str = str(query)
        mock_result = MagicMock()
        
        try:
            compiled = str(query.compile(compile_kwargs={"literal_binds": True}))
        except Exception:
            compiled = ""
        
        if "WHERE models.id" in query_str:
             if M1 in compiled:
                 mock_result.scalars.return_value.first.return_value = models[M1]
             elif M2 in compiled:
                 mock_result.scalars.return_value.first.return_value = models[M2]
             elif M3 in compiled:
                 mock_result.scalars.return_value.first.return_value = models[M3]
             else:
                 mock_result.scalars.return_value.first.return_value = None
                 
        elif "WHERE scan_records.model_id" in query_str:
             if M2 in compiled:
                 res = ScanRecord(model_id=M2, results_json={"fairness_score": 90.0, "performance_score": 80.0})
             elif M1 in compiled:
                 res = ScanRecord(model_id=M1, results_json={"fairness_score": 60.0, "performance_score": 75.0})
             else:
                 res = None
             mock_result.scalars.return_value.first.return_value = res
             
        elif "models" in query_str and "parent_model_id" in query_str:
            # return all for graph
            mock_result.all.return_value = list(models.values())
            
        elif "scan_records" in query_str:
            # empty scan records
            mock_result.all.return_value = []
        return mock_result

    mock_db.execute = mock_execute
    # Simulate commit
    mock_db.commit = AsyncMock()
    yield mock_db

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_circular_reference_rejection():
    # Attempt to set M1's parent to M3. The mock DB has M1->M2->M3.
    # So tracing M3 parent goes M3->M2->M1. Setting M1 parent to M3 creates a circle.
    res = client.post(f"/api/lineage/{M1}/set-parent?parent_id={M3}")
    assert res.status_code == 400
    assert "circular" in res.json()["detail"].lower()


def test_diff_direction_labeling():
    # Diff M2 vs M1
    res = client.get(f"/api/lineage/{M2}/diff/{M1}")
    assert res.status_code == 200
    data = res.json()
    
    fairness_diff = next(d for d in data if d["dimension"] == "fairness")
    assert fairness_diff["score_a"] == 90.0
    assert fairness_diff["score_b"] == 60.0
    assert fairness_diff["delta"] == 30.0
    assert fairness_diff["direction"] == "improved"

    perf_diff = next(d for d in data if d["dimension"] == "performance")
    assert perf_diff["direction"] == "improved"
    assert perf_diff["delta"] == 5.0
    
    sec_diff = next(d for d in data if d["dimension"] == "security")
    assert sec_diff["direction"] == "unchanged"


def test_dag_serialization():
    # Test Graph endpoint (fetch DAG subset for M1)
    res = client.get(f"/api/lineage/{M1}/graph")
    assert res.status_code == 200
    data = res.json()
    
    # 3 nodes: M1, M2, M3 are fully connected
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 2
    
    edge_pairs = [(e["parent_id"], e["child_id"]) for e in data["edges"]]
    assert (M1, M2) in edge_pairs
    assert (M2, M3) in edge_pairs

