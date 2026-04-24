import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import FastAPI
import sys
import os
from unittest.mock import patch, MagicMock

os.environ["MLGUARD_ENV"] = "development"
import app.core.config
# Avoid Pydantic AttributeError by not setting undefined fields
# The database fix will now use DATABASE_URL as fallback

from app.api.v1.endpoints.compliance import router as compliance_router
from unittest.mock import patch, MagicMock
from ml_guard.core.compliance_packs.fda_ai_guidance import generate_fda_ai_guidance_report
from app.db.models import Model

@pytest.mark.anyio
async def test_fda_ai_guidance_failure():
    # Mock db session
    mock_db = MagicMock(spec=AsyncSession)
    
    # Mock model
    mock_model = MagicMock(spec=Model)
    mock_model.parent_model_id = None
    mock_model.version = 2
    import datetime
    mock_model.created_at = datetime.datetime.utcnow() - datetime.timedelta(days=1) # Won't pass 30 days
    
    # Mock get
    async def mock_get(*args, **kwargs):
        return mock_model
    
    # Mock execute
    async def mock_execute(*args, **kwargs):
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [] # No drift checks, no shaps, no reports
        return mock_result
        
    mock_db.get = mock_get
    mock_db.execute = mock_execute
    
    report = await generate_fda_ai_guidance_report("test-model-1", mock_db)
    
    assert len(report) == 3
    assert report[0]["title"] == "Predetermined Change Control Plan"
    assert report[0]["status"] == "fail"
    
    assert report[1]["title"] == "Real-World Performance Monitoring"
    assert report[1]["status"] == "fail"
    
    assert report[2]["title"] == "Transparency and Disclosure"
    assert report[2]["status"] == "fail"

app = FastAPI()
app.include_router(compliance_router, prefix="/api/compliance")
client = TestClient(app)

def test_compliance_pdf_generator():
    # Mock the get_compliance_pack function directly
    with patch("app.api.v1.endpoints.compliance.get_compliance_pack") as mock_get_pack:
        async def mock_pack(*args, **kwargs):
            return {
                "status": "partial",
                "score": 50,
                "checks": [
                    {"article": "Test", "title": "Test Check", "status": "pass", "evidence": "test", "remediation": ""}
                ]
            }
        mock_get_pack.side_effect = mock_pack
        
        # We also need to mock the db get for Model
        with patch("sqlalchemy.ext.asyncio.AsyncSession.get") as mock_get:
            async def mock_db_get(*args, **kwargs):
                return MagicMock(name="MockModel")
            mock_get.side_effect = mock_db_get
            
            response = client.get("/api/compliance/test-model-1/pack/sr_11_7/pdf")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert "X-Report-Hash" in response.headers
            assert len(response.content) > 0 # PDF bytes or fallback message
