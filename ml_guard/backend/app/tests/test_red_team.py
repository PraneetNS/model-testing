import pytest
from unittest.mock import MagicMock, patch
from ml_guard.core.red_team_scheduler import run_red_team_profile

class MockSandbox:
    def __init__(self):
        self.predict = MagicMock(return_value={"output": [0.1, 0.9], "type": "probability"})

def test_exhaustive_profile_coverage():
    """Assert exhaustive profile calls all major attack types."""
    sandbox = MockSandbox()
    results = run_red_team_profile("exhaustive", sandbox, {"model_id": "test-uuid"})
    
    assert results["profile"] == "exhaustive"
    attacks = results["attack_results"]
    
    # Check for all 5 required attack types/probes in exhaustive profile
    assert "fgsm" in attacks
    assert "pgd" in attacks
    assert "model_extraction" in attacks
    assert "membership_inference" in attacks
    assert "prompt_injection" in attacks
    
    # Accuracy check within probes
    assert attacks["model_extraction"]["risk"] is True
    assert attacks["membership_inference"]["vulnerable"] is True

def test_quick_profile_efficiency():
    """Assert quick profile only runs a subset of attacks."""
    sandbox = MockSandbox()
    results = run_red_team_profile("quick", sandbox, {"model_id": "test-uuid"})
    
    attacks = results["attack_results"]
    assert "fgsm" in attacks
    assert "prompt_injection" in attacks
    assert "model_extraction" not in attacks

@pytest.mark.asyncio
async def test_regression_detection_logic():
    """
    Test the regression logic that fires an alert if 
    robustness drops by > 5 points.
    """
    from ml_guard.backend.app.workers.tasks import run_red_team_task
    
    # Mocking the database and sandbox managers
    with patch("app.db.session.SessionLocal") as mock_session_cls, \
         patch("ml_guard.sandbox.sandbox_runner.ModelSandbox.create_sandbox") as mock_create, \
         patch("ml_guard.core.red_team_scheduler.run_red_team_profile") as mock_run_profile:
        
        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db
        
        # Mocking finding the schedule with baseline 90.0
        mock_sched = MagicMock()
        mock_sched.baseline_robustness_score = 90.0
        
        # This is a bit complex due to async session execution in tasks.py
        # We simulate the result of: await db.execute(select(RedTeamSchedule)...)
        mock_exec_result = MagicMock()
        mock_exec_result.scalars.return_value.first.return_value = mock_sched
        mock_db.execute.return_value = mock_exec_result
        
        # Mock red team run returning 80.0 (10 point drop > 5 threshold)
        mock_run_profile.return_value = {
            "robustness_score": 80.0,
            "attack_results": {"fgsm": {}}
        }
        
        # Run the task
        from ml_guard.backend.app.workers.tasks import run_red_team_task
        import asyncio
        asyncio.run(run_red_team_task("test-model-id", "standard"))
        
        # Verify that a SecurityAlert was added to the session
        # We check the arguments passed to db.add()
        added_objects = [call.args[0] for call in mock_db.add.call_args_list]
        from app.db.models import SecurityAlert, RedTeamRun
        
        alert_found = any(isinstance(obj, SecurityAlert) and obj.alert_type == "adversarial_regression" for obj in added_objects)
        run_found = any(isinstance(obj, RedTeamRun) for obj in added_objects)
        
        assert alert_found is True
        assert run_found is True
