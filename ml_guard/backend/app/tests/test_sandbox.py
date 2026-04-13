import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import os
import shutil
from ml_guard.sandbox.sandbox_runner import ModelSandbox
from ml_guard.sandbox.attacks import square_attack

@patch("docker.from_env")
def test_sandbox_container_options(mock_docker_env):
    mock_client = MagicMock()
    mock_docker_env.return_value = mock_client
    
    # Mocking setup to avoid actual IO/Docker calls
    with patch("os.path.exists", return_value=True), \
         patch("shutil.copy"), \
         patch("builtins.open", MagicMock()), \
         patch("tempfile.mkdtemp", return_value="/tmp/test"):
        
        sandbox = ModelSandbox()
        
        # Mock container attributes
        mock_container = MagicMock()
        mock_container.ports = {'8000/tcp': [{'HostPort': '8001'}]}
        mock_client.containers.run.return_value = mock_container
        
        # Mock health check (request)
        with patch("requests.get"):
            sandbox.create_sandbox("test.pkl")
            
    # Assertions
    args, kwargs = mock_client.containers.run.call_args
    assert kwargs["network_disabled"] is True
    assert kwargs["mem_limit"] == "512m"
    assert "no-new-privileges:true" in kwargs["security_opt"]
    assert kwargs["cpu_quota"] == 50000

def test_square_attack_queries():
    # Mock predict function
    predict_fn = MagicMock(return_value={"output": 1})
    X = np.zeros((1, 5))
    
    # Square attack should return something
    adv_X = square_attack(predict_fn, X, n_queries=10)
    assert adv_X.shape == X.shape
    # Should have called predict multiple times
    # 1 for initial plus up to 10 for queries
    assert predict_fn.call_count > 0

@patch("os.path.exists", return_value=True)
def test_prompt_injection_suite_loading(mock_exists):
    from ml_guard.sandbox.attacks import prompt_injection_suite
    predict_fn = MagicMock(return_value={"output": "safe"})
    
    # Mock file reading
    m = MagicMock()
    m.__enter__.return_value = MagicMock(readlines=MagicMock(return_value=["payload1", "payload2"]))
    with patch("builtins.open", return_value=m.__enter__().return_value):
        results = prompt_injection_suite(predict_fn, "dummy.txt")
        assert len(results) > 0
        assert results[0]["payload"] == "payload1"
