import pytest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
from ml_guard.core.aibom import generate_aibom

@pytest.fixture
def sample_model_file():
    fd, path = tempfile.mkstemp(suffix=".pkl")
    with os.fdopen(fd, 'wb') as f:
        f.write(b"dummy model data")
    yield path
    try:
        os.remove(path)
    except: pass

@pytest.fixture
def sample_dataset_file():
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, 'wb') as f:
        f.write(b"col1,col2\n1,2")
    yield path
    try:
        os.remove(path)
    except: pass

def test_aibom_generation_and_hash(sample_model_file, sample_dataset_file):
    metadata = {
        "model_id": "test-model-uuid",
        "model_name": "TestModel",
        "framework": "sklearn"
    }
    aibom = generate_aibom(sample_model_file, [sample_dataset_file], metadata)
    
    assert "aibom_hash" in aibom
    assert aibom["model_id"] == "test-model-uuid"
    assert len(aibom["training_datasets"]) == 1
    assert aibom["training_datasets"][0]["sha256_hash"] != ""
    assert aibom["base_model"]["sha256_hash"] != ""

@patch("ml_guard.core.aibom.check_osv_vulnerabilities")
def test_cve_detection_alert(mock_osv, sample_model_file):
    # Mocking OSV to return a vulnerability
    mock_osv.return_value = [{
        "package": "insecure-pkg",
        "version": "1.0.0",
        "cve_id": "CVE-2026-1234",
        "severity": "HIGH"
    }]
    
    # We need to mock importlib.metadata to return our insecure package
    with patch("importlib.metadata.distributions") as mock_dist:
        mock_pkg = MagicMock()
        mock_pkg.metadata = {"Name": "insecure-pkg"}
        mock_pkg.version = "1.0.0"
        mock_dist.return_value = [mock_pkg]
        
        metadata = {"model_id": "test-uuid"}
        aibom = generate_aibom(sample_model_file, [], metadata)
        
        assert len(aibom["cve_alerts"]) > 1 # insecure-pkg + whatever else is in env
        found = False
        for alert in aibom["cve_alerts"]:
            if alert["cve_id"] == "CVE-2026-1234":
                found = True
                break
        assert found

def test_hash_mismatch_detection(sample_model_file):
    metadata = {"model_id": "test-uuid"}
    aibom_original = generate_aibom(sample_model_file, [], metadata)
    original_hash = aibom_original["base_model"]["sha256_hash"]
    
    # Modify the model file
    with open(sample_model_file, "wb") as f:
        f.write(b"tampered model data")
        
    aibom_new = generate_aibom(sample_model_file, [], metadata)
    new_hash = aibom_new["base_model"]["sha256_hash"]
    
    assert original_hash != new_hash
