import requests
import json
import sys

BASE = "http://127.0.0.1:8000"

def test_health():
    print("\n-- Health Checks ------------------")
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    print(f"OK /health: {r.json()}")
    
    r = requests.get(f"{BASE}/health/database")
    assert r.status_code == 200
    print(f"OK /health/database: {r.json()}")
    
    r = requests.get(f"{BASE}/health/worker")
    print(f"OK /health/worker: {r.json()}")

def test_register_model():
    print("\n-- Model Registration -------------")
    payload = {
        "name": "TestChurnModel-v1",
        "provider": "sklearn",
        "metadata_json": {
            "task": "classification",
            "framework": "scikit-learn"
        }
    }
    # Try model registry endpoint
    r = requests.post(f"{BASE}/api/v1/models", json=payload)
    print(f"  POST /api/v1/models -> {r.status_code}")
    if r.status_code in [200, 201]:
        model = r.json()
        model_id = model.get("id") or model.get("model_id")
        print(f"OK Model registered: {model_id}")
        return model_id
    else:
        print(f"  Response: {r.text[:200]}")
        return None

def test_governance_score(model_id: str):
    print("\n-- Governance Score ---------------")
    r = requests.get(
        f"{BASE}/api/v1/governance/{model_id}/score"
    )
    print(f"  GET /governance/{model_id}/score -> {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"OK Overall score: {data.get('overall_score')}")
        print(f"OK Live score: {data.get('live_score')}")
        print(f"OK Verdict: {data.get('verdict')}")
        return data
    else:
        print(f"  Response: {r.text[:200]}")
        return None

def test_generate_certificate(model_id: str):
    print("\n-- Certificate Generation ---------")
    r = requests.post(
        f"{BASE}/api/v1/governance/{model_id}/certify",
        json={"force_regenerate": False}
    )
    print(f"  POST /governance/{model_id}/certify -> {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        cert_hash = data.get("cert_hash")
        print(f"OK Certificate generated: {cert_hash}")
        print(f"OK Verdict: {data.get('verdict')}")
        print(f"OK Score: {data.get('overall_score')}")
        return cert_hash
    else:
        print(f"  Response: {r.text[:200]}")
        return None

def test_verify_certificate(cert_hash: str):
    print("\n-- Certificate Verification -------")
    r = requests.get(
        f"{BASE}/api/v1/governance/verify/{cert_hash}"
    )
    print(f"  GET /governance/verify/{cert_hash[:16]}... -> {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"OK Valid: {data.get('valid')}")
        print(f"OK Still compliant: {data.get('still_compliant')}")
        print(f"OK Drift events since issue: {data.get('drift_events_since_issue')}")
        print(f"OK Message: {data.get('message')}")
    else:
        print(f"  Response: {r.text[:200]}")

def test_ci_gate(model_id: str):
    print("\n-- CI/CD Policy Gate --------------")
    r = requests.post(
        f"{BASE}/api/v1/governance/{model_id}/gate",
        json={
            "policy_config": {
                "min_governance_score": 60,
                "max_psi": 0.25,
                "min_accuracy": 0.70
            }
        }
    )
    print(f"  POST /governance/{model_id}/gate → {r.status_code}")
    if r.status_code in [200, 422]:
        data = r.json()
        print(f"✓ Passed: {data.get('passed')}")
        print(f"✓ Score: {data.get('score')}")
        failures = data.get('failures', [])
        if failures:
            print(f"  Failures: {failures}")
    else:
        print(f"  Response: {r.text[:200]}")

if __name__ == "__main__":
    print("-----------------------------------")
    print("  ML Guard v7.2 -- E2E Test Suite   ")
    print("-----------------------------------")
    
    try:
        test_health()
        model_id = test_register_model()
        
        if model_id:
            score_data = test_governance_score(model_id)
            cert_hash = test_generate_certificate(model_id)
            
            if cert_hash:
                test_verify_certificate(cert_hash)
            
            test_ci_gate(model_id)
        
        print("\n-----------------------------------")
        print("  OK E2E Test Complete              ")
        print("-----------------------------------")
    
    except requests.ConnectionError:
        print("\n✗ Cannot connect to server.")
        print("  Start it with:")
        print("  uvicorn app.main:app --host 127.0.0.1 --port 8000")
        sys.exit(1)
