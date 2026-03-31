import requests
import json
import sys
import os

# Base configuration for the ML Guard structural rescue validation
BASE = "http://127.0.0.1:8000"

# Attempt to load credentials from the dev seed environment
API_KEY = None
MODEL_ID = None

env_dev_path = os.path.join(os.path.dirname(__file__), ".env.dev")
if os.path.exists(env_dev_path):
    with open(env_dev_path) as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                if key == "DEV_API_KEY":
                    API_KEY = val
                if key == "DEV_MODEL_ID":
                    MODEL_ID = val

if not API_KEY:
    print("(!) ERROR: No API key found in .env.dev.")
    print("    Please run: python seed_dev.py first.")
    sys.exit(1)

# Persistent headers for all authorized requests
HEADERS = {"X-API-Key": API_KEY}
print(f"[*] Authenticating with API Key: {API_KEY[:20]}...")
print(f"[*] Targeting Model ID:        {MODEL_ID}")

def test_health():
    print("\n-- Health Checks ------------------")
    for endpoint in ["/health", "/health/database", "/health/worker"]:
        try:
            r = requests.get(f"{BASE}{endpoint}", timeout=5)
            status = "OK" if r.status_code == 200 else f"ERR ({r.status_code})"
            print(f"{status} {endpoint}: {r.json()}")
        except Exception as e:
            print(f"(!) {endpoint} Check Failed: {e}")

def test_governance_score():
    print("\n-- Governance Score ---------------")
    r = requests.get(
        f"{BASE}/api/v1/governance/{MODEL_ID}/score",
        headers=HEADERS
    )
    print(f"  GET /governance/.../score -> {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"OK Overall score: {data.get('overall_score')}")
        print(f"OK Live score:    {data.get('live_score')}")
        print(f"OK Verdict:       {data.get('verdict')}")
        return data
    else:
        print(f"  Response: {r.text[:200]}")
        return None

def test_generate_certificate():
    print("\n-- Certificate Generation ---------")
    r = requests.post(
        f"{BASE}/api/v1/governance/{MODEL_ID}/certify",
        json={"force_regenerate": False},
        headers=HEADERS
    )
    print(f"  POST /governance/.../certify -> {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        cert_hash = data.get("cert_hash")
        print(f"OK cert_hash: {cert_hash[:32]}...")
        print(f"OK Verdict:   {data.get('verdict')}")
        print(f"OK Score:     {data.get('overall_score')}")
        return cert_hash
    else:
        print(f"  Response: {r.text[:200]}")
        return None

def test_verify_certificate(cert_hash: str):
    print("\n-- Certificate Verification -------")
    # PUBLIC verification point - no auth header required
    r = requests.get(
        f"{BASE}/api/v1/governance/verify/{cert_hash}"
    )
    print(f"  GET /governance/verify/... -> {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"OK Valid:            {data.get('valid')}")
        print(f"OK Still compliant:  {data.get('still_compliant')}")
        print(f"OK Drift events:     {data.get('drift_events_since_issue')}")
        print(f"OK Message: {data.get('message')}")
    else:
        print(f"  Response: {r.text[:200]}")

def test_ci_gate():
    print("\n-- CI/CD Policy Gate --------------")
    r = requests.post(
        f"{BASE}/api/v1/governance/{MODEL_ID}/gate",
        json={
            "policy_config": {
                "min_governance_score": 60,
                "max_psi": 0.25,
                "min_accuracy": 0.70
            }
        },
        headers=HEADERS
    )
    print(f"  POST /governance/.../gate -> {r.status_code}")
    if r.status_code in [200, 422]:
        data = r.json()
        print(f"OK Passed:   {data.get('passed')}")
        print(f"OK Score:    {data.get('score')}")
        failures = data.get('failures', [])
        if failures:
            print(f"  Failures: {failures}")
    else:
        print(f"  Response: {r.text[:200]}")

def test_ingest_prediction():
    print("\n-- Prediction Ingestion -----------")
    r = requests.post(
        f"{BASE}/api/v1/ingest/predict",
        json={
            "model_id": MODEL_ID,
            "features": {
                "age": 34,
                "income": 75000,
                "tenure": 5
            },
            "prediction": "1",
            "prediction_proba": 0.87,
            "latency_ms": 23.4,
            "environment": "production"
        },
        headers=HEADERS
    )
    print(f"  POST /ingest/predict -> {r.status_code}")
    if r.status_code in [200, 201]:
        data = r.json()
        print(f"OK Log ID: {data.get('log_id')}")
        print(f"OK Status: {data.get('status')}")
    else:
        print(f"  Response: {r.text[:200]}")

if __name__ == "__main__":
    print("-----------------------------------")
    print("  ML Guard v7.2 -- E2E Test Suite   ")
    print("-----------------------------------")
    
    try:
        test_health()
        test_governance_score()
        cert_hash = test_generate_certificate()
        if cert_hash:
            test_verify_certificate(cert_hash)
        test_ci_gate()
        test_ingest_prediction()
        
        print("\n-----------------------------------")
        print("  OK E2E Test Complete              ")
        print("-----------------------------------")
    
    except requests.ConnectionError:
        print("\n(!) ERROR: Cannot connect to ML Guard server.")
        print("    Ensure the API is running at: http://127.0.0.1:8000")
        sys.exit(1)
