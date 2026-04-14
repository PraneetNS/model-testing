import requests
import json
import random
import time
import uuid
import sys

BACKEND_URL = "http://localhost:8000/api/v1"

def get_real_model_id():
    """Fetch the first available model_id from the registry."""
    try:
        # Use models endpoint to find valid IDs
        resp = requests.get(f"{BACKEND_URL}/models")
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("items", [])
            if models:
                mid = models[0].get("model_id") or models[0].get("id")
                print(f"Found model in registry: {mid}")
                return mid
    except Exception as e:
        print(f"Could not fetch models: {e}")
    return None

def simulate_predictions(model_id, count=50):
    print(f"Simulating {count} predictions for model {model_id}...")
    
    for i in range(count):
        # Generate some random feature data
        payload = {
            "model_id": model_id,
            "features": {
                "feature1": random.uniform(0, 1),
                "feature2": random.uniform(0, 1),
                "feature3": random.uniform(0, 1),
                "age": random.randint(18, 90),
                "income": random.uniform(20000, 150000)
            },
            "prediction": random.choice([0, 1]),
            "prediction_proba": random.uniform(0.5, 0.99),
            "latency_ms": random.uniform(10, 250),
            "environment": "production"
        }
        
        try:
            resp = requests.post(f"{BACKEND_URL}/ingest/predict", json=payload)
            if resp.status_code == 202:
                print(f"  [{i+1}/{count}] Logged prediction: {resp.json().get('log_id')}")
            else:
                print(f"  Failed: {resp.text}")
        except Exception as e:
            print(f"  Error: {e}")
        
        time.sleep(0.05) 

def generate_drift_events(model_id):
    print("Generating a drift event...")
    for i in range(10):
        payload = {
            "model_id": model_id,
            "features": {
                "feature1": random.uniform(5, 10), # SHIFTED!
                "feature2": random.uniform(5, 10), # SHIFTED!
                "feature3": random.uniform(0, 1),
                "age": random.randint(18, 90),
                "income": random.uniform(500000, 1000000) # SHIFTED!
            },
            "prediction": 1,
            "prediction_proba": 0.99,
            "latency_ms": 300,
            "environment": "production"
        }
        requests.post(f"{BACKEND_URL}/ingest/predict", json=payload)

if __name__ == "__main__":
    model_id = get_real_model_id()
    if not model_id:
        print("ERROR: No models found in registry. Please register a model first.")
        sys.exit(1)
        
    simulate_predictions(model_id, count=20)
    generate_drift_events(model_id)
    print("\nDone! Now go to the 'Observe' tab and refresh the 'Global Feed'.")
