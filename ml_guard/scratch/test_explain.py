import requests
import time
import os

BASE_URL = "http://127.0.0.1:8000/api/v1"
API_KEY = "mlg_PeNfpwQSOtJkWr1Tow62Kr5luLuEugGi"

headers = {
    "X-API-Key": API_KEY
}

def test_explainability():
    print("Triggering Explainability Computation...")
    
    model_path = "../backend/fair_loan_model.pkl"
    dataset_path = "../backend/fair_loan_test.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(dataset_path):
        print("Missing required files.")
        return

    files = {
        "model_file": ("fair_loan_model.pkl", open(model_path, "rb"), "application/octet-stream"),
        "dataset_file": ("fair_loan_test.csv", open(dataset_path, "rb"), "text/csv"),
    }
    
    data = {
        "model_id": "00000000-0000-0000-0000-000000000001",
        "max_samples": 10
    }
    
    try:
        response = requests.post(f"{BASE_URL}/explainability/compute", headers=headers, data=data, files=files)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            model_id = result.get("model_id")
            print(f"Task ID: {task_id}")
            print("Waiting for Celery worker to finish...")
            
            # Polling for results
            for _ in range(15):
                time.sleep(2)
                res = requests.get(f"{BASE_URL}/explainability/{model_id}", headers=headers)
                if res.status_code == 200:
                    print("Explainability Results found!")
                    # print(res.json())
                    return True
                else:
                    print("Still waiting for results...")
            
            print("Timeout waiting for results.")
            return False
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

test_explainability()
