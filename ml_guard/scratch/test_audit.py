import requests
import os

BASE_URL = "http://127.0.0.1:8000/api/v1"
API_KEY = "mlg_PeNfpwQSOtJkWr1Tow62Kr5luLuEugGi"

headers = {
    "X-API-Key": API_KEY
}

def trigger_audit():
    print("Triggering Audit Scan (Multipart)...")
    
    # Paths to local files in backend/
    model_path = "../backend/fair_loan_model.pkl"
    dataset_path = "../backend/fair_loan_test.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(dataset_path):
        print("Missing required files in backend directory.")
        return

    files = {
        "model_file": ("fair_loan_model.pkl", open(model_path, "rb"), "application/octet-stream"),
        "val_file": ("fair_loan_test.csv", open(dataset_path, "rb"), "text/csv"),
        "train_file": ("fair_loan_test.csv", open(dataset_path, "rb"), "text/csv"),
    }
    
    data = {
        "model_name": "TestLoanModel",
        "label_col": "loan_status", # Based on fair_loan_test.csv usually
        "selected": ["drift", "performance", "fairness"]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/audit/run", headers=headers, data=data, files=files)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("Scan Successful!")
            print(f"Governance Score: {result.get('governance', {}).get('governance_score')}")
            print(f"Risk Level: {result.get('risk_level')}")
            return result
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

trigger_audit()
