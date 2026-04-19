import requests
import uuid

BASE_URL = "http://127.0.0.1:8000/api/v1"
API_KEY = "mlg_PeNfpwQSOtJkWr1Tow62Kr5luLuEugGi"

headers = {
    "X-API-Key": API_KEY
}

def test_insurance():
    print("Testing Insurance Scoring...")
    
    try:
        res = requests.get(f"{BASE_URL}/models", headers=headers)
        data = res.json()
        
        items = data.get("items", [])
        
        model_id = None
        for m in items:
            if m.get("name") == "TestLoanModel":
                model_id = m.get("id") or m.get("model_id")
                break
        
        if not model_id:
            print("TestLoanModel not found.")
            return

        print(f"Found Model ID: {model_id}")
        
        res = requests.get(f"{BASE_URL}/governance/{model_id}/insurance-score", headers=headers)
        print(f"Status: {res.status_code}")
        if res.status_code == 200:
            report = res.json()
            print("Insurance Report Successful!")
            print(f"Tier: {report.get('tier')}")
            print(f"Total Score: {report.get('total_score')}")
            return True
        else:
            print(f"Error: {res.text}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

test_insurance()
