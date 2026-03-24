
import requests
import os

API_URL = "http://localhost:8000"
# Use a real key if possible, but let's just trigger it as if we are the CLI
# Actually, I'll just check if the model registry works first.

model_path = "fair_loan_model.pkl"
data_path = "fair_loan_test.csv"
api_key = "mlg_QpAlNI9_NQYbfFIQA9hHpMujRthXpIogfxb4OMtTEW4"

files = {
    "model_file": (os.path.basename(model_path), open(model_path, 'rb')),
    "val_file": (os.path.basename(data_path), open(data_path, 'rb'))
}
data = {
    "model_name": "TestModel_" + os.urandom(4).hex(),
    "label_col": "target",
}
headers = {"X-Api-Key": api_key}

print("Triggering audit...")
res = requests.post(f"{API_URL}/api/v1/audit/run", headers=headers, files=files, data=data)
print(f"Response: {res.status_code}")
print(f"Body: {res.text}")

if res.status_code == 200:
    job_id = res.json().get("job_id")
    import time
    for _ in range(20):
        s_res = requests.get(f"{API_URL}/api/v1/jobs/{job_id}", headers=headers)
        status = s_res.json().get("status")
        print(f"Status: {status}")
        if status in ["COMPLETED", "FAILED"]:
            print(f"Final: {s_res.text}")
            break
        time.sleep(2)
