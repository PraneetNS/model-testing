import requests
url = "http://127.0.0.1:8001/predict"
payload = {"annual_income": 85000, "credit_score": 720}
try:
    response = requests.post(url, json=payload)
    print(response.status_code)
    print(response.json())
except Exception as e:
    print(f"Error: {e}")
