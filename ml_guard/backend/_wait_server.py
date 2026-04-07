import requests
import time

for i in range(15):
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if r.status_code == 200:
            print("Backend ready:", r.json())
            break
    except Exception:
        pass
    time.sleep(2)
    print(f"Waiting... {i+1}/15")
