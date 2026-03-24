import httpx
import time

def check_health():
    urls = {
        "DB": "http://127.0.0.1:8090/health/database",
        "Storage": "http://127.0.0.1:8090/health/storage"
    }
    
    for name, url in urls.items():
        try:
            print(f"Checking {name} at {url}...")
            r = httpx.get(url, timeout=10)
            print(f"{name} Status: {r.status_code}")
            print(f"{name} Response: {r.json()}")
        except Exception as e:
            print(f"{name} Check Failed: {e}")

if __name__ == "__main__":
    check_health()
