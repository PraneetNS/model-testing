import requests
import sys

def check_service(name, url):
    print(f"Checking {name} ({url})...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            print("[\033[92mONLINE\033[0m]")
            print(f"  Detail: {r.json()}")
        else:
            print(f"[\033[91mERROR {r.status_code}\033[0m]")
    except Exception as e:
        print(f"[\033[91mOFFLINE\033[0m]")
        print(f"  Error: {e}")

if __name__ == "__main__":
    print("=== ML Guard Cloud Connectivity Audit ===")
    # 1. API Health
    check_service("FastAPI Backend", "http://127.0.0.1:8000/health")
    
    # 2. Database (Neon)
    check_service("Neon PostgreSQL", "http://127.0.0.1:8000/health/database")
    
    # 3. Cache/Message Broker (Upstash Redis)
    check_service("Upstash Redis", "http://127.0.0.1:8000/health/worker")
    
    # 4. Object Storage (MinIO)
    check_service("Object Storage", "http://127.0.0.1:8000/health/storage")
    
    print("\nNext Step: Deploy to Render/Railway to enable the Dedicated Cloud MinIO service.")
