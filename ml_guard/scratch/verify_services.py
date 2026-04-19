import requests

def check_service(url, name):
    try:
        response = requests.get(url, timeout=5)
        print(f"[{name}] {url} - Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"[{name}] Response: {response.text[:200]}...")
        return True
    except Exception as e:
        print(f"[{name}] {url} - FAILED: {e}")
        return False

print("Checking ML Guard Services...")
backend_ok = check_service("http://127.0.0.1:8000/health", "Backend Health")
root_ok = check_service("http://127.0.0.1:8000/", "Backend Root")
frontend_ok = check_service("http://localhost:3000", "Frontend")

if backend_ok and root_ok and frontend_ok:
    print("\nALL SERVICES ARE RESPONDING!")
else:
    print("\nSOME SERVICES ARE DOWN OR UNRESPONSIVE.")
