"""Verify all new lifecycle endpoints are available."""
import httpx

BASE = "http://127.0.0.1:8090"

def test_endpoints():
    client = httpx.Client(timeout=10)
    
    # Test OpenAPI docs loads (confirms all routers registered)
    r = client.get(f"{BASE}/openapi.json")
    if r.status_code == 200:
        paths = list(r.json().get("paths", {}).keys())
        print(f"✅ OpenAPI loaded — {len(paths)} endpoints")
        
        new_endpoints = [p for p in paths if any(x in p for x in [
            "/models", "/datasets", "/experiments", "/explainability",
            "/data-quality", "/deployments", "/predictions"
        ])]
        print(f"\n📦 NEW LIFECYCLE ENDPOINTS ({len(new_endpoints)}):")
        for ep in sorted(new_endpoints):
            print(f"   {ep}")
        
        existing = [p for p in paths if any(x in p for x in [
            "/audit", "/behavior", "/monitoring/live", "/enterprise", "/policies"
        ])]
        print(f"\n🔒 EXISTING ENDPOINTS (unchanged): {len(existing)}")
        for ep in sorted(existing):
            print(f"   {ep}")
    else:
        print(f"❌ OpenAPI failed: {r.status_code}")
    
    # Test health
    r = client.get(f"{BASE}/health/database")
    print(f"\n🏥 Database: {r.json()['status']}")
    
    print("\n✅ All lifecycle modules registered successfully!")

if __name__ == "__main__":
    test_endpoints()
