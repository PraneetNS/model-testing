"""Quick API scan to check all major endpoints for errors."""
import httpx
import asyncio
import json

BASE = "http://localhost:8000"
API_KEY = "mlg_PeNfpwQSOtJkWr1Tow62Kr5luLuEugGi"
HEADERS = {"X-API-Key": API_KEY}

GET_ENDPOINTS = [
    "/",
    "/health",
    "/api/health/db",
    "/health/storage",
    "/health/worker",
    "/api/v1/models",
    "/api/v1/history",
    "/api/v1/alerts/rules",
    "/api/v1/alerts/events",
    "/api/v1/policies",
    "/api/v1/policies/active",
    "/api/v1/enterprise/models",
    "/api/v1/enterprise/summary",
    "/api/v1/enterprise/scans",
    "/api/v1/enterprise/audit-logs",
    "/api/v1/governance/status",
    "/api/v1/llm/history",
    "/api/v1/predictions/logs",
    "/api/v1/predictions/stats",
    "/api/v1/drift/health",
    "/api/v1/performance/health",
    "/api/v1/preflight/health",
    "/api/v1/contracts",
    "/api/v1/deployments",
    "/api/v1/experiments",
    "/api/v1/datasets",
    "/api/v1/feed",
    "/api/v1/notifications/config",
    "/api/v1/stream/models",
    "/api/v1/orgs",
    "/api/auth/keys",
    "/api/audit-log",
    "/api/tasks/dlq",
    "/api/plugins/available",
    "/api/v1/security/scans",
    "/api/v1/audit-logs",
    "/api/v1/behavior/scenarios",
    "/api/v1/monitoring/predictions",
    "/api/v1/monitoring/predictions/trends",
    "/api/v1/ci/integrations",
    "/api/v1/deployments/environments",
]

async def scan():
    results = {"ok": [], "errors": [], "warnings": []}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for ep in GET_ENDPOINTS:
            try:
                r = await client.get(f"{BASE}{ep}", headers=HEADERS)
                if r.status_code == 200:
                    results["ok"].append(f"✓ {ep} → 200")
                elif r.status_code in (401, 403):
                    results["errors"].append(f"✗ AUTH {ep} → {r.status_code}: {r.text[:100]}")
                elif r.status_code == 422:
                    results["warnings"].append(f"⚠ VALIDATION {ep} → 422: {r.text[:100]}")
                elif r.status_code == 500:
                    results["errors"].append(f"✗ SERVER ERROR {ep} → 500: {r.text[:200]}")
                elif r.status_code == 404:
                    results["warnings"].append(f"⚠ NOT FOUND {ep} → 404")
                else:
                    results["warnings"].append(f"⚠ {ep} → {r.status_code}: {r.text[:100]}")
            except Exception as e:
                results["errors"].append(f"✗ EXCEPTION {ep}: {e}")
    
    print("\n=== ✓ OK ENDPOINTS ===")
    for r in results["ok"]:
        print(r)
    
    print("\n=== ⚠ WARNINGS ===")
    for r in results["warnings"]:
        print(r)
    
    print("\n=== ✗ ERRORS ===")
    for r in results["errors"]:
        print(r)
    
    print(f"\n=== SUMMARY: {len(results['ok'])} OK, {len(results['warnings'])} warnings, {len(results['errors'])} errors ===")

asyncio.run(scan())
