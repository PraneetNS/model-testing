"""Comprehensive verification of all ML Guard API endpoints."""
import httpx
import asyncio

BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": "mlg_PeNfpwQSOtJkWr1Tow62Kr5luLuEugGi"}

def _len_or_total(data):
    if isinstance(data, list): return len(data)
    if isinstance(data, dict): return data.get("total") or len(data.get("items", []))
    return "?"

async def chk(client, path, label=None):
    r = await client.get(f"{BASE}{path}", headers=HEADERS)
    data = r.json()
    count = _len_or_total(data)
    print(f"{'OK' if r.status_code == 200 else 'ERR'} [{r.status_code}] {label or path} -> count/total={count}")
    if r.status_code not in (200, 422, 405):
        print(f"     >> {str(data)[:150]}")
    return r.status_code == 200


async def run():
    passed = failed = 0
    async with httpx.AsyncClient(timeout=8.0) as client:
        tests = [
            ("/health",                           "Health"),
            ("/api/health/db",                    "DB health"),
            ("/api/v1/models",                    "Models list"),
            ("/api/v1/enterprise/summary",        "Enterprise summary"),
            ("/api/v1/governance/status",         "Governance status"),
            ("/api/v1/history",                   "Scan history"),
            ("/api/v1/alerts/rules",              "Alert rules"),
            ("/api/v1/alerts/events",             "Alert events"),
            ("/api/v1/datasets",                  "Datasets"),
            ("/api/v1/experiments",               "Experiments"),
            ("/api/v1/deployments",               "Deployments"),
            ("/api/v1/deployments/environments",  "Deploy environments"),
            ("/api/v1/predictions/logs",          "Prediction logs"),
            ("/api/v1/predictions/stats",         "Prediction stats"),
            ("/api/v1/enterprise/audit-logs",     "Enterprise audit logs"),
            ("/api/auth/keys",                    "Auth keys list"),
            ("/api/audit-log",                    "Audit log"),
            ("/api/v1/policies",                  "Policies"),
            ("/api/v1/policies/active",           "Active policies"),
            ("/api/v1/llm/history",               "LLM history"),
            ("/api/v1/stream/models",             "Stream models"),
            ("/api/v1/ci/integrations",           "CI integrations"),
            ("/api/v1/enterprise/models",         "Enterprise models"),
            ("/api/v1/enterprise/scans",          "Enterprise scans"),
            ("/api/v1/behavior/scenarios",        "Behavior scenarios"),
            ("/api/v1/security/scans",            "Security scans"),
            ("/api/v1/audit-logs",                "Audit logs"),
            ("/api/v1/drift/health",              "Drift health"),
            ("/api/v1/performance/health",        "Performance health"),
            ("/api/v1/preflight/health",          "Preflight health"),
            ("/api/v1/feed",                      "Feed"),
            ("/api/v1/orgs",                      "Organizations"),
            ("/api/plugins/available",            "Plugins available"),
            ("/api/tasks/dlq",                    "Tasks DLQ"),
        ]
        for path, label in tests:
            ok = await chk(client, path, label)
            if ok: passed += 1
            else: failed += 1

    print(f"\n=== RESULT: {passed} passed, {failed} failed ===")

asyncio.run(run())
